"""
将表格数据中的wh_what、wh_where、wh_when、wh_who、wh_why、wh_how提取出来，
存储进neo4j数据库中。
"""
from neo4j import GraphDatabase
import pandas as pd

#首先将what、where、when、who、why、how全部从xlsx表格中抽取出来，至于content内容暂时不抽取，如果需要，再作为chunk存储。
file_path = "../data/5w1h 1000.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
target_columns = ['wh_what','wh_where','wh_when','wh_who','wh_why','wh_how']
extracted_df = df[target_columns].dropna(how='all')



# Neo4j连接器 部署地址、用户名、密码、数据库名称
class Neo4jConnector:
    def __init__(self, uri, user, password, database):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_lifetime=30*60
        )
        self.database = database
        
    def close(self):
        self.driver.close()
        
    def test_connection(self):
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN '连接成功' AS status")
                return result.single()["status"] == "连接成功"
            print(f"成功连接到数据库 {self.database}")
            return True
        except Exception as e:
            print(f"连接失败: {str(e)}")
            if "does not exist" in str(e):
                print(f"请确认数据库 {self.database} 已创建且用户有访问权限")
            return False

# 初始化连接（新增database参数）
conn = Neo4jConnector(
    uri="bolt://172.20.31.130:7687",
    user="neo4j",
    password="neo4j@openspg",
    database="dawnjiang5w1h"  # 新增数据库名称参数
)
#如果连接不成功则退出
if not conn.test_connection():
    print("""
    neo4j数据库没有正常连接
    """)
    exit()

# 创建约束（确保在目标数据库执行）
def create_constraints():
    with conn.driver.session(database=conn.database) as session:
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:WhWhat) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:WhWhere) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:WhWhen) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:WhWho) REQUIRE n.name IS UNIQUE"
        ]
        for cql in constraints:
            session.run(cql)
        print("唯一约束创建完成")

create_constraints()  # 只在首次运行时需要

# 数据导入函数（带空值处理）
def import_to_neo4j(tx, row):
    # 清洗数据
    wh_what = str(row['wh_what']).strip() if pd.notna(row['wh_what']) else None
    if not wh_what:  # 跳过无效的主节点
        return
    
    # 创建主节点（使用MERGE确保唯一性）
    tx.run("""
        MERGE (what:WhWhat {name: $wh_what})
        SET what.why = coalesce($wh_why, what.why),
            what.how = coalesce($wh_how, what.how)
    """, parameters={
        'wh_what': wh_what,
        'wh_why': str(row['wh_why']).strip() if pd.notna(row['wh_why']) else None,  #将why作为属性传给事件节点
        'wh_how': str(row['wh_how']).strip() if pd.notna(row['wh_how']) else None   #将how作为属性传给how节点
    })
    
    # 关系处理函数
    def process_relation(col_name, label, rel_type):
        value = str(row[col_name]).strip() if pd.notna(row[col_name]) else None
        if not value:
            return
        # 创建目标节点
        tx.run(
            f"MERGE (n:{label} {{name: $value}})",
            parameters={'value': value}
        )
        # 创建关系（使用MERGE避免重复）
        tx.run(f"""
            MATCH (a:WhWhat {{name: $what}}), (b:{label} {{name: $value}})
            MERGE (a)-[r:{rel_type}]->(b)
        """, parameters={'what': wh_what, 'value': value})
    
    process_relation('wh_where', 'WhWhere', '地点')#将事件和where、when、who节点连接起来
    process_relation('wh_when', 'WhWhen', '时间')
    process_relation('wh_who', 'WhWho', '参与者')

# 增强版批量导入（带进度条）
from tqdm import tqdm

with conn.driver.session(database=conn.database) as session:
    # 分批处理数据
    batch_size = 100  #我的 mac air 差不多够用
    total = len(extracted_df)
    
    with tqdm(total=total, desc="导入进度") as pbar:
        for start in range(0, total, batch_size):
            batch = extracted_df.iloc[start:start+batch_size]
            session.execute_write(
                lambda tx: batch.apply(
                    lambda row: import_to_neo4j(tx, row), axis=1
                )
            )
            pbar.update(len(batch))

conn.close()
print(f"数据已成功导入到 {conn.database} 数据库")

