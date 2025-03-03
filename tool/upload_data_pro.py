"""
优化提取逻辑，即使title重复也能存入neo4j数据库。
将表格数据中的wh_what、wh_where、wh_when、wh_who、wh_why、wh_how提取出来，
存储进neo4j数据库中。
"""
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

# ================= 配置部分 =================
file_path = "../data/5w1h 1000.xlsx"
target_columns = ['wh_what', 'wh_where', 'wh_when', 'wh_who', 'wh_why', 'wh_how', 'title']

neo4j_config = {
    "uri": "bolt://172.20.129.190:7687",
    "user": "neo4j",
    "password": "neo4j@openspg",
    "database": "dawnjiang5w1h"
}
# ===========================================

class Neo4jConnector:
    def __init__(self, config):
        self.driver = GraphDatabase.driver(
            config["uri"],
            auth=(config["user"], config["password"]),
            max_connection_lifetime=30 * 60
        )
        self.database = config["database"]
    
    def close(self):
        self.driver.close()

def data_preprocessing(file_path):
    """数据预处理（不基于name+title去重）"""
    df = pd.read_excel(file_path, engine='openpyxl')
    extracted = df[target_columns].copy()
    
    # 仅过滤空值（不再去重）
    extracted = extracted.dropna(subset=['wh_what', 'title'], how='any')
    
    # 简单清洗
    for col in ['wh_what', 'title']:
        extracted[col] = extracted[col].astype(str).str.strip()
    
    print(f"有效数据记录数: {len(extracted)} (保留原始重复项)")
    return extracted

def remove_old_constraints(conn):
    """动态删除旧约束（关键修复）"""
    with conn.driver.session(database=conn.database) as session:
        # 查询所有约束
        result = session.run("SHOW CONSTRAINTS")
        constraints_to_drop = []
        
        for record in result:
            if (
                record["entityType"] == "NODE" 
                and record["labelsOrTypes"] == ["WhWhat"]
            ):
                constraints_to_drop.append(record["name"])
        
        # 删除目标约束
        for name in constraints_to_drop:
            session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            print(f"已删除约束: {name}")
        
        print("旧约束清理完成" if constraints_to_drop else "无需清理旧约束")

def create_new_constraints(conn):
    """创建新约束（仅其他节点保持唯一性）"""
    with conn.driver.session(database=conn.database) as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:WhWhere) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:WhWhen) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:WhWho) REQUIRE n.name IS UNIQUE")
        print("新约束创建完成")

def import_batch(tx, batch):
    """批量导入函数（允许重复）"""
    for _, row in batch.iterrows():
        try:
            wh_what = row['wh_what']
            title = row['title']
            if not wh_what or not title:
                continue

            # 创建主节点（直接CREATE允许重复）
            tx.run("""
                CREATE (what:WhWhat {
                    name: $wh_what,
                    title: $title,
                    why: $wh_why,
                    how: $wh_how
                })
            """, {
                'wh_what': wh_what,
                'title': title,
                'wh_why': row['wh_why'] if pd.notna(row['wh_why']) else None,
                'wh_how': row['wh_how'] if pd.notna(row['wh_how']) else None
            })

            # 处理关联节点（仍保持唯一性）
            def link_node(col, label, rel_type):
                value = str(row[col]).strip() if pd.notna(row[col]) else None
                if value:
                    tx.run(f"MERGE (n:{label} {{name: $value}})", {'value': value})
                    tx.run(f"""
                        MATCH (a:WhWhat {{name: $what, title: $title}})
                        MATCH (b:{label} {{name: $value}})
                        CREATE (a)-[:{rel_type}]->(b)
                    """, {'what': wh_what, 'title': title, 'value': value})

            link_node('wh_where', 'WhWhere', '地点')
            link_node('wh_when', 'WhWhen', '时间')
            link_node('wh_who', 'WhWho', '参与者')

        except Exception as e:
            print(f"\n错误行: {row.to_dict()}\n错误详情: {str(e)}")
            continue

if __name__ == "__main__":
    # 初始化
    conn = Neo4jConnector(neo4j_config)
    df_clean = data_preprocessing(file_path)
    
    # 约束管理
    remove_old_constraints(conn)
    create_new_constraints(conn)

    # 批量导入
    batch_size = 50
    total = len(df_clean)
    
    with conn.driver.session(database=conn.database) as session:
        with tqdm(total=total, desc="数据导入进度") as pbar:
            for start in range(0, total, batch_size):
                batch = df_clean.iloc[start:start + batch_size]
                session.execute_write(import_batch, batch)
                pbar.update(len(batch))
    
    conn.close()
    print(f"完成！总导入记录数: {len(df_clean)}（允许重复）")
