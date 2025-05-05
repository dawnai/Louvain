"""
优化提取逻辑，即使title重复也能存入neo4j数据库。
将表格数据中的what、where、when、who、why、how提取出来，并且通过逗号拆分who和where，分别作为节点存入。
存储进neo4j数据库中。
"""
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

# ================= 配置部分 =================
file_path = "./data/waite_to_neo4j/xlsx/1-5.xlsx"
target_columns = ['what', 'where', 'when', 'who', 'why', 'how', 'title','organization','news_id']

neo4j_config = {
    "uri": "bolt://172.20.77.180:7687",
    "user": "neo4j",
    "password": "neo4j@openspg",
    "database": "allday"
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
    extracted = extracted.dropna(subset=['what'], how='any')  # 过滤掉what为空的数据
    
    # 简单清洗
    for col in ['what', 'title']:
        extracted[col] = extracted[col].astype(str).str.strip()  # 去除收尾空格
    
    print(f"有效数据记录数: {len(extracted)}")
    return extracted

def remove_old_constraints(conn):
    """动态删除约束"""
    with conn.driver.session(database=conn.database) as session:
        # 查询所有约束
        result = session.run("SHOW CONSTRAINTS")
        constraints_to_drop = []
        
        for record in result:
            if (
                record["entityType"] == "NODE" 
                and record["labelsOrTypes"] == ["What"]
            ):
                constraints_to_drop.append(record["name"])
        
        # 删除目标约束
        for name in constraints_to_drop:
            session.run(f"DROP CONSTRAINT {name} IF EXISTS")
            print(f"已删除约束: {name}")
        
        print("旧约束清理完成" if constraints_to_drop else "无需清理旧约束")

def create_new_constraints(conn):
    """创建新约束（where when保持唯一性）"""
    with conn.driver.session(database=conn.database) as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Where) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:When) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Who) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:organization) REQUIRE n.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:category) REQUIRE n.name IS UNIQUE")
        print("新约束创建完成")

def import_batch(tx, batch):
    """批量导入函数"""
    for _, row in batch.iterrows():
        try:
            what = row['what']
            title = row['title']
            if not what or not title:
                continue

            # 创建主节点（直接CREATE允许重复）
            tx.run("""
                CREATE (what:What {
                    name: $what,
                    title: $title,
                    why: $why,
                    how: $how,
                   news_id:$news_id
                })
            """, {
                'what': what,
                'title': title,
                'why': row['why'] if pd.notna(row['why']) else None,
                'how': row['how'] if pd.notna(row['how']) else None,
                'news_id': row['news_id'] if pd.notna(row['news_id']) else None
            })

            # 处理关联节点（仍保持唯一性）
            def link_node(col, label, rel_type):
                value = str(row[col]).strip() if pd.notna(row[col]) else None
                if value:
                    values = [v.strip() for v in value.split(",")]
                    for v in values:
                        if not v:
                            continue
                        if label == 'Who':
                            if '-' in v:
                                # 分割成名字-职位-机构三部分
                                parts = v.split('-', 2)
                                if len(parts) == 3:
                                    person, position, org = [part.strip() for part in parts]
                                    # 检查所有部分非空
                                    if person and position and org:
                                        # 创建或合并Who节点和机构节点
                                        tx.run("MERGE (n:Who {name: $person})", {'person': person})
                                        tx.run("MERGE (o:organization {name: $org})", {'org': org})
                                        # 创建带职位属性的BELONGS_TO关系
                                        tx.run("""
                                            MATCH (w:Who {name: $person}), (o:organization {name: $org})
                                            MERGE (w)-[r:BELONGS_TO]->(o)
                                            SET r.position = $position
                                        """, {'person': person, 'org': org, 'position': position})
                                        # 连接What到Who
                                        tx.run("""
                                            MATCH (a:What {name: $what, title: $title}), (b:Who {name: $person})
                                            MERGE (a)-[:参与者]->(b)
                                        """, {'what': what, 'title': title, 'person': person})
                                        continue  # 处理成功，跳过后续逻辑
                                # 处理无效情况（分割不足三部分或部分为空）
                                # 回退到旧逻辑尝试分割成两部分（可选）
                                parts = v.split('-', 1)
                                person = parts[0].strip()
                                org = parts[1].strip() if len(parts) > 1 else ''
                                if person and org:
                                    tx.run("MERGE (n:Who {name: $person})", {'person': person})
                                    tx.run("MERGE (o:organization {name: $org})", {'org': org})
                                    tx.run("""
                                        MATCH (w:Who {name: $person}), (o:organization {name: $org})
                                        MERGE (w)-[:BELONGS_TO]->(o)
                                    """, {'person': person, 'org': org})
                                    tx.run("""
                                        MATCH (a:What {name: $what, title: $title}), (b:Who {name: $person})
                                        MERGE (a)-[:参与者]->(b)
                                    """, {'what': what, 'title': title, 'person': person})
                                    continue
                            # 其他无效情况直接创建Who节点
                            tx.run("MERGE (n:Who {name: $v})", {'v': v})
                            tx.run("""
                                MATCH (a:What {name: $what, title: $title}), (b:Who {name: $v})
                                MERGE (a)-[:参与者]->(b)
                            """, {'what': what, 'title': title, 'v': v})
                        else:
                            # 处理其他字段（Where/When）
                            tx.run(f"MERGE (n:{label} {{name: $v}})", {'v': v})
                            tx.run(f"""
                                MATCH (a:What {{name: $what, title: $title}})
                                MATCH (b:{label} {{name: $v}})
                                MERGE (a)-[:{rel_type}]->(b)
                            """, {'what': what, 'title': title, 'v': v})
            link_node('where', 'Where', '地点')
            link_node('when', 'When', '时间')
            link_node('organization', 'organization', '组织')
            link_node('who', 'Who', '参与者')


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
    print(f"完成！总导入记录数: {len(df_clean)}")
