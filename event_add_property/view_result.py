from neo4j import GraphDatabase
import pandas as pd

# 数据库连接配置
URI = "neo4j://172.20.196.206:7687"
AUTH = ("neo4j", "neo4j@openspg")
DATABASE = "cache"

# Cypher查询语句
QUERY = """
MATCH (c:Cluster)
WHERE c.when IS NOT NULL
RETURN c
ORDER BY c.when
"""

def fetch_clusters():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    try:
        with driver.session(database=DATABASE) as session:
            result = session.run(QUERY)
            
            clusters = []
            for record in result:
                node = record["c"]
                cluster_data = dict(node.items())
                clusters.append(cluster_data)
            
            df = pd.DataFrame(clusters)
            
            # 调整列顺序
            if not df.empty:
                # 获取所有列名
                all_columns = df.columns.tolist()
                # 确保核心列存在
                core_columns = ['when', 'what']
                # 其他列按字母顺序排序
                other_columns = sorted([col for col in all_columns if col not in core_columns])
                # 组合最终列顺序
                final_columns = core_columns + other_columns
                # 重组DataFrame
                df = df.reindex(columns=final_columns)
            
            # 保存为Excel
            df.to_excel("clusters.xlsx", index=False, engine='openpyxl')
            print("数据已保存至 clusters.xlsx 按照时间先后排序")
            
    except Exception as e:
        print(f"操作出错: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    fetch_clusters()
