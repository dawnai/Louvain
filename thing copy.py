from neo4j import GraphDatabase
import json
from functools import partial
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
from typing import List, Dict, Any
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据库配置（建议实际使用时从环境变量读取）
neo4j_config = {
    "uri": "bolt://172.20.97.143:7687",
    "user": "neo4j",
    "password": "neo4j@openspg",
    "database": "test",
    "max_connection_pool_size": 50,
    "connection_timeout": 30
}

class Neo4jConnector:
    """封装Neo4j连接管理"""
    def __init__(self, config):
        self._driver = GraphDatabase.driver(
            config["uri"],
            auth=(config["user"], config["password"]),
            database=config["database"],
            max_connection_pool_size=config.get("max_connection_pool_size", 50),
            connection_timeout=config.get("connection_timeout", 30)
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def execute_read(self, query_func, **kwargs):
        """带重试机制的读操作"""
        with self._driver.session() as session:
            return session.execute_read(query_func, **kwargs)
    
    def close(self):
        self._driver.close()

def node_to_dict(node) -> Dict[str, Any]:
    """更新ID处理逻辑"""
    return {
        "internal_id": id(node),  # 使用原生ID
        "element_id": node.element_id,
        "labels": list(node.labels),
        "properties": dict(node)
    }

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_cluster_with_what_nodes(connector: Neo4jConnector) -> List[Any]:
    """获取包含多个What节点的Cluster"""
    def _fetch(tx):
        query = """
        MATCH (w:What)-[:BELONGS_TO]->(c:Cluster)
        WITH c, collect(w) AS what_nodes
        WHERE size(what_nodes) >= 2
        RETURN c, what_nodes
        """
        result = tx.run(query)
        return [(record["c"], record["what_nodes"]) for record in result]
    return connector.execute_read(_fetch)

def batch_fetch_related_nodes(connector: Neo4jConnector, what_node_ids: List[str], batch_size=100) -> Dict[str, List[Any]]:
    """修正后的批量查询方法"""
    results = {}
    
    for i in range(0, len(what_node_ids), batch_size):
        batch_ids = what_node_ids[i:i+batch_size]
        logger.info(f"正在处理批次 {i//batch_size + 1}/{(len(what_node_ids)-1)//batch_size + 1}")

        # 转换ID格式：从"4:xxx:92"提取92
        converted_ids = [int(id.split(":")[-1]) for id in batch_ids]

        def _fetch(tx):
            query = """
            UNWIND $what_node_ids AS node_id
            MATCH (what)-[r:参与者|时间|地点|组织]->(related)
            WHERE id(what) = node_id  // 使用内部ID
            RETURN id(what) as what_id, collect(related) AS related_nodes
            """
            params = {"what_node_ids": converted_ids}
            result = tx.run(query, params)
            return {str(record["what_id"]): record["related_nodes"] for record in result}

        batch_result = connector.execute_read(_fetch)
        results.update(batch_result)
    
    return results


def process_results(raw_data, related_nodes) -> List[Dict]:
    """处理查询结果结构"""
    processed = []
    
    for cluster, what_nodes in raw_data:
        related_info = {}
        for node in what_nodes:
            node_id = node.element_id
            if node_id in related_nodes:
                # 转换每个相关节点信息
                related_info[node_id] = [node_to_dict(n) for n in related_nodes[node_id]]
        
        cluster_data = {
            "cluster": node_to_dict(cluster),
            "what_nodes": [node_to_dict(n) for n in what_nodes],
            "related_relations": related_info
        }
        processed.append(cluster_data)
    
    return processed

class EnhancedJSONEncoder(json.JSONEncoder):
    """处理日期类型的JSON序列化"""
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def save_results_to_file(results, filename="full_results.json"):
    """保存结果到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, 
                 ensure_ascii=False, 
                 indent=4, 
                 cls=EnhancedJSONEncoder)
    logger.info(f"已保存 {len(results)} 个聚类结果到 {filename}")

def extract_neo4j_id(node) -> str:
    parts = node.element_id.split(":")
    return parts[-1] if len(parts) > 2 else node.element_id
def main():
    """主流程"""
    connector = Neo4jConnector(neo4j_config)
    
    try:
        # 获取基础数据
        cluster_data = fetch_cluster_with_what_nodes(connector)
        logger.info(f"共获取 {len(cluster_data)} 个有效聚类")

        # 提取并去重所有What节点ID
        all_what_ids = list({extract_neo4j_id(node) for _, nodes in cluster_data for node in nodes})
        logger.info(f"需要处理的What节点数量：{len(all_what_ids)}")
        print("all_what_ids:",all_what_ids)

        # 批量获取关联节点
        related_nodes = batch_fetch_related_nodes(connector, all_what_ids)
        logger.info(f"获取到 {sum(len(v) for v in related_nodes.values())} 个关联关系")
        
        print("related_nodes",related_nodes)
        # 处理结果结构
        processed_results = process_results(cluster_data, related_nodes)
        
        # 保存结果
        save_results_to_file(processed_results)
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise
    finally:
        connector.close()

if __name__ == "__main__":
    main()
