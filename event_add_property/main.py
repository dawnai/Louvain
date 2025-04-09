from neo4j import GraphDatabase
import json
from openai import OpenAI
from get_thing_type import get_thing_type
from multiprocessing import Pool, cpu_count  # 引入多进程模块
from tqdm import tqdm  # 引入 tqdm 库
import logging
# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置信息
neo4j_config = {
    "uri": "bolt://172.20.196.206:7687",
    "user": "neo4j",
    "password": "neo4j@openspg",
    "database": "cache"
}

# 大模型配置
CONFIG = {
    "api": {
        "key": "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c",
        "base_url": "https://llm.jnu.cn/v1",
        "model": "Qwen2.5-72B-Instruct",
        "retries": 3,
        "timeout": 30
    }
}

# 初始化 OpenAI 客户端
client = OpenAI(api_key=CONFIG["api"]["key"], base_url=CONFIG["api"]["base_url"])

# 连接到 Neo4j 数据库
driver = GraphDatabase.driver(
    neo4j_config["uri"],
    auth=(neo4j_config["user"], neo4j_config["password"]),
    database=neo4j_config["database"]
)

# 事件类别划分
event_type = get_thing_type("事件属性.json")

def fetch_cluster_with_what_nodes(tx):
    """
    查询所有符合条件的集群及其包含的 What 节点 必须包含两个以上的what节点。但是目前没有限制when是否为空的Cluster节点
    """
    query = """
    MATCH (w:What)-[:BELONGS_TO]->(c:Cluster)
    WITH c, collect(w) AS what_nodes
    WHERE size(what_nodes) >= 2
    RETURN c, what_nodes
    """
    result = tx.run(query)
    return [(record["c"], record["what_nodes"]) for record in result]

def fetch_related_nodes(tx, what_node_ids):
    """
    查询与 What 节点相关的节点（如参与者、时间、地点、组织等）
    """
    query = """
    UNWIND $what_node_ids AS what_id
    MATCH (what)-[r]->(related)
    WHERE elementId(what) = what_id AND type(r) IN ['参与者', '时间', '地点', '组织']
    RETURN what, collect({node: related, label: labels(related)[0]}) AS related_nodes
    """
    result = tx.run(query, what_node_ids=what_node_ids)
    return {record["what"].element_id: record["related_nodes"] for record in result}

def save_results_to_file(results, filename="results.json"):
    """
    将结果保存为 JSON 文件
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {filename}")

def generate_5w1h_summary(cluster_data, thing):
    """
    使用大模型生成 5W1H 总结
    """
    # 构建提示词
    prompt = f"""
    输入: {cluster_data["related_nodes"]}
    """
    system_prompt = f"""
# 

### 1.概述

你是一个顶级算法，旨在从一堆结构化信息中总结出5w1h以及组织和事件类型，以丰富知识图谱。

5w1h是指：what、who、why、where、when、how

目的是实现事件知识图谱的简单性和清晰性，使其可供广大受众使用。

### 2.节点

**what**：确保可以总结事件的主要内容，你可能会获取多个what，请概括总结成一个what，尽可能详细。         

**how**：确保可以总结事件发生后的结果，你可能会获取多个how，请概括总结成一个how，尽可能详细。 

**why**：确保可以总结事件发生的主要原因，你可能会获取多个why，请概括总结成一个why，尽可能保持精简和详细。         

**who**：事件的参与者大多数都不是一个人，你会得到一些人名，如果有重复的人名，请删除，多个人名使用逗号","分隔开，例如（罗尼·吉尔·加万、费迪南德·R·马科斯、雷蒙德·M·鲍威尔、特蕾莎·马班尼亚），**注意！不要返回公司、组织、机构名称！**

**where**：确保是事件的主要发生地点，可能会出现多个地点，每个地点之间使用个逗号","分隔开。例如（黄岩岛、苏比克湾、三描礼士省、扎姆巴莱斯），

**when**：确保事件发生的时间，并按照 年-月-日格式返回(例如："2025-01-19") ，如果输入中没有时间，那么设为空字符串，不做任何解释。如果出现了多个时间，请以频率最高的时间作为返回时间。

 **organization**：确保是事件所牵涉的组织，如果输入中出现多个组织，请以逗号","分隔开。例如（中国海岸警卫队,菲律宾海岸警卫队）

**type：**事件的类别划分最为重要，你需要从{thing}中找到和事件相匹配的等级代码，每个事件有三级分类，同时还需要提取以下信息：

**影响范围**：

- 全球性 (G)
- 区域性 (R)
- 国家性 (N)
- 地方性 (L)

**紧急程度**

- 紧急 (U)：需立即响应
- 高度 (H)：48小时内需响应
- 中度 (M)：一周内需跟进
- 常规 (S)：常规监测即可

**事件状态**

- 开始 (B)：事件刚发生
- 发展 (D)：事件正在进行
- 高潮 (P)：事件达到高潮
- 缓解 (E)：事件开始缓解
- 结束 (F)：事件已结束

**信息可靠性**

- 确认 (C)：多渠道确认
- 可能 (P)：单一可靠来源
- 未确认 (U)：消息来源不明确
- 谣言 (R)：可能是虚假信息

最终type的返回格式：主码.子码.细码-属性组合，比如：01.01.03-N-H-B-C

### 3.严格遵守         

严格遵守节点抽取规则。不合规将导致终止。

### 4.返回格式

以JSON格式返回："what":"", "when":"", "where":"", "why":"", "who":"", "how":"","organization","type"
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG["api"]["model"],  # 使用 暨大模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        # 提取生成的总结
        result = json.loads(response.choices[0].message.content)
        return result  # 将生成的总结解析为 JSON
    except Exception as e:
        print(f"Error generating 5W1H summary: {e}")
        return None

def update_cluster_properties(tx, cluster_id, properties):
    """
    更新Cluster节点的属性
    """
    query = """
    MATCH (c:Cluster) 
    WHERE elementId(c) = $cluster_id
    SET c += $properties
    RETURN c
    """
    tx.run(query, cluster_id=cluster_id, properties=properties)

def process_and_update_cluster(cluster_data):
    """
    处理单个集群数据并更新到Neo4j
    """
    summary = generate_5w1h_summary(cluster_data, event_type)
    if summary:
        cluster_data["5w1h_summary"] = summary
        
        # 准备要更新的属性
        properties_to_update = {
            "what": summary.get("what", ""),
            "when": summary.get("when", ""),
            "where": summary.get("where", ""),
            "why": summary.get("why", ""),
            "who": summary.get("who", ""),
            "how": summary.get("how", ""),
            "organization": summary.get("organization", ""),
            "type": summary.get("type", ""),
            "processed": True  # 添加标记表示已处理
        }
        
        # 更新到Neo4j
        with driver.session() as session:
            session.execute_write(
                update_cluster_properties, 
                cluster_data["cluster"].get("elementId"), 
                properties_to_update
            )
    
    return cluster_data

def delete_single_what_clusters(tx) -> int:
    """
    删除只包含一个What节点的Cluster节点
    返回删除的节点数量
    """
    query = """
    MATCH (c:Cluster)
    WITH c
    MATCH (w:What)-[:BELONGS_TO]->(c)
    WITH c, count(w) AS what_count
    WHERE what_count = 1
    DETACH DELETE c
    RETURN count(c) AS deleted_count
    """
    result = tx.run(query)
    return result.single()["deleted_count"]

def cleanup_single_what_clusters() -> None:
    """
    清理只包含一个What节点的Cluster节点
    """
    try:
        with driver.session() as session:
            deleted_count = session.execute_write(delete_single_what_clusters)
            logger.info(f"已删除 {deleted_count} 个只包含单个What节点的Cluster节点")
    except Exception as e:
        logger.error(f"清理Cluster节点时出错: {e}")
        raise


def main():
    with driver.session() as session:
        # 查询所有符合条件的集群及其包含的 What 节点
        cluster_with_what_nodes = session.execute_read(fetch_cluster_with_what_nodes)
        logger.info(f"成功获取到: {len(cluster_with_what_nodes)}条事件")

        # 提取所有 What 节点的 element_id
        all_what_node_ids = [node.element_id for _, what_nodes in cluster_with_what_nodes for node in what_nodes]
        
        # 查询与 What 节点相关的节点
        related_nodes = session.execute_read(fetch_related_nodes, all_what_node_ids)
        logger.info(f"成功获取到: {len(all_what_node_ids)}条关系")

        # 构建集群数据列表
        cluster_data_list = []
        for cluster, what_nodes in cluster_with_what_nodes:
            cluster_data = {
                "cluster": {
                    "elementId": cluster.element_id,
                    "properties": dict(cluster)
                },
                "related_nodes": []
            }
            for what_node in what_nodes:
                what_id = what_node.element_id
                if what_id in related_nodes:
                    cluster_data["related_nodes"].append({
                        "what_node": dict(what_node),
                        "related": [
                            {record["label"]: record["node"]["name"]}
                            for record in related_nodes[what_id]
                        ]
                    })
            cluster_data_list.append(cluster_data)
        
        # 使用多进程并行处理集群数据，并添加进度条
        with Pool(processes=5) as pool:
            results = list(tqdm(
                pool.imap(process_and_update_cluster, cluster_data_list), 
                total=len(cluster_data_list), 
                desc="Processing and updating clusters"
            ))
        
        # 保存结果到文件
        save_results_to_file(results)
    # 清理只包含一个What节点的Cluster节点
    cleanup_single_what_clusters()
    # 关闭数据库连接
    driver.close()

if __name__ == "__main__":
    main()
