from neo4j import GraphDatabase
import json
from openai import OpenAI
from event_add_property.get_thing_type import get_thing_type
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class EventProcessor():
    def __init__(self, CONFIG, all):
        # 只保存配置，不初始化不可序列化的对象
        self.neo4j_config = CONFIG["neo4j_config"]
        self.api_config = CONFIG["api"]
        self.event_type = get_thing_type("./event_add_property/事件属性.json")
        self.client = None
        self.driver = None
        self.all = all #判断是在哪个数据库执行
    
    def init_resources(self):
        """初始化资源（在每个子进程中调用）"""
        self.client = OpenAI(api_key=self.api_config["key"], base_url=self.api_config["base_url"])
        self.driver = GraphDatabase.driver(
            self.neo4j_config["uri"],
            auth=(self.neo4j_config["user"], self.neo4j_config["password"]),
            database=self.neo4j_config["alldatabase"] if self.all else self.neo4j_config["database"]
        )
    
    def close_resources(self):
        """关闭资源"""
        if self.driver is not None:
            self.driver.close()
    
    def fetch_cluster_with_what_nodes(self, tx):
        """查询所有符合条件的集群及其包含的What节点"""
        query = """
        MATCH (w:What)-[:BELONGS_TO]->(c:Cluster)
        WITH c, collect(w) AS what_nodes
        WHERE size(what_nodes) >= 2 AND c.when IS NULL
        RETURN c, what_nodes
        """
        result = tx.run(query)
        return [(record["c"], record["what_nodes"]) for record in result]
    
    def fetch_related_nodes(self, tx, what_node_ids):
        """查询与What节点相关的节点"""
        query = """
        UNWIND $what_node_ids AS what_id
        MATCH (what)-[r]->(related)
        WHERE elementId(what) = what_id AND type(r) IN ['参与者', '时间', '地点', '组织']
        RETURN what, collect({node: related, label: labels(related)[0]}) AS related_nodes
        """
        result = tx.run(query, what_node_ids=what_node_ids)
        return {record["what"].element_id: record["related_nodes"] for record in result}
    
    def save_results_to_file(self, results, filename="results.json"):
        """将结果保存为JSON文件"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    
    def generate_5w1h_summary(self, cluster_data):
        """使用大模型生成5W1H总结"""
        prompt = f"输入: {cluster_data['related_nodes']}"
        system_prompt = f"""
# 概述
你是一个顶级算法，旨在从一堆结构化信息中总结出5w1h以及组织和事件类型，以丰富知识图谱。

5w1h是指：what、who、why、where、when、how

目的是实现事件知识图谱的简单性和清晰性，使其可供广大受众使用。

### 2.节点
​**what**​：确保可以总结事件的主要内容，你可能会获取多个what，请概括总结成一个what，尽可能详细。         
​**how**​：确保可以总结事件发生后的结果，你可能会获取多个how，请概括总结成一个how，尽可能详细。 
​**why**​：确保可以总结事件发生的主要原因，你可能会获取多个why，请概括总结成一个why，尽可能保持精简和详细。         
​**who**​：事件的参与者大多数都不是一个人，你会得到一些人名，如果有重复的人名，请删除，多个人名使用逗号","分隔开。
​**where**​：确保是事件的主要发生地点，可能会出现多个地点，每个地点之间使用个逗号","分隔开。
​**when**​：确保事件发生的时间，并按照年-月-日格式返回，如果输入中没有时间，那么设为空字符串。
​**organization**​：确保是事件所牵涉的组织，如果输入中出现多个组织，请以逗号","分隔开。
​**type**​：事件的类别划分最为重要，你需要从{self.event_type}中找到和事件相匹配的等级代码。

最终type的返回格式：主码.子码.细码-属性组合，比如：01.01.03-N-H-B-C

### 3.严格遵守
严格遵守节点抽取规则。不合规将导致终止。

### 4.返回格式
以JSON格式返回："what":"", "when":"", "where":"", "why":"", "who":"", "how":"","organization","type"
"""
        try:
            response = self.client.chat.completions.create(
                model=self.api_config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating 5W1H summary: {e}")
            return None
    
    def update_cluster_properties(self, tx, cluster_id, properties):
        """更新Cluster节点的属性"""
        query = """
        MATCH (c:Cluster) 
        WHERE elementId(c) = $cluster_id
        SET c += $properties
        RETURN c
        """
        tx.run(query, cluster_id=cluster_id, properties=properties)
    
    def process_and_update_cluster(self, cluster_data):
        """处理单个集群数据并更新到Neo4j"""
        summary = self.generate_5w1h_summary(cluster_data)
        if summary:
            cluster_data["5w1h_summary"] = summary
            properties_to_update = {
                "what": summary.get("what", ""),
                "when": summary.get("when", ""),
                "where": summary.get("where", ""),
                "why": summary.get("why", ""),
                "who": summary.get("who", ""),
                "how": summary.get("how", ""),
                "organization": summary.get("organization", ""),
                "type": summary.get("type", ""),
                "processed": True
            }
            with self.driver.session() as session:
                session.execute_write(
                    self.update_cluster_properties, 
                    cluster_data["cluster"].get("elementId"), 
                    properties_to_update
                )
        return cluster_data
    
    def delete_single_what_clusters(self, tx) -> int:
        """删除只包含一个What节点的Cluster节点"""
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
    
    def cleanup_single_what_clusters(self):
        """清理只包含一个What节点的Cluster节点"""
        try:
            with self.driver.session() as session:
                deleted_count = session.execute_write(self.delete_single_what_clusters)
                print(f"Deleted {deleted_count} single-what clusters")
        except Exception as e:
            print(f"Error cleaning up single-what clusters: {e}")
    
    @staticmethod
    def process_cluster_task(args):
        """静态方法，用于多进程处理单个集群"""
        cluster_data, config,all_flag = args
        try:
            # 在每个子进程中创建新的处理器实例
            processor = EventProcessor(config,all_flag)
            processor.init_resources()
            result = processor.process_and_update_cluster(cluster_data)
            processor.close_resources()
            return result
        except Exception as e:
            print(f"Error processing cluster: {e}")
            return None
    
    def event(self):
        """主函数，执行整个流程"""
        try:
            # 在主进程中初始化资源用于查询
            self.init_resources()
            
            with self.driver.session() as session:
                # 查询集群和what节点
                cluster_with_what_nodes = session.execute_read(self.fetch_cluster_with_what_nodes)
                
                # 提取所有what节点ID
                all_what_node_ids = [node.element_id for _, what_nodes in cluster_with_what_nodes for node in what_nodes]
                
                # 查询相关节点
                related_nodes = session.execute_read(self.fetch_related_nodes, all_what_node_ids)
                
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
            
            # 准备配置数据
            config = {
                "neo4j_config": self.neo4j_config,
                "api": self.api_config,
                "event_type": self.event_type  # 确保事件类型也被传递
            }
            
            # 使用多进程处理
            with Pool(processes=min(cpu_count(), len(cluster_data_list))) as pool:
                # 使用静态方法处理，避免序列化问题
                results = list(tqdm(
                    pool.imap(
                        EventProcessor.process_cluster_task,
                        [(data, config,self.all) for data in cluster_data_list]
                    ),
                    total=len(cluster_data_list),
                    desc="Processing and updating clusters"
                ))
            
            # 过滤掉可能的None结果
            results = [r for r in results if r is not None]
            
            # 保存结果
            self.save_results_to_file(results)
            
            # 清理单what节点集群
            self.cleanup_single_what_clusters()
            
        finally:
            # 确保关闭数据库连接
            self.close_resources()


# 对外暴露的接口函数
def event(CONFIG,all):
    """执行整个事件处理流程"""
    EventProcessor(CONFIG,all).event()