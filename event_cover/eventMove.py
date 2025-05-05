from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from tool.TextProcessor import TextProcessor

class ClusterMigrationService:
    def __init__(self, neo4j_uri: str = "bolt://172.20.25.129:7687", 
                 neo4j_auth: tuple = ("neo4j", "neo4j@openspg"),
                 max_workers: int = 4):
        """
        初始化集群迁移服务
        
        参数:
            neo4j_uri: Neo4j数据库连接URI
            neo4j_auth: Neo4j认证信息 (用户名, 密码)
            max_workers: 文本处理的最大工作线程数
        """
        self.NEO4J_URI = neo4j_uri
        self.NEO4J_AUTH = neo4j_auth
        self.processor = TextProcessor(max_workers=max_workers)
        self.driver = GraphDatabase.driver(self.NEO4J_URI, auth=self.NEO4J_AUTH)
    
    def migrate_clusters(self, 
                        source_db: str = "cache", 
                        target_db: str = "all",
                        similarity_threshold: float = 0.8,
                        why_weight: float = 0.5,
                        what_weight: float = 0.5,
                        what_similarity_threshold: float = 0.95,
                        clear_source: bool = False) -> dict:
        """
        执行完整的集群迁移流程
        
        参数:
            source_db: 源数据库名称 (默认: "cache")
            target_db: 目标数据库名称 (默认: "all")
            similarity_threshold: Cluster节点相似度阈值 (默认: 0.8)
            why_weight: why属性的权重 (默认: 0.5)
            what_weight: what属性的权重 (默认: 0.5)
            what_similarity_threshold: What节点相似度阈值 (默认: 0.95)
            clear_source: 迁移完成后是否清空源数据库 (默认: False)
            
        返回:
            包含迁移统计信息的字典
        """
        stats = {
            "total_source_clusters": 0,
            "total_source_whats": 0,
            "matched_clusters": 0,
            "unmatched_clusters": 0,
            "matched_whats": 0,
            "unmatched_whats": 0
        }
        
        try:
            # 1. 获取源数据库和目标数据库的Cluster节点
            print(f"从 {source_db} 数据库获取Cluster节点...")
            clusters_source = self._get_clusters_with_properties(source_db)
            print(f"从 {target_db} 数据库获取Cluster节点...")
            clusters_target = self._get_clusters_with_properties(target_db)
            
            stats["total_source_clusters"] = len(clusters_source)
            
            if not clusters_source:
                print(f"{source_db} 数据库中没有找到带有why和what属性的Cluster节点")
                return stats
            
            if not clusters_target:
                print(f"{target_db} 数据库中没有找到带有why和what属性的Cluster节点")
                # 直接迁移所有源Cluster节点
                return self._migrate_all_clusters(source_db, target_db, stats, clear_source)
            
            # 2. 计算相似对
            similar_pairs = self._calculate_combined_similarity_pairs(
                clusters_target, 
                clusters_source,
                threshold=similarity_threshold,
                why_weight=why_weight,
                what_weight=what_weight
            )
            
            stats["matched_clusters"] = len(similar_pairs)
            print(f"\n找到 {len(similar_pairs)} 对综合相似度 > {similarity_threshold} 的Cluster节点")
            
            # 3. 执行迁移
            with self.driver.session(database=target_db) as target_session, \
                 self.driver.session(database=source_db) as source_session:
                
                # 3.1 迁移匹配的Cluster节点
                matched_source_ids = set()
                for pair in similar_pairs:
                    cluster_target, cluster_source, combined_sim, why_sim, what_sim = pair
                    
                    print(f"\n处理节点对 - 综合相似度: {combined_sim:.4f}")
                    print(f"{target_db} 数据库节点ID: {cluster_target['id']}")
                    print(f"{source_db} 数据库节点ID: {cluster_source['id']}")
                    
                    new_cluster_id = self._migrate_cluster_with_relations(
                        target_session,
                        source_session,
                        cluster_target["id"],
                        cluster_source["id"]
                    )
                    
                    if new_cluster_id is not None:
                        matched_source_ids.add(cluster_source["id"])
                        print(f"成功迁移Cluster {cluster_source['id']} 到 {target_db} 数据库，新ID: {new_cluster_id}")
                
                # 3.2 迁移未匹配的Cluster节点
                print("\n开始迁移未匹配的Cluster节点...")
                unmatched_migrated = self._migrate_unmatched_clusters(
                    target_session,
                    source_session,
                    clusters_source,
                    matched_source_ids
                )
                stats["unmatched_clusters"] = unmatched_migrated
                
                # 3.3 迁移What节点
                print("\n开始处理What节点...")
                # 获取源数据库中孤立的What节点
                orphan_whats = self._get_orphan_what_nodes(source_db)
                stats["total_source_whats"] = len(orphan_whats)
                
                if orphan_whats:
                    # 匹配并迁移高相似度What节点
                    high_sim_migrated, matched_what_ids = self._match_and_migrate_high_similarity_whats(
                        source_db,
                        target_session,
                        source_session,
                        threshold=what_similarity_threshold
                        
                    )
                    stats["matched_whats"] = high_sim_migrated
                    
                    # 迁移未匹配的What节点
                    unmatched_migrated = self._migrate_unmatched_what_nodes(
                        target_session,
                        source_session,
                        matched_what_ids
                    )
                    stats["unmatched_whats"] = unmatched_migrated
                
                # 4. 可选: 清空源数据库
                if clear_source:
                    print(f"\n正在清空 {source_db} 数据库...")
                    self._clear_database(source_db)
                    print(f"{source_db} 数据库已清空")
            
            print("\n迁移完成!")
            return stats
            
        except Exception as e:
            print(f"迁移过程中发生错误: {str(e)}")
            return stats
        finally:
            self.driver.close()
    

    
    def _get_clusters_with_properties(self, database: str) -> List[Dict]:
        """从指定数据库中获取带有why和what属性的Cluster节点"""
        with self.driver.session(database=database) as session:
            query = """
            MATCH (c:Cluster)
            WHERE c.why IS NOT NULL AND c.what IS NOT NULL
            RETURN elementId(c) AS id, c.why AS why, c.what AS what, c.how AS how, 
                   c.organization AS organization, c.type AS type, c.when AS when, 
                   c.where AS where, c.who AS who
            """
            result = session.run(query)
            return [{
                "id": record["id"], 
                "why": record["why"], 
                "what": record["what"],
                "how": record["how"],
                "organization": record["organization"],
                "type": record["type"],
                "when": record["when"],
                "where": record["where"],
                "who": record["who"]
            } for record in result]
    
    def _get_related_nodes(self, session, cluster_id: str) -> Dict:
        """获取与Cluster节点相关的所有What节点及其关系"""
        query = """
        MATCH (w:What)-[r1]->(c:Cluster)
        WHERE elementId(c) = $cluster_id
        OPTIONAL MATCH (w)-[r2]->(n)
        WHERE NOT n:Cluster
        RETURN 
            properties(c) as cluster,
            properties(w) as what,
            type(r1) as rel1_type,
            properties(r1) as rel1_props,
            type(r2) as rel2_type,
            properties(r2) as rel2_props,
            labels(n) as node_labels,
            properties(n) as node_props,
            elementId(w) as what_id
        ORDER BY what_id
        """
        result = session.run(query, cluster_id=cluster_id)
        
        grouped_data = {}
        for record in result:
            data = record.data()
            what_id = data["what_id"]
            if what_id not in grouped_data:
                grouped_data[what_id] = {
                    "what": data["what"],
                    "rel_to_cluster": {
                        "type": data["rel1_type"],
                        "props": data["rel1_props"]
                    },
                    "related_nodes": []
                }
            if data["node_labels"]:
                grouped_data[what_id]["related_nodes"].append({
                    "labels": data["node_labels"],
                    "props": data["node_props"],
                    "rel_type": data["rel2_type"],
                    "rel_props": data["rel2_props"]
                })
        
        return list(grouped_data.values())
    
    def _migrate_cluster_with_relations(self, target_session, source_session, target_cluster_id: str, source_cluster_id: str):
        """将源数据库中的Cluster及其所有What节点和相关节点迁移到目标数据库"""
        # 1. 获取源中的Cluster节点属性
        get_cluster_query = """
        MATCH (c:Cluster)
        WHERE elementId(c) = $cluster_id
        RETURN properties(c) as props
        """
        cluster_result = source_session.run(get_cluster_query, cluster_id=source_cluster_id)
        cluster_data = cluster_result.single()
        
        if not cluster_data:
            print(f"警告: 源数据库中没有找到Cluster {source_cluster_id}")
            return None
        
        # 2. 获取所有What节点及其相关节点
        what_nodes_data = self._get_related_nodes(source_session, source_cluster_id)
        
        if not what_nodes_data:
            print(f"警告: 源数据库中没有找到与Cluster {source_cluster_id} 相关的What节点")
            return None
        
        # 3. 在目标数据库中创建新Cluster节点
        create_cluster_query = """
        MATCH (source:Cluster) WHERE elementId(source) = $source_id
        CREATE (new:Cluster $props)
        CREATE (source)-[:SIMILAR_TO {similarity: $similarity}]->(new)
        RETURN elementId(new) as new_id
        """
        
        try:
            # 创建新Cluster节点并建立与源Cluster的关系
            new_cluster_id = target_session.run(create_cluster_query, 
                                             source_id=target_cluster_id,
                                             props=cluster_data["props"],
                                             similarity=1.0).single()["new_id"]
            
            # 迁移所有What节点及其关系
            for what_data in what_nodes_data:
                # 创建What节点及其到Cluster的关系
                create_what_query = f"""
                MATCH (c:Cluster) WHERE elementId(c) = $cluster_id
                MERGE (w:What {{name: $what_name}})
                ON CREATE SET w += $props
                MERGE (w)-[r:{what_data["rel_to_cluster"]["type"]}]->(c)
                ON CREATE SET r = $rel_props
                RETURN elementId(w) as what_id
                """
                what_id = target_session.run(create_what_query,
                                          cluster_id=new_cluster_id,
                                          what_name=what_data["what"].get("name", ""),
                                          props=what_data["what"],
                                          rel_props=what_data["rel_to_cluster"]["props"]).single()["what_id"]
                
                # 创建该What节点指向的所有相关节点
                for related_node in what_data["related_nodes"]:
                    node_labels = related_node["labels"]
                    node_props = related_node["props"]
                    rel_type = related_node["rel_type"]
                    rel_props = related_node["rel_props"]
                    
                    # 获取唯一标识属性
                    unique_props = []
                    if "name" in node_props:
                        unique_props.append(("name", node_props["name"]))
                    elif "id" in node_props:
                        unique_props.append(("id", node_props["id"]))
                    else:
                        unique_props = [(k, v) for k, v in node_props.items() if v is not None]
                    
                    merge_conditions = []
                    params = {}
                    for i, (key, value) in enumerate(unique_props):
                        merge_conditions.append(f"{key}: $unique_val_{i}")
                        params[f"unique_val_{i}"] = value
                    
                    create_node_query = f"""
                    MATCH (w:What) WHERE elementId(w) = $what_id
                    MERGE (n:{':'.join(node_labels)} {{{', '.join(merge_conditions)}}})
                    ON CREATE SET n += $props
                    ON MATCH SET n += $props
                    MERGE (w)-[r:{rel_type}]->(n)
                    ON CREATE SET r = $rel_props
                    ON MATCH SET r += $rel_props
                    """
                    target_session.run(
                        create_node_query,
                        what_id=what_id,
                        props=node_props,
                        rel_props=rel_props,
                        **params
                    )
            
            return new_cluster_id
        except Exception as e:
            print(f"迁移Cluster {source_cluster_id} 失败: {str(e)}")
            return None
    
    def _migrate_unmatched_clusters(self, target_session, source_session, clusters_source, matched_source_ids):
        """迁移未匹配的Cluster节点及其所有What节点和相关节点"""
        migrated_count = 0
        
        for cluster in tqdm(clusters_source, desc="迁移未匹配Cluster节点"):
            if cluster["id"] in matched_source_ids:
                continue
            
            try:
                # 深拷贝cluster属性并移除系统字段
                import copy
                cluster_props = copy.deepcopy(cluster)
                for sys_field in ["id", "labels", "elementId", "internal_id"]:
                    cluster_props.pop(sys_field, None)
                
                # 创建新Cluster节点
                create_cluster_query = """
                CREATE (new:Cluster $props)
                RETURN elementId(new) AS new_id
                """
                new_cluster_id = target_session.run(
                    create_cluster_query,
                    props=cluster_props
                ).single()["new_id"]
                
                # 迁移What节点及其相关节点
                what_nodes_data = self._get_related_nodes(source_session, cluster["id"])
                
                for what_data in what_nodes_data:
                    what_node = copy.deepcopy(what_data["what"])
                    what_name = what_node.get("name", "")
                    
                    if "name" in what_data["what"]:
                        what_node["name"] = what_name
                    
                    rel_data = what_data["rel_to_cluster"]
                    
                    create_what_query = f"""
                    MATCH (c:Cluster) WHERE elementId(c) = $cluster_id
                    MERGE (w:What {{name: $name}})
                    ON CREATE SET w = $props
                    ON MATCH SET w += $props
                    MERGE (w)-[r:{rel_data["type"]}]->(c)
                    ON CREATE SET r = $rel_props
                    RETURN elementId(w) AS what_id
                    """
                    
                    what_id = target_session.run(
                        create_what_query,
                        cluster_id=new_cluster_id,
                        name=what_name,
                        props=what_node,
                        rel_props=rel_data["props"]
                    ).single()["what_id"]
                    
                    # 迁移相关节点
                    for related_node in what_data["related_nodes"]:
                        node_props = copy.deepcopy(related_node["props"])
                        labels = ":".join(related_node["labels"])
                        
                        merge_conditions = []
                        params = {}
                        
                        for uid_field in ["name", "id", "uuid", "key"]:
                            if uid_field in node_props:
                                merge_conditions.append(f"{uid_field}: ${uid_field}")
                                params[uid_field] = node_props.pop(uid_field)
                                break
                        else:
                            non_null_props = {k:v for k,v in node_props.items() if v is not None}
                            merge_conditions.extend(f"{k}: ${k}" for k in non_null_props)
                            params.update(non_null_props)
                        
                        create_node_query = f"""
                        MATCH (w:What) WHERE elementId(w) = $what_id
                        MERGE (n:{labels} {{ {', '.join(merge_conditions)} }})
                        ON CREATE SET n = $props
                        ON MATCH SET n += $props
                        MERGE (w)-[r:{related_node["rel_type"]}]->(n)
                        ON CREATE SET r = $rel_props
                        ON MATCH SET r += $rel_props
                        """
                        
                        target_session.run(
                            create_node_query,
                            what_id=what_id,
                            props=node_props,
                            rel_props=related_node["rel_props"],
                            **params
                        )
                
                migrated_count += 1
                print(f"✅ 成功迁移Cluster {cluster['id']} -> 新ID: {new_cluster_id}")
                
            except Exception as e:
                print(f"❌ 迁移失败 Cluster {cluster['id']}: {str(e)}")
                if 'new_cluster_id' in locals():
                    target_session.run(
                        "MATCH (c:Cluster) WHERE elementId(c) = $id DETACH DELETE c",
                        id=new_cluster_id
                    )
        
        return migrated_count
    
    def _calculate_combined_similarity_pairs(self, clusters_target, clusters_source, threshold, why_weight, what_weight):
        """计算两个集群列表之间的综合相似度对"""
        why_texts_target = [c["why"] for c in clusters_target]
        why_texts_source = [c["why"] for c in clusters_source]
        what_texts_target = [c["what"] for c in clusters_target]
        what_texts_source = [c["what"] for c in clusters_source]
        
        print("计算why属性的嵌入向量...")
        why_embeddings_target = self.processor.get_embeddings(why_texts_target)
        why_embeddings_source = self.processor.get_embeddings(why_texts_source)
        
        print("计算what属性的嵌入向量...")
        what_embeddings_target = self.processor.get_embeddings(what_texts_target)
        what_embeddings_source = self.processor.get_embeddings(what_texts_source)
        
        print("计算why相似度矩阵...")
        why_similarity_matrix = cosine_similarity(why_embeddings_target, why_embeddings_source)
        
        print("计算what相似度矩阵...")
        what_similarity_matrix = cosine_similarity(what_embeddings_target, what_embeddings_source)
        
        combined_similarity_matrix = why_weight * why_similarity_matrix + what_weight * what_similarity_matrix
        
        similar_pairs = []
        for i in tqdm(range(len(clusters_target)), desc="寻找相似对"):
            for j in range(len(clusters_source)):
                sim = combined_similarity_matrix[i][j]
                if sim > threshold:
                    similar_pairs.append((
                        clusters_target[i], 
                        clusters_source[j], 
                        sim,
                        why_similarity_matrix[i][j],
                        what_similarity_matrix[i][j]
                    ))
        
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs
    
    def _get_orphan_what_nodes(self, database: str) -> List[Dict]:
        """获取没有指向Cluster的What节点"""
        query = """
        MATCH (w:What)
        WHERE NOT (w)-[]->(:Cluster)
        RETURN elementId(w) AS id, w.name AS name, properties(w) AS properties
        """
        with self.driver.session(database=database) as session:
            result = session.run(query)
            return [{
                "id": record["id"], 
                "name": record["name"], 
                "properties": record["properties"]
            } for record in result]
    
    def _get_what_node_with_relations(self, session, what_id: str) -> Dict:
        """获取What节点及其所有相关节点"""
        query = """
        MATCH (w:What) WHERE elementId(w) = $what_id
        OPTIONAL MATCH (w)-[r]->(n)
        WHERE NOT n:Cluster
        RETURN 
            properties(w) as what,
            type(r) as rel_type,
            properties(r) as rel_props,
            labels(n) as node_labels,
            properties(n) as node_props
        """
        result = session.run(query, what_id=what_id)
        
        related_nodes = []
        what_data = None
        
        for record in result:
            data = record.data()
            if what_data is None:
                what_data = data["what"]
            
            if data["node_labels"]:
                related_nodes.append({
                    "labels": data["node_labels"],
                    "props": data["node_props"],
                    "rel_type": data["rel_type"],
                    "rel_props": data["rel_props"]
                })
        
        return {
            "what": what_data,
            "related_nodes": related_nodes
        }
    
    def _migrate_what_node_with_relations(self, target_session, source_session, what_id: str, cluster_id: str, 
                                        relation_type: str = "BELONGS_TO", relation_props: dict = None):
        """将What节点及其相关节点从源数据库迁移到目标数据库"""
        try:
            what_data = self._get_what_node_with_relations(source_session, what_id)
            if not what_data or not what_data["what"]:
                print(f"警告: 源数据库中没有找到What节点 {what_id}")
                return False
            
            what_name = what_data["what"].get("name", "")
            create_what_query = f"""
            MATCH (c:Cluster) WHERE elementId(c) = $cluster_id
            MERGE (w:What {{name: $what_name}})
            ON CREATE SET w += $props
            MERGE (w)-[r:{relation_type}]->(c)
            ON CREATE SET r = $rel_props
            RETURN elementId(w) as new_what_id
            """
            
            result = target_session.run(
                create_what_query,
                cluster_id=cluster_id,
                what_name=what_name,
                props=what_data["what"],
                rel_props=relation_props or {}
            )
            
            new_what_id = result.single()["new_what_id"]
            print(f"成功创建What节点 {new_what_id} 并关联到Cluster {cluster_id}")
            
            for related_node in what_data["related_nodes"]:
                node_labels = related_node["labels"]
                node_props = related_node["props"]
                rel_type = related_node["rel_type"]
                rel_props = related_node["rel_props"]
                
                unique_props = []
                if "name" in node_props:
                    unique_props.append(("name", node_props["name"]))
                elif "id" in node_props:
                    unique_props.append(("id", node_props["id"]))
                else:
                    unique_props = [(k, v) for k, v in node_props.items() if v is not None]
                
                merge_conditions = []
                params = {}
                for i, (key, value) in enumerate(unique_props):
                    merge_conditions.append(f"{key}: $unique_val_{i}")
                    params[f"unique_val_{i}"] = value
                
                create_node_query = f"""
                MATCH (w:What) WHERE elementId(w) = $what_id
                MERGE (n:{':'.join(node_labels)} {{{', '.join(merge_conditions)}}})
                ON CREATE SET n += $props
                ON MATCH SET n += $props
                MERGE (w)-[r:{rel_type}]->(n)
                ON CREATE SET r = $rel_props
                ON MATCH SET r += $rel_props
                """
                target_session.run(
                    create_node_query,
                    what_id=new_what_id,
                    props=node_props,
                    rel_props=rel_props,
                    **params
                )
                print(f"成功创建相关节点 {node_labels} 并关联到What节点 {new_what_id}")
            
            return True
        
        except Exception as e:
            print(f"迁移What节点 {what_id} 失败: {str(e)}")
            return False
    
    def _match_and_migrate_high_similarity_whats(self,source_db, target_session, source_session, threshold: float = 0.95):
        """匹配并迁移高相似度的What节点到目标数据库中的Cluster节点"""
        print("\n开始匹配并迁移高相似度What节点...")
        
        # 1. 获取源数据库中没有指向Cluster的What节点
        print("从源数据库获取孤立的What节点...")
        orphan_whats = self._get_orphan_what_nodes(source_db)
        if not orphan_whats:
            print("源数据库中没有孤立的What节点")
            return 0, set()
        
        print(f"找到 {len(orphan_whats)} 个孤立的What节点")
        
        # 2. 获取目标数据库中所有Cluster的what属性
        print("从目标数据库获取Cluster节点的what属性...")
        get_cluster_whats_query = """
        MATCH (c:Cluster)
        WHERE c.what IS NOT NULL
        RETURN elementId(c) AS cluster_id, c.what AS what_text
        """
        cluster_whats = target_session.run(get_cluster_whats_query)
        cluster_whats = [(record["cluster_id"], record["what_text"]) for record in cluster_whats]
        
        if not cluster_whats:
            print("目标数据库中没有找到带有what属性的Cluster节点")
            return 0, set()
        
        print(f"找到 {len(cluster_whats)} 个带有what属性的Cluster节点")
        
        # 3. 准备文本数据用于嵌入计算
        what_texts = [what["name"] for what in orphan_whats if what.get("name")]
        cluster_texts = [text for _, text in cluster_whats]
        
        # 4. 计算嵌入向量
        print("计算What节点和Cluster节点的嵌入向量...")
        what_embeddings = self.processor.get_embeddings(what_texts)
        cluster_embeddings = self.processor.get_embeddings(cluster_texts)
        
        # 5. 计算相似度矩阵
        print("计算相似度矩阵...")
        similarity_matrix = cosine_similarity(what_embeddings, cluster_embeddings)
        
        # 6. 找出相似度>threshold的匹配对
        matches = []
        for i in range(len(orphan_whats)):
            what_name = orphan_whats[i].get("name")
            if not what_name:
                continue
                
            for j in range(len(cluster_whats)):
                sim = similarity_matrix[i][j]
                if sim > threshold:
                    matches.append({
                        "what_id": orphan_whats[i]["id"],
                        "what_name": what_name,
                        "cluster_id": cluster_whats[j][0],
                        "cluster_what": cluster_whats[j][1],
                        "similarity": sim
                    })
        
        if not matches:
            print(f"没有找到相似度>{threshold}的匹配对")
            return 0, set()
        
        print(f"\n找到 {len(matches)} 对相似度>{threshold}的What节点和Cluster节点:")
        for match in sorted(matches, key=lambda x: x["similarity"], reverse=True):
            print(f"What节点(ID: {match['what_id']}, 名称: {match['what_name']})")
            print(f"匹配到Cluster节点(ID: {match['cluster_id']}, what属性: {match['cluster_what']})")
            print(f"相似度: {match['similarity']:.4f}\n")
        
        # 7. 迁移匹配的What节点
        migrated_count = 0
        matched_what_ids = set()
        for match in tqdm(matches, desc="迁移高相似度What节点"):
            success = self._migrate_what_node_with_relations(
                target_session,
                source_session,
                match["what_id"],
                match["cluster_id"],
                relation_type="BELONGS_TO",
                relation_props={"similarity": match["similarity"]}
            )
            if success:
                migrated_count += 1
                matched_what_ids.add(match["what_id"])
                print(f"成功迁移What节点 {match['what_id']} 到Cluster {match['cluster_id']}")
            else:
                print(f"迁移What节点 {match['what_id']} 失败")
        
        print(f"\n总共迁移了 {migrated_count} 个高相似度What节点及其相关节点")
        return migrated_count, matched_what_ids
    
    def _migrate_unmatched_what_nodes(self, target_session, source_session, matched_what_ids: set[str]):
        """迁移未匹配的What节点及其相关节点到目标数据库"""
        print("\n开始迁移未匹配的What节点...")
        
        # 1. 获取源中未被匹配的What节点
        print("从源数据库获取未被匹配的What节点...")
        query = """
        MATCH (w:What)
        WHERE NOT (w)-[:BELONGS_TO]->(:Cluster)
        RETURN elementId(w) as id, properties(w) as props
        """
        result = source_session.run(query)
        unmatched_whats = [{"id": record["id"], "props": record["props"]} for record in result]
        
        if not unmatched_whats:
            print("没有未匹配的What节点需要迁移")
            return 0
        
        # 过滤掉已经匹配的What节点
        unmatched_whats = [what for what in unmatched_whats if what["id"] not in matched_what_ids]
        
        if not unmatched_whats:
            print("所有未匹配的What节点已被处理")
            return 0
        
        print(f"找到 {len(unmatched_whats)} 个未匹配的What节点需要迁移")
        
        # 2. 迁移这些What节点及其相关节点
        migrated_count = 0
        for what in tqdm(unmatched_whats, desc="迁移未匹配的What节点"):
            try:
                what_data = self._get_what_node_with_relations(source_session, what["id"])
                if not what_data or not what_data["what"]:
                    continue
                
                # 在目标数据库中创建What节点(不关联到Cluster)
                create_what_query = """
                MERGE (w:What {name: $what_name})
                ON CREATE SET w += $props
                RETURN elementId(w) as new_what_id
                """
                
                result = target_session.run(
                    create_what_query,
                    what_name=what_data["what"].get("name", ""),
                    props=what_data["what"]
                )
                
                new_what_id = result.single()["new_what_id"]
                
                # 迁移相关节点
                for related_node in what_data["related_nodes"]:
                    node_labels = related_node["labels"]
                    node_props = related_node["props"]
                    rel_type = related_node["rel_type"]
                    rel_props = related_node["rel_props"]
                    
                    unique_props = []
                    if "name" in node_props:
                        unique_props.append(("name", node_props["name"]))
                    elif "id" in node_props:
                        unique_props.append(("id", node_props["id"]))
                    else:
                        unique_props = [(k, v) for k, v in node_props.items() if v is not None]
                    
                    merge_conditions = []
                    params = {}
                    for i, (key, value) in enumerate(unique_props):
                        merge_conditions.append(f"{key}: $unique_val_{i}")
                        params[f"unique_val_{i}"] = value
                    
                    create_node_query = f"""
                    MATCH (w:What) WHERE elementId(w) = $what_id
                    MERGE (n:{':'.join(node_labels)} {{{', '.join(merge_conditions)}}})
                    ON CREATE SET n += $props
                    ON MATCH SET n += $props
                    MERGE (w)-[r:{rel_type}]->(n)
                    ON CREATE SET r = $rel_props
                    ON MATCH SET r += $rel_props
                    """
                    target_session.run(
                        create_node_query,
                        what_id=new_what_id,
                        props=node_props,
                        rel_props=rel_props,
                        **params
                    )
                
                migrated_count += 1
            
            except Exception as e:
                print(f"迁移未匹配的What节点 {what['id']} 失败: {str(e)}")
        
        print(f"\n总共迁移了 {migrated_count} 个未匹配的What节点及其相关节点")
        return migrated_count
    
    def _migrate_all_clusters(self, source_db: str, target_db: str, stats: dict, clear_source: bool = False) -> dict:
        """当目标数据库为空时，直接迁移所有源Cluster节点"""
        print("目标数据库为空，直接迁移所有源Cluster节点...")
        
        with self.driver.session(database=target_db) as target_session, \
             self.driver.session(database=source_db) as source_session:
            
            # 获取源数据库中的所有Cluster节点
            clusters_source = self._get_clusters_with_properties(source_db)
            stats["total_source_clusters"] = len(clusters_source)
            
            # 迁移所有Cluster节点
            migrated_count = self._migrate_unmatched_clusters(
                target_session,
                source_session,
                clusters_source,
                set()  # 没有已匹配的ID
            )
            
            stats["unmatched_clusters"] = migrated_count
            
            # 迁移What节点
            orphan_whats = self._get_orphan_what_nodes(source_db)
            stats["total_source_whats"] = len(orphan_whats)
            
            if orphan_whats:
                # 迁移所有What节点
                migrated_whats = self._migrate_unmatched_what_nodes(
                    target_session,
                    source_session,
                    set()  # 没有已匹配的ID
                )
                stats["unmatched_whats"] = migrated_whats
            
            # 可选: 清空源数据库
            if clear_source:
                print(f"\n正在清空 {source_db} 数据库...")
                self._clear_database(source_db)
                print(f"{source_db} 数据库已清空")
        
        return stats
    
    def _clear_database(self, database: str):
        """清空指定数据库"""
        with self.driver.session(database=database) as session:
            query = """
            MATCH (n)
            DETACH DELETE n;
            """
            session.run(query)
            print(f"已清空 {database} 数据库")


# 使用示例
if __name__ == "__main__":
    # 创建迁移服务实例
    migration_service = ClusterMigrationService(
        neo4j_uri="bolt://172.20.25.129:7687",
        neo4j_auth=("neo4j", "neo4j@openspg"),
        max_workers=4
    )
    
    # 执行迁移
    stats = migration_service.migrate_clusters(
        source_db="cache",
        target_db="all",
        similarity_threshold=0.8,
        why_weight=0.5,
        what_weight=0.5,
        what_similarity_threshold=0.95,
        clear_source=False  # 设置为True将在迁移完成后清空源数据库
    )
    
    # 打印统计信息
    print("\n迁移统计:")
    print(f"源数据库Cluster节点总数: {stats['total_source_clusters']}")
    print(f"匹配并迁移的Cluster节点数: {stats['matched_clusters']}")
    print(f"未匹配但迁移的Cluster节点数: {stats['unmatched_clusters']}")
    print(f"源数据库What节点总数: {stats['total_source_whats']}")
    print(f"匹配并迁移的What节点数: {stats['matched_whats']}")
    print(f"未匹配但迁移的What节点数: {stats['unmatched_whats']}")