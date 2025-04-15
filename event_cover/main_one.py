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

# 配置Neo4j连接
NEO4J_URI = "bolt://172.20.52.33:7687"  # 根据实际情况修改
NEO4J_AUTH = ("neo4j", "neo4j@openspg")   # 根据实际情况修改

# 初始化文本处理器
processor = TextProcessor(max_workers=4)  # 使用你提供的TextProcessor类

def get_clusters_with_properties(database: str) -> List[Dict]:
    """从指定数据库中获取带有why和what属性的Cluster节点"""
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        with driver.session(database=database) as session:
            query = """
            MATCH (c:Cluster)
            WHERE c.why IS NOT NULL AND c.what IS NOT NULL
            RETURN elementId(c) AS id, c.why AS why, c.what AS what, c.how AS how, c.organization AS organization, c.type AS type, c.when AS when, c.where AS where, c.who AS who
            """
            result = session.run(query)
            return [{"id": record["id"], "why": record["why"], "what": record["what"],"how":record["how"],"organization":record["organization"],"type":record["type"],"when":record["when"],"where":record["where"],"who":record["who"]} for record in result]

def get_related_nodes(session, cluster_id: str) -> Dict:
    """获取与Cluster节点相关的所有What节点及其关系"""
    query = """
    MATCH (w:What)-[r1]->(c:Cluster)
    WHERE elementId(c) = $cluster_id
    OPTIONAL MATCH (w)-[r2]->(n)
    WHERE NOT n:Cluster  // 排除Cluster节点本身
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
    ORDER BY what_id  // 确保顺序一致
    """
    result = session.run(query, cluster_id=cluster_id)
    
    # 按What节点分组相关数据
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
        if data["node_labels"]:  # 如果有相关节点
            grouped_data[what_id]["related_nodes"].append({
                "labels": data["node_labels"],
                "props": data["node_props"],
                "rel_type": data["rel2_type"],
                "rel_props": data["rel2_props"]
            })
    
    return list(grouped_data.values())


def migrate_cluster_with_relations(all_session, cache_session, all_cluster_id: str, cache_cluster_id: str):
    """将cache中的Cluster及其所有What节点和相关节点迁移到all数据库"""
    # 1. 获取cache中的Cluster节点属性
    get_cluster_query = """
    MATCH (c:Cluster)
    WHERE elementId(c) = $cluster_id
    RETURN properties(c) as props
    """
    cluster_result = cache_session.run(get_cluster_query, cluster_id=cache_cluster_id)
    cluster_data = cluster_result.single()
    
    if not cluster_data:
        print(f"警告: cache数据库中没有找到Cluster {cache_cluster_id}")
        return None
    
    # 2. 获取所有What节点及其相关节点
    what_nodes_data = get_related_nodes(cache_session, cache_cluster_id)
    
    if not what_nodes_data:
        print(f"警告: cache数据库中没有找到与Cluster {cache_cluster_id} 相关的What节点")
        return None
    
    # 3. 在all数据库中创建新Cluster节点
    create_cluster_query = """
    MATCH (source:Cluster) WHERE elementId(source) = $source_id
    CREATE (new:Cluster $props)
    CREATE (source)-[:SIMILAR_TO {similarity: $similarity}]->(new)
    RETURN elementId(new) as new_id
    """
    
    try:
        # 创建新Cluster节点并建立与源Cluster的关系
        new_cluster_id = all_session.run(create_cluster_query, 
                                       source_id=all_cluster_id,
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
            what_id = all_session.run(create_what_query,
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
                all_session.run(
                    create_node_query,
                    what_id=what_id,
                    props=node_props,
                    rel_props=rel_props,
                    **params
                )
        
        return new_cluster_id
    except Exception as e:
        print(f"迁移Cluster {cache_cluster_id} 失败: {str(e)}")
        return None


def migrate_unmatched_clusters(all_session, cache_session, clusters_cache, matched_cache_ids):
    """迁移未匹配的Cluster节点及其所有What节点和相关节点"""
    migrated_count = 0
    
    for cluster in tqdm(clusters_cache, desc="迁移未匹配Cluster节点"):
        if cluster["id"] in matched_cache_ids:
            continue
        
        # 使用完整属性（原cluster节点所有属性）
        create_cluster_query = """
        CREATE (new:Cluster $props)
        RETURN elementId(new) as new_id
        """
        try:
            # 创建新Cluster节点
            new_cluster_id = all_session.run(
                create_cluster_query,
                props=cluster
            ).single()["new_id"]
            
            # 获取所有What节点及其相关节点
            what_nodes_data = get_related_nodes(cache_session, cluster["id"])
            
            # 迁移所有What节点及其相关节点
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
                what_id = all_session.run(
                    create_what_query,
                    cluster_id=new_cluster_id,
                    what_name=what_data["what"].get("name", ""),
                    props=what_data["what"],
                    rel_props=what_data["rel_to_cluster"]["props"]
                ).single()["what_id"]
                
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
                    all_session.run(
                        create_node_query,
                        what_id=what_id,
                        props=node_props,
                        rel_props=rel_props,
                        **params
                    )
            
            migrated_count += 1
            print(f"成功迁移未匹配Cluster {cluster['id']} 到all数据库，新ID: {new_cluster_id}")
        
        except Exception as e:
            print(f"迁移未匹配Cluster {cluster['id']} 失败: {str(e)}")
            # 可以选择在这里添加回滚逻辑，删除已创建的部分节点
    
    return migrated_count

def calculate_combined_similarity_pairs(
    clusters_all: List[Dict], 
    clusters_cache: List[Dict],
    threshold: float = 0.9,
    why_weight: float = 0.5,
    what_weight: float = 0.5
) -> List[Tuple[Dict, Dict, float]]:
    """计算两个集群列表之间的综合相似度对"""
    # 提取所有why和what文本
    why_texts_all = [c["why"] for c in clusters_all]
    why_texts_cache = [c["why"] for c in clusters_cache]
    what_texts_all = [c["what"] for c in clusters_all]
    what_texts_cache = [c["what"] for c in clusters_cache]
    
    # 获取why和what的嵌入向量
    print("计算all数据库why属性的嵌入向量...")
    why_embeddings_all = processor.get_embeddings(why_texts_all)
    print("计算cache数据库why属性的嵌入向量...")
    why_embeddings_cache = processor.get_embeddings(why_texts_cache)
    
    print("计算all数据库what属性的嵌入向量...")
    what_embeddings_all = processor.get_embeddings(what_texts_all)
    print("计算cache数据库what属性的嵌入向量...")
    what_embeddings_cache = processor.get_embeddings(what_texts_cache)
    
    # 计算why和what的相似度矩阵
    print("计算why相似度矩阵...")
    why_similarity_matrix = cosine_similarity(why_embeddings_all, why_embeddings_cache)
    
    print("计算what相似度矩阵...")
    what_similarity_matrix = cosine_similarity(what_embeddings_all, what_embeddings_cache)
    
    # 计算综合相似度矩阵（加权平均）
    combined_similarity_matrix = why_weight * why_similarity_matrix + what_weight * what_similarity_matrix
    
    # 找出相似度大于阈值的对
    similar_pairs = []
    for i in tqdm(range(len(clusters_all)), desc="寻找相似对"):
        for j in range(len(clusters_cache)):
            sim = combined_similarity_matrix[i][j]
            if sim > threshold:
                # 同时记录各个分项相似度
                similar_pairs.append((
                    clusters_all[i], 
                    clusters_cache[j], 
                    sim,
                    why_similarity_matrix[i][j],
                    what_similarity_matrix[i][j]
                ))
    
    # 按综合相似度降序排序
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    return similar_pairs

def get_orphan_what_nodes(database: str) -> List[Dict]:
    """获取没有指向Cluster的What节点"""
    query = """
    MATCH (w:What)
    WHERE NOT (w)-[]->(:Cluster)
    RETURN elementId(w) AS id, w.name AS name, properties(w) AS properties
    """
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as driver:
        with driver.session(database=database) as session:
            result = session.run(query)
            return [{"id": record["id"], "name": record["name"], "properties": record["properties"]} 
                    for record in result]

def match_orphan_whats_to_clusters(all_session, cache_session):
    """匹配没有指向Cluster的What节点到all数据库中的Cluster节点"""
    print("\n开始匹配孤立的What节点到Cluster节点...")
    
    # 1. 获取cache中没有指向Cluster的What节点
    print("从cache数据库获取孤立的What节点...")
    orphan_whats = get_orphan_what_nodes("cache")
    if not orphan_whats:
        print("cache数据库中没有孤立的What节点")
        return
    
    print(f"找到 {len(orphan_whats)} 个孤立的What节点")
    
    # 2. 获取all数据库中所有Cluster的what属性
    print("从all数据库获取Cluster节点的what属性...")
    get_cluster_whats_query = """
    MATCH (c:Cluster)
    WHERE c.what IS NOT NULL
    RETURN elementId(c) AS cluster_id, c.what AS what_text
    """
    cluster_whats = all_session.run(get_cluster_whats_query)
    cluster_whats = [(record["cluster_id"], record["what_text"]) for record in cluster_whats]
    
    if not cluster_whats:
        print("all数据库中没有找到带有what属性的Cluster节点")
        return
    
    print(f"找到 {len(cluster_whats)} 个带有what属性的Cluster节点")
    
    # 3. 准备文本数据用于嵌入计算
    what_texts = [what["name"] for what in orphan_whats]
    cluster_texts = [text for _, text in cluster_whats]
    
    # 4. 计算嵌入向量
    print("计算What节点和Cluster节点的嵌入向量...")
    what_embeddings = processor.get_embeddings(what_texts)
    cluster_embeddings = processor.get_embeddings(cluster_texts)
    
    # 5. 计算相似度矩阵
    print("计算相似度矩阵...")
    similarity_matrix = cosine_similarity(what_embeddings, cluster_embeddings)
    
    # 6. 找出相似度>0.85的匹配对
    threshold = 0.85
    matches = []
    for i in range(len(orphan_whats)):
        for j in range(len(cluster_whats)):
            sim = similarity_matrix[i][j]
            if sim > threshold:
                matches.append({
                    "what_id": orphan_whats[i]["id"],
                    "what_name": orphan_whats[i]["name"],
                    "cluster_id": cluster_whats[j][0],
                    "cluster_what": cluster_whats[j][1],
                    "similarity": sim
                })
    
    # 7. 输出匹配结果
    if not matches:
        print("没有找到相似度>0.85的匹配对")
        return
    
    print(f"\n找到 {len(matches)} 对匹配的What节点和Cluster节点:")
    for match in sorted(matches, key=lambda x: x["similarity"], reverse=True):
        print(f"What节点(ID: {match['what_id']}, 名称: {match['what_name']})")
        print(f"匹配到Cluster节点(ID: {match['cluster_id']}, what属性: {match['cluster_what']})")
        print(f"相似度: {match['similarity']:.4f}\n")
    
    return matches
        

def get_what_node_with_relations(cache_session, what_id: str) -> Dict:
    """从cache数据库获取What节点及其所有相关节点"""
    query = """
    MATCH (w:What) WHERE elementId(w) = $what_id
    OPTIONAL MATCH (w)-[r]->(n)
    WHERE NOT n:Cluster  // 排除Cluster节点
    RETURN 
        properties(w) as what,
        type(r) as rel_type,
        properties(r) as rel_props,
        labels(n) as node_labels,
        properties(n) as node_props
    """
    result = cache_session.run(query, what_id=what_id)
    
    related_nodes = []
    what_data = None
    
    for record in result:
        data = record.data()
        if what_data is None:
            what_data = data["what"]
        
        if data["node_labels"]:  # 如果有相关节点
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

def migrate_what_node_with_relations(all_session, cache_session, what_id: str, cluster_id: str, 
                                   relation_type: str = "BELONGS_TO", relation_props: dict = None):
    """将What节点及其相关节点从cache迁移到all数据库"""
    try:
        # 1. 从cache获取What节点及其相关节点
        what_data = get_what_node_with_relations(cache_session, what_id)
        if not what_data or not what_data["what"]:
            print(f"警告: cache数据库中没有找到What节点 {what_id}")
            return False
        
        # 2. 在all数据库中创建What节点
        what_name = what_data["what"].get("name", "")
        create_what_query = f"""
        MATCH (c:Cluster) WHERE elementId(c) = $cluster_id
        MERGE (w:What {{name: $what_name}})
        ON CREATE SET w += $props
        MERGE (w)-[r:{relation_type}]->(c)
        ON CREATE SET r = $rel_props
        RETURN elementId(w) as new_what_id
        """
        
        result = all_session.run(
            create_what_query,
            cluster_id=cluster_id,
            what_name=what_name,
            props=what_data["what"],
            rel_props=relation_props or {}
        )
        
        new_what_id = result.single()["new_what_id"]
        print(f"成功创建What节点 {new_what_id} 并关联到Cluster {cluster_id}")
        
        # 3. 迁移相关节点
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
            all_session.run(
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


def match_and_migrate_high_similarity_whats(all_session, cache_session, threshold: float = 0.95):
    """匹配并迁移高相似度的What节点到all数据库中的Cluster节点"""
    print("\n开始匹配并迁移高相似度What节点...")
    
    # 1. 获取cache中没有指向Cluster的What节点
    print("从cache数据库获取孤立的What节点...")
    orphan_whats = get_orphan_what_nodes("cache")
    if not orphan_whats:
        print("cache数据库中没有孤立的What节点")
        return 0, set()  # 返回计数和空集合
    
    print(f"找到 {len(orphan_whats)} 个孤立的What节点")
    
    # 2. 获取all数据库中所有Cluster的what属性
    print("从all数据库获取Cluster节点的what属性...")
    get_cluster_whats_query = """
    MATCH (c:Cluster)
    WHERE c.what IS NOT NULL
    RETURN elementId(c) AS cluster_id, c.what AS what_text
    """
    cluster_whats = all_session.run(get_cluster_whats_query)
    cluster_whats = [(record["cluster_id"], record["what_text"]) for record in cluster_whats]
    
    if not cluster_whats:
        print("all数据库中没有找到带有what属性的Cluster节点")
        return 0, set()  # 返回计数和空集合
    
    print(f"找到 {len(cluster_whats)} 个带有what属性的Cluster节点")
    
    # 3. 准备文本数据用于嵌入计算
    what_texts = [what["name"] for what in orphan_whats if what.get("name")]
    cluster_texts = [text for _, text in cluster_whats]
    
    # 4. 计算嵌入向量
    print("计算What节点和Cluster节点的嵌入向量...")
    what_embeddings = processor.get_embeddings(what_texts)
    cluster_embeddings = processor.get_embeddings(cluster_texts)
    
    # 5. 计算相似度矩阵
    print("计算相似度矩阵...")
    similarity_matrix = cosine_similarity(what_embeddings, cluster_embeddings)
    
    # 6. 找出相似度>threshold的匹配对
    matches = []
    for i in range(len(orphan_whats)):
        what_name = orphan_whats[i].get("name")
        if not what_name:  # 跳过没有name的What节点
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
        return 0, set()  # 返回计数和空集合
    
    print(f"\n找到 {len(matches)} 对相似度>{threshold}的What节点和Cluster节点:")
    for match in sorted(matches, key=lambda x: x["similarity"], reverse=True):
        print(f"What节点(ID: {match['what_id']}, 名称: {match['what_name']})")
        print(f"匹配到Cluster节点(ID: {match['cluster_id']}, what属性: {match['cluster_what']})")
        print(f"相似度: {match['similarity']:.4f}\n")
    
    # 7. 迁移匹配的What节点
    migrated_count = 0
    matched_what_ids = set()
    for match in tqdm(matches, desc="迁移高相似度What节点"):
        success = migrate_what_node_with_relations(
            all_session,
            cache_session,
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

def migrate_unmatched_what_nodes(all_session, cache_session, matched_what_ids: set[str]):
    """迁移未匹配的What节点及其相关节点到all数据库"""
    print("\n开始迁移未匹配的What节点...")
    
    # 1. 获取cache中未被匹配的What节点
    print("从cache数据库获取未被匹配的What节点...")
    query = """
    MATCH (w:What)
    WHERE NOT (w)-[:BELONGS_TO]->(:Cluster)
    RETURN elementId(w) as id, properties(w) as props
    """
    result = cache_session.run(query)
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
            # 获取What节点及其相关节点
            what_data = get_what_node_with_relations(cache_session, what["id"])
            if not what_data or not what_data["what"]:
                print(f"警告: 无法获取What节点 {what['id']} 的详细信息")
                continue
            
            # 在all数据库中创建What节点(不关联到Cluster)
            create_what_query = """
            MERGE (w:What {name: $what_name})
            ON CREATE SET w += $props
            RETURN elementId(w) as new_what_id
            """
            
            result = all_session.run(
                create_what_query,
                what_name=what_data["what"].get("name", ""),
                props=what_data["what"]
            )
            
            new_what_id = result.single()["new_what_id"]
            print(f"成功创建未关联的What节点 {new_what_id}")
            
            # 迁移相关节点
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
                all_session.run(
                    create_node_query,
                    what_id=new_what_id,
                    props=node_props,
                    rel_props=rel_props,
                    **params
                )
                print(f"成功创建相关节点 {node_labels} 并关联到What节点 {new_what_id}")
            
            migrated_count += 1
        
        except Exception as e:
            print(f"迁移未匹配的What节点 {what['id']} 失败: {str(e)}")
    
    print(f"\n总共迁移了 {migrated_count} 个未匹配的What节点及其相关节点")
    return migrated_count
# 修改后的main函数
def main():
    # 从两个数据库获取数据
    print("从all数据库获取Cluster节点...")
    clusters_all = get_clusters_with_properties("all")
    print(f"找到 {len(clusters_all)} 个带有why和what属性的Cluster节点")
    
    print("从cache数据库获取Cluster节点...")
    clusters_cache = get_clusters_with_properties("cache")
    print(f"找到 {len(clusters_cache)} 个带有why和what属性的Cluster节点")
    
    if not clusters_all or not clusters_cache:
        print("一个或两个数据库中没有找到带有why和what属性的Cluster节点")
        return
    
    # 计算相似对
    similar_pairs = calculate_combined_similarity_pairs(
        clusters_all, 
        clusters_cache,
        threshold=0.8,
        why_weight=0.5,
        what_weight=0.5
    )
    
    # 输出结果
    print(f"\n找到 {len(similar_pairs)} 对综合相似度 > 0.8 的Cluster节点:")
    
    # 创建数据库驱动
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    # 迁移相似节点及其相关节点
    migrated_count = 0
    with driver.session(database="all") as all_session, \
         driver.session(database="cache") as cache_session:
        
        for pair in similar_pairs:
            cluster_all, cluster_cache, combined_sim, why_sim, what_sim = pair
            
            print(f"\n处理节点对 - 综合相似度: {combined_sim:.4f}")
            print(f"all数据库节点ID: {cluster_all['id']}")
            print(f"cache数据库节点ID: {cluster_cache['id']}")
            
            # 迁移Cluster及其相关节点
            new_cluster_id = migrate_cluster_with_relations(
                all_session, 
                cache_session,
                cluster_all["id"],
                cluster_cache["id"]
            )
            
            if new_cluster_id is not None:
                migrated_count += 1
                print(f"成功迁移Cluster {cluster_cache['id']} 到all数据库，新ID: {new_cluster_id}")
            else:
                print(f"迁移Cluster {cluster_cache['id']} 失败")
        
        # 迁移相似节点后的新增逻辑
        # 收集已匹配的cache集群ID
        matched_cache_ids = {pair[1]["id"] for pair in similar_pairs}
        
        print("\n开始迁移未匹配的Cluster节点...")
        unmatched_migrated = migrate_unmatched_clusters(
            all_session,
            cache_session,
            clusters_cache,
            matched_cache_ids
        )
        
        print(f"\n总共迁移了 {migrated_count} 个相似Cluster节点")
        print(f"迁移了 {unmatched_migrated} 个未匹配Cluster节点")
        print(f"总迁移节点数: {migrated_count + unmatched_migrated}/{len(clusters_cache)}")
        
        # # 调用新功能匹配孤立的What节点
        # match_orphan_whats_to_clusters(all_session, cache_session)
        
         # 新增功能：匹配并迁移高相似度What节点
        high_sim_migrated, matched_what_ids = match_and_migrate_high_similarity_whats(
            all_session,
            cache_session,
            threshold=0.95
        )
        print(f"\n迁移了 {high_sim_migrated} 个高相似度What节点及其相关节点")
        
        # 新增功能：迁移未匹配的What节点
        unmatched_migrated = migrate_unmatched_what_nodes(
            all_session,
            cache_session,
            matched_what_ids
        )
        print(f"\n迁移了 {unmatched_migrated} 个未匹配的What节点及其相关节点")
        print("抹除cache数据库")
        with driver.session(database="cache") as session:
            query = """
            MATCH (n)
            DETACH DELETE n;
            """
            result = session.run(query)
            print(result)
if __name__ == "__main__":
    main()

