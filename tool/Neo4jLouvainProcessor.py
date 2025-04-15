"""
主流程
"""
from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
from community import community_louvain
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from itertools import combinations
import logging
from tool.TextProcessor import TextProcessor
from sklearn.neighbors import NearestNeighbors
import numpy as np
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jLouvainProcessor:
    #resolution调整louvain的分辨率，数字越大，社区越小。random_state：随机种子 
    def __init__(self, uri, user, password, db_name,embedding_uri,embedding_name,semantic_threshold,louvain_params={'resolution': 3.0, 'random_state': 42}):
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            max_connection_lifetime=30,
            keep_alive=True
        )
        self.db_name = db_name
        self.semantic_threshold = semantic_threshold #语义相似度阈值
        self.embedding_uri= embedding_uri
        self.embedding_name= embedding_name
        self.text_processor = TextProcessor(self.embedding_uri,self.embedding_name)
        
        self.nodes_df = None
        self.edges_df = None
        self.G = None
        self.partition = None
        self.louvain_params = louvain_params  #Louvain参数

    def close(self):
        self.driver.close()
    
    # 判断是all数据库进行louvain还是其他数据库进行louvain
    def export_nodes(self,all):
        if all:
            """仅导出不指向Cluster的What节点"""
            logger.info("开始不指向Cluster的What节点...")
            node_query="""
            MATCH (n:What)
            WHERE NOT EXISTS((n)-[:BELONGS_TO]->(:Cluster))
            RETURN 
                id(n) AS node_id, 
                COALESCE(n.why, '') AS why_text,
                COALESCE(n.how, '') AS how_text,
                COALESCE(n.name, '') AS name_text,
                COALESCE(n.title, '') AS title_text
        """
        else:
            """导出所有What节点"""
            logger.info("开始导出What节点...")
            node_query = """
            MATCH (n:What)
            RETURN 
                id(n) AS node_id, 
                COALESCE(n.why, '') AS why_text,
                COALESCE(n.how, '') AS how_text,
                COALESCE(n.name, '') AS name_text,
                COALESCE(n.title, '') AS title_text
            """
        with self.driver.session(database=self.db_name) as session:
            result = session.run(node_query)
            self.nodes_df = pd.DataFrame([dict(record) for record in result])
        logger.info(f"共导出 {len(self.nodes_df)} 个What节点")

    def find_semantic_pairs(self):
        """带嵌入预计算和复用的混合方法"""
        logger.info("启动混合语义计算（嵌入复用版）...")
        
        # ===== 嵌入预计算 =====
        if not hasattr(self, 'cached_embeddings'):
            logger.info("预计算所有字段嵌入...")
            
            # 字段权重配置（与后续计算共用）
            self.field_weights = {
                'why_text': 0.4,
                'how_text': 0.2,
                'name_text': 0.4
            }
            
            # 预计算并缓存所有字段的归一化嵌入
            self.cached_embeddings = {}
            for field in self.field_weights.keys():
                texts = self.nodes_df[field].fillna('').tolist()
                raw_emb = self.text_processor.get_embeddings(texts)
                self.cached_embeddings[field] = normalize(raw_emb)
                
                # 内存优化：转换数据类型
                self.cached_embeddings[field] = self.cached_embeddings[field].astype(
                    np.float32, copy=False
                )
            
            # 计算综合嵌入（用于最近邻预筛）
            combined_emb = np.mean(list(self.cached_embeddings.values()), axis=0)
            self.cached_embeddings['_combined'] = normalize(combined_emb)
            
            logger.info(f"嵌入缓存完成，占用内存：{self._get_emb_size_mb()} MB")
        
        # ===== 第一阶段：最近邻预筛选 =====
        logger.info("阶段1：基于综合嵌入的最近邻预筛...")
        nbrs = NearestNeighbors(
            n_neighbors=100, 
            metric='cosine',
            algorithm='brute'  # 使用预计算嵌入时最优
        ).fit(self.cached_embeddings['_combined'])
        
        # 获取候选对（内存优化版）
        candidate_pairs = self._find_candidate_pairs(nbrs)
        
        # ===== 第二阶段：加权多字段精确计算 =====
        logger.info("阶段2：复用缓存的字段嵌入进行精确计算...")
        self.semantic_pairs = self._calculate_weighted_sims(candidate_pairs)
        
        logger.info(f"最终获得 {len(self.semantic_pairs)} 个有效对")

    def _find_candidate_pairs(self, nbrs):
        """高效生成候选对（修复版）"""
        # 获取节点ID数组
        node_ids = self.nodes_df['node_id'].values
        
        # 获取最近邻结果（需传入待查询数据）
        distances, indices = nbrs.kneighbors(self.cached_embeddings['_combined'])
        
        candidate_pairs = set()
        
        for i in range(len(indices)):
            src_id = node_ids[i]
            
            # 遍历每个节点的邻居（跳过自己）
            for d, j in zip(distances[i], indices[i]):
                if i == j:  # 排除自身
                    continue
                
                # 转换距离到相似度：cos_sim = 1 - cosine_distance
                if (1 - d) < (self.semantic_threshold * 0.8):  # 动态预筛选阈值
                    continue
                
                tgt_id = node_ids[j]
                # 使用排序元组去重
                pair = tuple(sorted((src_id, tgt_id)))
                candidate_pairs.add(pair)
        
        return candidate_pairs


    def _calculate_weighted_sims(self, candidate_pairs):
        """复用缓存的嵌入计算加权相似度"""
        id_to_idx = {nid: idx for idx, nid in enumerate(self.nodes_df['node_id'])}
        valid_pairs = []
        
        for (a, b) in candidate_pairs:
            idx_a = id_to_idx[a]
            idx_b = id_to_idx[b]
            
            total_sim = 0.0
            for field, weight in self.field_weights.items():
                emb = self.cached_embeddings[field]
                total_sim += weight * emb[idx_a].dot(emb[idx_b])
            
            if total_sim >= self.semantic_threshold:
                valid_pairs.append((a, b, round(total_sim, 4)))
        
        # 按相似度降序排序
        return sorted(valid_pairs, key=lambda x: -x[2])

    def _get_emb_size_mb(self):
        """计算嵌入缓存的内存占用量"""
        total = 0
        for emb in self.cached_embeddings.values():
            total += emb.nbytes
        return f"{total / (1024**2):.1f}"

    def fetch_relations(self, batch_size=1000):
            """批量获取关联边"""
            logger.info("开始查询关联边...")
            
            # 分批次处理节点对
            all_pairs = list(self.semantic_pairs)
            relation_edges = []
            
            for i in tqdm(range(0, len(all_pairs), batch_size)):
                batch = all_pairs[i:i+batch_size]
                
                cypher = """
                UNWIND $pairs AS pair
                MATCH (a)-[r1]->(common)<-[r2]-(b) 
                WHERE id(a) = pair[0] AND id(b) = pair[1]
                RETURN 
                    id(a) AS source,
                    id(b) AS target,
                    COUNT(DISTINCT common) AS weight,
                    COLLECT(DISTINCT type(r1)) AS r1_types,
                    COLLECT(DISTINCT type(r2)) AS r2_types
                """
                with self.driver.session(database=self.db_name) as session:
                    result = session.run(cypher, {"pairs": batch})
                    relation_edges.extend([dict(record) for record in result])
            
            self.edges_df = pd.DataFrame(relation_edges)
            logger.info(f"获取到 {len(self.edges_df)} 条关联边")

    def calculate_weights(self):
        """权重计算优化"""
        logger.info("计算权重")
        
        if self.edges_df.empty:
            return
        # 创建节点对到相似度的映射
        semantic_similarity_dict = {(src, tgt): sim for src, tgt, sim in self.semantic_pairs}
        
        # 语义权重缓存
        node_id_to_idx = {nid: idx for idx, nid in enumerate(self.nodes_df['node_id'])}
        # combined_texts = self.nodes_df['combined'].values
        
        # 使用生成器计算相似度
        def similarity_generator():
            for _, row in self.edges_df.iterrows():
                src_id = row['source']
                tgt_id = row['target']
                yield semantic_similarity_dict.get((src_id, tgt_id), 0.0)
        
        self.edges_df['semantic_weight'] = list(tqdm(
        similarity_generator(), 
        total=len(self.edges_df),
        desc="语义权重计算"
    ))
        
        # 关系权重处理
        relation_weights = {
            "地点": 0.1,
            "参与者": 0.7,
            "时间": 0.1,
            "_DEFAULT": 0.0
        }
        
        def calc_relation_weight(row):
            types = row['r1_types'] + row['r2_types']
            if not types:
                return 0.0
            total = sum(relation_weights.get(t, 0.0) for t in types)
            return total / len(types)
        
        self.edges_df['relation_weight'] = self.edges_df.apply(
            calc_relation_weight, axis=1
        )
        
        # 综合权重
        self.edges_df['final_weight'] = (
            0.7 * self.edges_df['semantic_weight'] +
            0.3 * self.edges_df['relation_weight']
        )
        
        logger.info("权重计算完成")

    def build_graph(self):
        """内存优化的图构建"""
        logger.info("构建稀疏图...")
        
        # 使用生成器逐步添加边
        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes_df['node_id'].tolist())
        
        for _, row in tqdm(self.edges_df.iterrows(), total=len(self.edges_df)):
            self.G.add_edge(
                row['source'], 
                row['target'],
                weight=row['final_weight']
            )
        
        logger.info(f"最终图结构建立完毕")

    def detect_communities(self):
        """执行Louvain社区发现"""
        logger.info("开始社区发现...")
        self.partition = community_louvain.best_partition(self.G, weight='final_weight',resolution=self.louvain_params['resolution'],random_state=self.louvain_params['random_state'])
        self.nodes_df['community'] = self.nodes_df['node_id'].map(self.partition)
        logger.info(f"发现 {self.nodes_df['community'].nunique()} 个社区")

    def write_results(self, batch_size=1000):
        """将结果写回Neo4j"""
        logger.info("开始写回结果...")
        
        # 清理旧数据
        cleanup_cypher = "MATCH (c:Cluster) DETACH DELETE c"

        with self.driver.session(database=self.db_name) as session:
            session.run(cleanup_cypher)
        
        # 写入社区属性
        data = self.nodes_df[['node_id', 'community']].to_dict('records')
        for i in tqdm(range(0, len(data), batch_size), desc="写入社区属性"):
            batch = data[i:i+batch_size]
            cypher = """
            UNWIND $batch AS row
            MATCH (n) WHERE id(n) = row.node_id
            SET n.community = row.community
            """
            with self.driver.session(database=self.db_name) as session:
                session.run(cypher, {'batch': batch})
        
        # 创建Cluster节点
        communities = self.nodes_df['community'].unique()
        for com_id in tqdm(communities, desc="创建聚类节点"):
            cypher = """
            CREATE (c:Cluster {community_id: $com_id})
            WITH c
            MATCH (n:What {community: $com_id})
            MERGE (n)-[:BELONGS_TO]->(c)
            """
            with self.driver.session(database=self.db_name) as session:
                session.run(cypher, {'com_id': com_id})
        
        logger.info("数据写回完成")

    def analyze_results(self):
        """分析结果"""
        # 社区规模统计
        community_size = self.nodes_df['community'].value_counts()
        logger.info(f"社区规模分布:\n{community_size.describe()}")
        # 边权重分析
        logger.info(f"语义权重统计:\n{self.edges_df['semantic_weight'].describe()}")
        logger.info(f"关系权重统计:\n{self.edges_df['relation_weight'].describe()}")
        logger.info(f"最终权重统计:\n{self.edges_df['final_weight'].describe()}")
