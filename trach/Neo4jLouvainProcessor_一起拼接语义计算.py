"""
主流程
"""
from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
from community import community_louvain
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import logging
from tool.TextProcessor import TextProcessor
from sklearn.neighbors import NearestNeighbors
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
    def export_nodes(self):
        """仅导出What节点"""
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
        """高效查找语义相似节点对并保存所有相似度值"""
        logger.info(f"开始语义相似度计算（阈值={self.semantic_threshold}）...")
        
        # 合并文本字段（原有逻辑）
        self.nodes_df['combined'] = (
            2*self.nodes_df['why_text'] + " " +
            1*self.nodes_df['how_text'] + " " +
            4*self.nodes_df['name_text'] + " " +
            3*self.nodes_df['title_text']
        )
        
        # 批量生成嵌入向量
        embeddings = self.text_processor.get_embeddings(
            self.nodes_df['combined'].tolist()
        )
        
        # 创建节点ID到索引的映射
        self.node_id_to_idx = {
            nid: idx for idx, nid in enumerate(self.nodes_df['node_id'])
        }
        
        # 使用近似最近邻搜索（优化n_neighbors）
        nbrs = NearestNeighbors(n_neighbors=50, metric='cosine').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # 收集相似节点对及其相似度（带阈值过滤和去重）
        self.semantic_pairs = []
        seen_pairs = set()  # 用于去重
        
        for idx, (dists, nbrs) in tqdm(enumerate(zip(distances, indices))):
            src_id = self.nodes_df.iloc[idx]['node_id']
            for d, nbr_idx in zip(dists, nbrs):
                if idx == nbr_idx:
                    continue  # 排除自身
                similarity = 1 - d
                if similarity >= self.semantic_threshold:
                    tgt_id = self.nodes_df.iloc[nbr_idx]['node_id']
                    # 去重：确保每个无序对只保留一次
                    if src_id < tgt_id:
                        pair = (src_id, tgt_id)
                    else:
                        pair = (tgt_id, src_id)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        self.semantic_pairs.append((pair[0], pair[1], similarity))
        
        logger.info(f"发现 {len(self.semantic_pairs)} 个节点对,相似度≥{self.semantic_threshold}")


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
        combined_texts = self.nodes_df['combined'].values
        
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
