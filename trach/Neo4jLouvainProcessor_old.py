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
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class Neo4jLouvainProcessor:
    #resolution调整louvain的分辨率，数字越大，社区越小。random_state：随机种子 
    def __init__(self, uri, user, password, db_name,ollama_uri,model_name,semantic_threshold=0.4,louvain_params={'resolution': 3.0, 'random_state': 42}):
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            max_connection_lifetime=30,
            keep_alive=True
        )
        self.db_name = db_name
        self.semantic_threshold = semantic_threshold #语义相似度阈值
        self.ollama_uri= ollama_uri
        self.model_name= model_name
        self.text_processor = TextProcessor(self.ollama_uri,self.model_name)
        
        self.nodes_df = None
        self.edges_df = None
        self.G = None
        self.partition = None
        self.louvain_params = louvain_params  #Louvain参数

    def close(self):
        self.driver.close()

    def export_data(self):
        """导出所有What节点并生成全连接边"""
        logger.info("开始从Neo4j导出数据...")
        
        # 导出What节点
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
        
        # 导出已有的关联关系（通过共同节点）
        rel_query = """
        MATCH (a:What)-[r1]->(common)<-[r2]-(b:What)
        WHERE id(a) < id(b)
        RETURN 
            id(a) AS source,
            id(b) AS target,
            COUNT(DISTINCT common) AS weight,
            COLLECT(DISTINCT type(r1)) AS r1_types,
            COLLECT(DISTINCT type(r2)) AS r2_types
        """
        with self.driver.session(database=self.db_name) as session:
            existing_edges = pd.DataFrame([dict(record) for record in session.run(rel_query)])
        
        # 生成全连接边（所有可能的What节点对）
        all_pairs = list(combinations(self.nodes_df['node_id'].tolist(), 2))
        full_edges = pd.DataFrame(all_pairs, columns=['source', 'target'])
        full_edges['weight'] = 0
        full_edges['r1_types'] = [[]]*len(full_edges)
        full_edges['r2_types'] = [[]]*len(full_edges)
        
        # 合并时重置索引，不然会报错
        self.edges_df = pd.concat(
            [existing_edges, full_edges], 
            ignore_index=True
        ).drop_duplicates(['source', 'target'])
        
        logger.info(f"总边数（含全连接）: {len(self.edges_df)}")

    def calculate_semantic_weights(self):
        """计算全连接语义权重"""
        logger.info("计算语义相似度（why×2 + how×1 +name×3 + title×4）...")
        
        # 合并文本并生成向量
        why_texts = self.nodes_df['why_text'].fillna('').tolist()
        how_texts = self.nodes_df['how_text'].fillna('').tolist()
        name_texts = self.nodes_df['name_text'].fillna('').tolist()  # 新增name
        title_texts = self.nodes_df['title_text'].fillna('').tolist()  # 新增title
        
        why_embeddings = self.text_processor.get_embeddings(why_texts)
        how_embeddings = self.text_processor.get_embeddings(how_texts)
        name_embeddings = self.text_processor.get_embeddings(name_texts)  # 新增name
        title_embeddings = self.text_processor.get_embeddings(title_texts)  # 新增title
        
        # 计算相似度矩阵
        why_sim = cosine_similarity(why_embeddings)
        how_sim = cosine_similarity(how_embeddings)
        name_sim = cosine_similarity(name_embeddings)  # 新增name
        title_sim = cosine_similarity(title_embeddings)  # 新增title
        
        # 创建节点ID到索引的映射
        node_id_to_idx = {nid: idx for idx, nid in enumerate(self.nodes_df['node_id'])}
        
        # 计算语义权重 why:how:name:title=2:1:4:3
        self.edges_df['semantic_weight'] = self.edges_df.apply(
            lambda row: (
                0.2 * why_sim[node_id_to_idx[row['source']], node_id_to_idx[row['target']]] +
                0.1 * how_sim[node_id_to_idx[row['source']], node_id_to_idx[row['target']]] +
                0.4 * name_sim[node_id_to_idx[row['source']], node_id_to_idx[row['target']]] +
                0.3 * title_sim[node_id_to_idx[row['source']], node_id_to_idx[row['target']]] 
            ),
            axis=1
        )

    def calculate_relation_weights(self):
        """使用加权平均计算关系权重"""
        logger.info("计算关系权重（加权平均）...")
        
        relation_weights = {
            "地点": 0.1,
            "参与者": 0.7,
            "时间": 0.1,
            "_DEFAULT": 0.0
        }
        
        # 计算加权平均
        self.edges_df['raw_relation_weight'] = self.edges_df.apply(
            lambda row: (
                # 计算关系类型权重的平均值
                sum(
                    relation_weights.get(rt, relation_weights["_DEFAULT"]) 
                    for rt in row['r1_types'] + row['r2_types']
                ) / max(len(row['r1_types'] + row['r2_types']), 1)  # 防止除零
            ) * row['weight'],  # 乘以边的原始权重
            axis=1
        )
        
        # 归一化到 [0,1]
        min_w = self.edges_df['raw_relation_weight'].min()
        max_w = self.edges_df['raw_relation_weight'].max()
        
        if max_w == min_w:
            self.edges_df['relation_weight'] = 0.5  # 全同值处理
        else:
            self.edges_df['relation_weight'] = (
                (self.edges_df['raw_relation_weight'] - min_w) / 
                (max_w - min_w)
            )
        
        # 清理中间列
        self.edges_df.drop(columns=['raw_relation_weight'], inplace=True)
        
        # 分析结果
        logger.info("加权平均后的关系权重统计:\n%s", 
                self.edges_df['relation_weight'].describe())

    def build_graph(self):
        """构建融合权重的图"""
        logger.info("构建加权图...")
        
        # 应用阈值过滤 语义权重：关系权重=7:3
        mask = self.edges_df['semantic_weight'] >= self.semantic_threshold
        self.edges_df['final_weight'] = 0.0
        self.edges_df.loc[mask, 'final_weight'] = (
            0.7*self.edges_df['semantic_weight'] + 
            0.3*self.edges_df['relation_weight']
        )
        
        # 过滤无效边
        self.edges_df = self.edges_df[self.edges_df['final_weight'] > 0]
        logger.info(f"有效边数: {len(self.edges_df)}")
        
        # 构建图
        self.G = nx.from_pandas_edgelist(
            self.edges_df,
            source='source',
            target='target',
            edge_attr='final_weight',
            create_using=nx.Graph()
        )
        
        # 添加孤立节点
        self.G.add_nodes_from(self.nodes_df['node_id'].tolist())

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
        logger.info(f"最终权重统计:\n{self.edges_df['final_weight'].describe()}")
