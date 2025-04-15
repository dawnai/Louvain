"""
主函数
1、导出数据：what节点、what节点属性name why how、time节点、who节点、where节点
2、分别计算语义相似度权重和关系权重
3、构建louvain需要的图
4、进行社区发现
5、结果分析
6、写回neo4j数据库
"""
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from tool.Neo4jLouvainProcessor import Neo4jLouvainProcessor


if __name__ == "__main__":
    # neo4j数据库配置信息
    config = {
        "uri": "bolt://172.20.52.33:7687",#neo4j地址
        "user": "neo4j",
        "password": "neo4j@openspg",
        "db_name": "all",#neo4j数据库
        "semantic_threshold": 0.8,  #语义相似度阈值
        "embedding_uri":'https://embedding.jnu.cn/v1',#ollama地址http://172.20.71.112:11434 暨大：https://embedding.jnu.cn/v1
        "embedding_name":'bge-m3'#模型
    }

    processor = Neo4jLouvainProcessor(**config)

    try:
        # Step 1: 数据导出
        processor.export_nodes(all=True)
        processor.find_semantic_pairs()
        processor.fetch_relations()
        processor.calculate_weights()

        
        # Step 3: 构建图
        processor.build_graph()
        
        # Step 4: 社区发现
        processor.detect_communities()
        
        # Step 5: 结果分析
        processor.analyze_results()
        
        # Step 6: 写回结果
        processor.write_results()

    except Exception as e:
        logger.error(f"流程执行失败: {str(e)}")
    finally:
        processor.close()
        logger.info("处理完成，连接已关闭")