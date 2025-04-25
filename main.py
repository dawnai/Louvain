"""
整个项目的main文件
"""
import logging
from tool.NewsProcessor import NewsProcessor
from tool.DataProcessor import DataPipeline_5w1h
from tool.upload import runUpload
from tool.louvain import louvain_process
# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    "input_json": "./data/waite_to_extrac/1-11.json",
    "output_files": {
        "excel": "./data/waite_to_neo4j/xlsx/1-11.xlsx",
        "json": "./data/waite_to_neo4j/json/1-11.json"
    },
    "api": {
        "key": "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c",
        "base_url": "https://llm.jnu.cn/v1",
        "model": "Qwen2.5-72B-Instruct",
        "retries": 3,
        "timeout": 30
    },
    "processing": {
        #多5w1h抽取策略
        "batch_size": 5,
        "max_events_per_news": 3,#每条新闻最多抽取多少5w1h
        "request_interval": 1,
        "text_truncate_length": 1000#文本截断
    },
    "neo4j_config":{
        "uri": "bolt://172.20.35.23:7687",
        "user": "neo4j",
        "password": "neo4j@openspg",
        "database": "cache"
    },
    "target_columns":['what', 'where', 'when', 'who', 'why', 'how', 'title','organization','news_id'],#用于判断xlsx表格中哪些列需要被抽取
    "config_louvain":{
        "semantic_threshold": 0.8,  #语义相似度阈值
        "embedding_uri":'https://embedding.jnu.cn/v1',#ollama地址http://172.20.71.112:11434 暨大：https://embedding.jnu.cn/v1
        "embedding_name":'bge-m3'#模型
    }

}

def main():

    # # ================= 第一步 多5w1h抽取 =================
    logger.info("启动新闻多5w1h抽取流程")
    try:
        # 数据加载
        raw_data = NewsProcessor.read_data(CONFIG["input_json"])
        logger.info(f"成功加载 {len(raw_data)} 条新闻数据")
        DataPipeline=DataPipeline_5w1h(CONFIG)#实例化5w1h流程处理类

        # 执行处理
        processed_data = DataPipeline.run_pipeline(news_data=raw_data)
        logger.info(f"成功处理 {len(processed_data)} 条事件")
        
        # 结果保存
        DataPipeline.save_results(data=processed_data)
        
    except Exception as e:
        logger.error(f"流程异常终止: {str(e)}")
        raise
    logger.info("新闻多5w1h抽取流程结束")

    # # ================= 第二步 将5w1h上传至neo4j数据库 =================
    logger.info("开始执行第二步，将当日抽取的5w1h上传至neo4j数据库")
    runUpload(CONFIG)
    logger.info("第二步执行完成")

    # # ================= 第三步 执行louvain聚类 =================
    logger.info("开始执行第三步，将当日抽取的5w1h上传至neo4j数据库")
    louvain_process(CONFIG,all=False)
    # ================= 第四步 执行事件属性添加 =================
    


    # ================= 第五步 执行当日事件入库 =================





if __name__ == "__main__":
    main()
