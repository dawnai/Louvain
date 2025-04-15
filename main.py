"""
整个项目的main文件
"""
import logging
from tool.NewsProcessor import NewsProcessor
from tool.DataProcessor import DataPipeline_5w1h

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
        "batch_size": 5,
        "max_events_per_news": 3,#每条新闻最多抽取多少5w1h
        "request_interval": 1,
        "text_truncate_length": 1000#文本截断
    }
}


def main():

    #第一步新闻多5w1h抽取
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

    #第二步将结果存入neo4j数据库
    



if __name__ == "__main__":
    main()
