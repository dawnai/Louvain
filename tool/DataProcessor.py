"""
数据加载类
"""
from tool.NewsProcessor import NewsProcessor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pandas as pd
import json




class DataPipeline_5w1h:
    """
    5w1h事件抽取的数据加载类和流程执行类
    包含：
        run_pipeline：执行抽取流程
        save_results：保存结果为json和xlsx
    """
    def __init__(self,CONFIG):
        self.CONFIG=CONFIG


    def run_pipeline(self,news_data):
        processor = NewsProcessor(self.CONFIG)
        results = []
        
        with tqdm(total=len(news_data), desc="新闻处理进度") as pbar:
            with ThreadPoolExecutor(max_workers=self.CONFIG["processing"]["batch_size"]) as executor:
                # 批量提交任务（每次提交batch_size个）
                batches = [news_data[i:i+self.CONFIG["processing"]["batch_size"]] 
                          for i in range(0, len(news_data), self.CONFIG["processing"]["batch_size"])]
                
                for batch in batches:
                    # 提交当前批次任务
                    futures = [executor.submit(processor.process_news_item, item) for item in batch]
                    
                    # 处理当前批次结果
                    for future in as_completed(futures):
                        try:
                            events = future.result()
                            if events:
                                results.extend(events)
                                pbar.update(1)
                        except Exception as e:
                            print(f"任务执行异常: {str(e)}")
                    
                    # 批次间添加间隔（替代逐个等待）
                    time.sleep(self.CONFIG["processing"]["request_interval"])
        
        return results
    
    def save_results(self,data):
        """保存处理结果"""
        # 保存JSON
        with open(self.CONFIG["output_files"]["json"], 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存Excel
        df = pd.DataFrame(data)
        column_order = [
            'event_id', 'news_id', 'what', 'why', 'how',
            'who', 'where', 'organization', 'when', 'news_time',
            'title', 'text'
        ]
        df[column_order].to_excel(self.CONFIG["output_files"]["excel"], index=False)
        
        print(f"结果已保存: {len(data)}条事件")
