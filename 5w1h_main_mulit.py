"""
新闻多事件5W1H抽取系统（支持单新闻多事件版）
"""
import logging
import json
import hashlib
import pandas as pd
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    "input_json": "./data/waite_to_extrac/1-11.json",
    "output_files": {
        "excel": "1-11.xlsx",
        "json": "1-11.json"
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

class NewsProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=CONFIG["api"]["key"], 
                            base_url=CONFIG["api"]["base_url"])
        
    @staticmethod
    def read_data(file_path):
        """读取并验证输入数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError("输入数据应为JSON数组")
                return data
        except Exception as e:
            logger.error(f"数据读取失败: {str(e)}")
            raise

    @staticmethod
    def generate_event_id(news_id, text, event_index):
        """生成唯一事件ID"""
        hash_str = hashlib.md5(f"{news_id}_{event_index}".encode()).hexdigest()[:8]
        return f"{news_id}-{hash_str}"

    def process_timestamp(self, date_str, ref_date):
        """统一时间处理"""
        def try_formats(s):
            formats = [
                "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y/%m/%d",
                "%d %B %Y", "%B %d, %Y", "%Y%m%d", "%Y年%m月%d日",
                "%d-%b-%Y", "%b %d, %Y", "%Y.%m.%d"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(s, fmt).date()
                except ValueError:
                    continue
            return None

        processed_date = try_formats(date_str) or try_formats(ref_date)
        return processed_date.strftime("%Y-%m-%d") if processed_date else ref_date[:10]

    def clean_entities(self, entity_str, entity_type):
        """实体清洗"""
        if not entity_str:
            return ""
        
        # 统一分隔符
        entities = entity_str.replace("，", ",").split(",")
        
        cleaned = []
        for ent in entities:
            ent = ent.strip()
            if not ent:
                continue
            
            # 处理特殊格式
            if entity_type == "who" and "-" in ent:
                parts = [p.strip() for p in ent.split("-") if p.strip()]
                if len(parts) >= 3:
                    ent = "-".join(parts[:3])  # 保留名称-职位-机构
                elif len(parts) == 2:
                    ent = f"{parts[0]}-未知-{parts[1]}"
            
            if entity_type == "where":
                ent = ent.rstrip("市").strip()  # 去除冗余后缀
            
            cleaned.append(ent)
        
        return ",".join(sorted(set(cleaned)))  # 去重排序

    def extract_events(self, item):
        """执行多事件抽取"""
        system_prompt = """
        ### 1.概述

​        您是一个顶级算法，旨在从新闻中提取5w1h结构化格式信息和事件牵涉的organization，以构建新闻事件知识图谱。

​        5w1h的内容包括：what、when、where、why、who、how。

​        目的是实现新闻知识图谱的简单性和清晰性，使其可供广大受众使用。

        ### 2.节点

​        **what**：确保可以总结新闻的主要内容，你可以适当参考title，确保尽可能精简。

​        **how**：确保可以总结新闻事件的处理手段以及结果，尽可能保持精简和详细

​        **why**：确保可以总结新闻事件发生的原因，尽可能保持精简和详细。

​        **who**：确保人物名字保持一致，如果某个人物(例如“John Doe”)在文本中多次提及，但使用不同的名称或代词(例如“Joe”、“he”),在整个知识图中始终使用该实体最完整的标识符，如果新闻事件中存在多个人物，请以逗号分隔开(例如:"唐纳德·特朗普,马丁·路德金,伊隆·马斯克"),请只抽取人物名字，并返回人物官方全称。**注意：**如果文中提及人物的所属机构和公司，那么需要抽取公司的全称和人物的职称，以"名字"+'-'+"职称"+'-'+"机构"格式返回，例如（"伊隆·马斯克-CEO-特斯拉,Elias Costianes-Chief-Justice Department"）。
who要么抽取三元组，要么只返回人物名字！确保who中的每一个字段一定包含人的名字。

​        **where**：确保是新闻事件中主要的发生地，并且精确到城市(例如“北京”、“上海”、“纽约”)，不要使用广泛地区名字(例如"美国"、“中国”、“欧洲”)。如果新闻事件中存在多个地名，请以逗号分隔开(例如："成都,北京,纽约")

​        **when**：确保是新闻事件发生的时间，如果文中没有提及时间，或是模糊时间(例如:"前天","后天","上周")，你可以根据新闻的发布时间(NewsTime)推理出事件实际发生时间，并按照 年-月-日格式返回(例如："2025-01-19")

​        **organization**：确保是新闻事件所牵涉的组织，而且必须返回组织全称（例如:“国际卫生组织”，“世界贸易组织”）。

​        **注意**：抽取结果应该是文本中的名称或人类可读的标识符，若是上述相关节点在新闻中没有提及，将其设置为空字符串，不必做任何说明。


        ### 3.共指解析

​        **维护实体一致性**:提取实体时，确保一致性至关重要，如果某个实体(例如“John Doe”)在文本中多次提及，但使用不同的名称或代词(例如“Joe”、“he”),在整个知识图中始终使用该实体最完整的标识符。请记住，知识图应该是连贯且易于理解的，因此保持实体引用的一致性至关重要

        ### 5.严格遵守

​        严格遵守节点抽取规则。不合规将导致终止。

        ### 6.返回格式
​        一条新闻可能包含多个事件,你需要抽取多个5w1h
        1. 直接返回数组，不要包裹在额外对象中
        2. 每个事件包含完整的字段
        3. 最多返回3个主要事件
        以JSON格式返回：[{"what":"", "when":"", "where":"", "why":"", "who":"", "how":"","organization"},{"what":"", "when":"", "where":"", "why":"", "who":"", "how":"","organization"}]
        ### 7.示例
        ---
        **input**：
            *"title"*: "涨电费、下架酒 加拿大对美国出手了！",
            *"created_at"*: "2025-03-11",
            *"text"*: "当地时间3月10日，加拿大两个省份接连宣布两项举措，作为对美国持续威胁对加商品增收关税的报复性措施的一部分。输美电力 涨价！当地时间3月10日，加拿大经济第一大省——安大略省政府宣布，即日起开始对输美电力征收25%的关税，作为对美国总统特朗普对加拿大商品征收关税的报复措施的一部分。安大略省表示，新征关税将使输往美国的电力每兆瓦时增加约10加元。美国从安大略省输入电力的州包括纽约州、密歇根州和明尼苏达州等北部边境州，共有大约150万客户。安大略省省长道格
        **output：**
        [{
        "what": "加拿大两省实施对美报复性措施：安大略省对输美电力征税，不列颠哥伦比亚省下架美国酒精饮品",
        "when": "2025-03-10",
        "where": "安大略省,不列颠哥伦比亚省",
        "why": "回应美国对加拿大商品加征关税的威胁及特朗普的政策",
        "who": "道格·福特,戴维·伊比,唐纳德·特朗普,特鲁多",
        "how": "安大略省对输美电力征收25%关税,不列颠哥伦比亚省下架美国产酒精饮品",
        "organization":"安大略省政府"
        }]
        ---
        """
        user_prompt = f"""请从以下新闻中提取关键事件信息：
        title: {item['title']}
        created_at: {item['created_at']}
        text: {item['text']}
        """
        
        for attempt in range(CONFIG["api"]["retries"]):
                try:
                    response = self.client.chat.completions.create(
                        model=CONFIG["api"]["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        response_format={"type": "json_object"},
                        timeout=CONFIG["api"]["timeout"]
                    )
                    
                    # 解析响应
                    result = json.loads(response.choices[0].message.content)
                    
                    # 处理不同格式的响应
                    if isinstance(result, list):
                        return result[:CONFIG["processing"]["max_events_per_news"]]
                    else:
                        logger.warning(f"未知响应格式: {type(result)}")
                        return []             
                except json.JSONDecodeError:
                    logger.warning("响应不是有效的JSON")
                    return []
                except Exception as e:
                    logger.warning(f"第{attempt+1}次尝试失败: {str(e)}")
                    time.sleep(2**attempt)
            
        return []

    def process_news_item(self, item):
        """处理单个新闻条目"""
        try:
            raw_events = self.extract_events(item)
            processed_events = []
            
            for idx, event in enumerate(raw_events):
                # 生成唯一事件ID
                event_id = self.generate_event_id(item["id"], item["text"], idx)
                
                # 时间处理
                event_time = self.process_timestamp(
                    event.get("when", ""),
                    item["created_at"]
                )
                
                # 构建事件对象
                processed_event = {
                    "event_id": event_id,
                    "news_id": item["id"],
                    "title": item["title"][:200],  # 限制标题长度
                    "news_time": item["created_at"],
                    "when": event_time,
                    "what": event.get("what", "")[:300],
                    "why": self.clean_entities(event.get("why", ""), "why"),
                    "how": event.get("how", "")[:500],
                    "who": self.clean_entities(event.get("who", ""), "who"),
                    "where": self.clean_entities(event.get("where", ""), "where"),
                    "organization": self.clean_entities(event.get("organization", ""), "org"),
                    "text": item["text"][:CONFIG["processing"]["text_truncate_length"]]
                }
                processed_events.append(processed_event)
            
            return processed_events
        
        except Exception as e:
            logger.error(f"处理新闻 {item.get('id')} 失败: {str(e)}")
            return []

class DataPipeline:
    @staticmethod
    def run_pipeline(news_data):
        processor = NewsProcessor()
        results = []
        
        with tqdm(total=len(news_data), desc="新闻处理进度") as pbar:
            with ThreadPoolExecutor(max_workers=CONFIG["processing"]["batch_size"]) as executor:
                # 批量提交任务（每次提交batch_size个）
                batches = [news_data[i:i+CONFIG["processing"]["batch_size"]] 
                          for i in range(0, len(news_data), CONFIG["processing"]["batch_size"])]
                
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
                            logger.error(f"任务执行异常: {str(e)}")
                    
                    # 批次间添加间隔（替代逐个等待）
                    time.sleep(CONFIG["processing"]["request_interval"])
        
        return results
    @staticmethod
    def save_results(data):
        """保存处理结果"""
        # 保存JSON
        with open(CONFIG["output_files"]["json"], 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 保存Excel
        df = pd.DataFrame(data)
        column_order = [
            'event_id', 'news_id', 'what', 'why', 'how',
            'who', 'where', 'organization', 'when', 'news_time',
            'title', 'text'
        ]
        df[column_order].to_excel(CONFIG["output_files"]["excel"], index=False)
        
        logger.info(f"结果已保存: {len(data)}条事件")

def main():
    logger.info("启动新闻处理流程")
    
    try:
        # 数据加载
        raw_data = NewsProcessor.read_data(CONFIG["input_json"])
        logger.info(f"成功加载 {len(raw_data)} 条新闻数据")
        
        # 执行处理
        processed_data = DataPipeline.run_pipeline(raw_data)
        logger.info(f"成功处理 {len(processed_data)} 条事件")
        
        # 结果保存
        DataPipeline.save_results(processed_data)
        
    except Exception as e:
        logger.error(f"流程异常终止: {str(e)}")
        raise

if __name__ == "__main__":
    main()
