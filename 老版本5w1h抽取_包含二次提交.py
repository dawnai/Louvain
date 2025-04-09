"""
抽取新闻的5w1h并进行where和who的数据清洗
"""
import logging
import json
import openai
import pandas as pd
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置参数
INPUT_JSON = "./data/waite_to_extrac/chunk_10_EnAndChinese.json"
OUTPUT_XLSX_Alignment = "5w1h_results_Alignment.xlsx"
OUTPUT_JSON_Alignment = "5w1h_results_Alignment.json"
OUTPUT_XLSX = "5w1h_results.xlsx"
OUTPUT_JSON = "5w1h_results.json"
OPENAI_API_KEY = "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c"
MODEL_NAME = "qwen2.5-72b-instruct"#暨大：Qwen2.5-72B-Instruct 阿里云：qwen2.5-72b-instruct
BATCH_SIZE = 5 #批处理
REQUEST_DELAY = 1

# 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")#暨大：https://llm.jnu.cn/v1 阿里云：

def read_news_data(file_path):
    """读取并解析JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_date(date_str):
    """日期格式化"""
    if not date_str:
        return ""
    
    formats = [
        "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y/%m/%d",
        "%d %B %Y", "%B %d, %Y", "%Y%m%d", "%Y年%m月%d日",
        "%d-%b-%Y", "%b %d, %Y", "%Y.%m.%d"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # 尝试提取前导日期部分
    for length in [10, 8, 7]:
        truncated = date_str[:length]
        for fmt in ["%Y-%m", "%Y%m"]:
            try:
                dt = datetime.strptime(truncated, fmt)
                return dt.strftime("%Y-%m")
            except:
                continue
    
    return date_str[:10] if len(date_str) >= 10 else date_str

def process_time_fields(item):
    """处理时间"""
    when = item.get("when", "")#没有when 用空字符串替代
    
    if not when.strip():
        when = item["created_at"]
    
    return {
        "when": format_date(when),
        "created_at": format_date(item["created_at"])
    }

def extract_5w1h(item):
    """处理单个新闻条目"""
    # prompt = f"""
    #     Please extract the 5W1H  information from the following news content and return the result in JSON format with the fields: what, when, where, why, who, how.
    #     If a piece of information does not exist, set the corresponding field to an empty string. Don't add additional description.

    #     title:{item['title']}
    #     text:{item['text']}
    #     NewsTime:{item['created_at']}
    #     """
    prompt = f"""
        title:{item['title']}
        text:{item['text']}
        NewsTime:{item['created_at']}
        """

    # system_prompt = """
    #     You need to extract the 5w1h  (what, when, where, why, who, how) from the news, please note that you can properly utilize your own knowledge base to extract, the extracted content must be the official name, not aliases and nicknames.
    #     At the same time the extraction of what needs to be concise, can summarize the main content of the news, why and how the same reason. 
    #     who must be the extraction of the main characters of the news, there can be more than one character,If there are multiple individuals, simply separate them with commas.The extraction of individuals must be based on personal names and not job titles. If only job titles are mentioned in the news, you are required to provide specific personal names based on your own understanding.
    #     when extraction must be accurate to the month and year and day,and You must return the date in the format of Year-Month-Day.ease note that "NewsTime" is not the time when the event occurred, but the time when the news was published. You can infer the time of the event based on this publication time. if there is no accurate time, please set to the empty string.
    #     The extraction of where must be accurate to a specific city or street, such as New York, Washington, the White House and so on, can not be a wide range of areas, such as the United States, Europe and so on. There can be more than one location, but it must be the main place where the news is happening. If the exact address does not exist, set it to an empty string.        
    #     Also, if 5w1h  does not exist, set the corresponding field to an empty string and do not add additional notes!
    #     final, returns results in JSON format strictly on demand.
    #     """
    system_prompt = """
        ### 1.概述

        您是一个顶级算法，旨在从新闻中提取5w1h结构化格式信息，以构建5w1h知识图谱。

        5w1h的内容包括：what、when、where、why、who、how。

        目的是实现5w1h知识图谱的简单性和清晰性，使其可供广大受众使用。

        ### 2.要求

        what：确保可以总结新闻的主要内容，你可以适当参考title，确保尽可能精简。

        how：确保可以总结新闻事件的处理手段以及结果，尽可能保持精简和详细

        why：确保可以总结新闻事件发生的原因，尽可能保持精简和详细。

        who：确保人物名保持一致，如果某个人物(例如“John Doe”)在文本中多次提及，但使用不同的名称或代词(例如“Joe”、“he”),在整个知识图中始终使用该实体最完整的标识符，如果新闻事件中存在多个人物，请以逗号分隔开(例如:"唐纳德·特朗普,马丁·路德金,伊隆·马斯克"),请只抽取人物名字，并返回人物官方全称。

        where：确保是新闻事件中主要的发生地，并且精确到城市(例如“北京”、“上海”、“纽约”)，不要使用广泛地区名字(例如"美国"、“中国”、“欧洲”)。如果新闻事件中存在多个地名，请以逗号分隔开(例如："成都,北京,纽约")

        when：确保是新闻事件发生的时间，如果文中没有提及时间，或是模糊时间(例如:"前天","后天","上周")，你可以根据新闻的发布时间(NewsTime)推理出事件实际发生时间，并按照 年-月-日格式返回(例如："2025-01-19")

        注意：抽取结果应该是文本中的名称或人类可读的标识符

        "5w1h+:".join(allowed_nodes) if allowed_nodes else " "

        ### 3.共指解析

        .**维护实体一致性**:提取实体时，确保一致性至关重要，如果某个实体(例如“John Doe”)在文本中多次提及，但使用不同的名称或代词(例如“Joe”、“he”),在整个知识图中始终使用该实体最完整的标识符。在此示例中，使用“John Doe”作为实体 ID。请记住，知识图应该是连贯且易于理解的，因此保持实体引用的一致性至关重要

        ### 5.严格遵守

        严格遵守规则。不合规将导致终止。

        ### 6.返回格式

        以JSON格式返回：{"what":"", "when":"", "where":"", "why":"", "who":"", "how":""}
        """


    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n处理ID {item['id']} 时出错: {str(e)}")
        result = {key: "" for key in ["what", "when", "where", "why", "who", "how"]}
    
    # 处理时间字段
    time_data = process_time_fields({**item, **result})
    
    return {
        "id": item["id"],
        "title": item["title"],
        "created_at": time_data["created_at"],
        "text": item["text"],
        **{k: v.strip() for k, v in result.items()}
    }

def clean_entities(item):
    """清洗人物和地点信息"""
    # 跳过空值处理
    if not item['who'].strip() and not item['where'].strip():
        return item

    prompt = f"""
        
        Please clean up the formats of "who" and "where" based on the content in the text:
        who:{item['who'] or " "}
        where:{item['where'] or " "}
        text:{item['text'] or " "}
        """
    system_prompt="""
        You are a data cleaning expert, specializing in extracting standardized names
        
        Cleaning Requirements:
        Person: Based on the names of people mentioned in "who," search the "text" for their belong to organizations. If an organization is found, return it in the format "name-organization."(If the organization is a country name, please do not extract it) At the same time, clean up the format of the people by removing their job titles and retaining only their official full names. If no  organization is found in the "text," return only the person's full name. Additionally, 
        remove any individuals from "who" who do not have a name.Separate them with commas.
        Location: Specify to the city level (e.g., "United States" → "New York"). Separate multiple locations with commas.
        The response must be returned in the following JSON format:{{"cleaned_who": "", "cleaned_where": ""}}
        """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            # stream=True
            
        )
        cleaned = json.loads(response.choices[0].message.content)
        # 保留原始值作为fallback
        item['who'] = cleaned.get('cleaned_who', item['who'])#如果没有cleaned_who 则用item['who']替代
        item['where'] = cleaned.get('cleaned_where', item['where'])
    except Exception as e:
        print(f"\n清洗ID {item['id']} 时出错: {str(e)}")
    
    return item

def process_batch(batch, processor):
    """通用批处理函数"""
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(processor, item) for item in batch]
        return [future.result() for future in as_completed(futures)]

def pipeline_processing(data, processor, desc):
    """处理流水线"""
    processed = []
    total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with tqdm(total=len(data), desc=desc, unit="条") as pbar:
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            processed.extend(process_batch(batch, processor))
            pbar.update(len(batch))
            time.sleep(REQUEST_DELAY)
    
    return processed

def save_to_xlsx(data, output_path):
    """保存为Excel文件"""
    df = pd.DataFrame([{
        **item,
        "text": (item["text"][:497] + "...") if len(item["text"]) > 500 else item["text"]#截断一下，反正neo4j中不存储chunk
    } for item in data])
    
    columns_order = ['what', 'when', 'where', 'why', 'who', 'how', 'text', 'created_at','title']
    df[columns_order].to_excel(output_path, index=False, engine='openpyxl')

def save_to_json(data, output_path):
    """保存为JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 1. 数据读取
    logger.info("开始读取数据")
    news_data = read_news_data(INPUT_JSON)
    logger.info("读取完成，数据量为%s",len(news_data))

    # 2. 5W1H提取
    logger.info("开始执行5W1H抽取")
    extracted_data = pipeline_processing(news_data, extract_5w1h, "5w1h提取进度")
    save_to_xlsx(extracted_data, OUTPUT_XLSX)
    save_to_json(extracted_data, OUTPUT_JSON)
    logger.info("抽取结束，成功了保存至%s",OUTPUT_XLSX)

    # 3. 数据清洗
    # logger.info("开始执行数据清洗")
    # final_data = pipeline_processing(extracted_data, clean_entities, "where,who 清洗进度")
    # # 4. 结果保存
    # save_to_xlsx(final_data, OUTPUT_XLSX_Alignment)
    # save_to_json(final_data, OUTPUT_JSON_Alignment)
    
    print(f"\n处理完成！结果已保存至：\n- Excel文件: {OUTPUT_XLSX}\n- JSON文件: {OUTPUT_JSON}")
