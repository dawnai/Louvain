"""
抽取新闻的5w1h并进行where和who的数据清洗
"""

import json
import openai
import pandas as pd
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 配置参数
INPUT_JSON = "./data/chunk_1.json"
OUTPUT_XLSX = "5w1h_results_pro.xlsx"
OUTPUT_JSON = "5w1h_results_pro.json"
OPENAI_API_KEY = "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c"
MODEL_NAME = "qwen2.5-72b-instruct"#暨大：Qwen2.5-72B-Instruct 阿里云：qwen2.5-72b-instruct
BATCH_SIZE = 5 #批处理
REQUEST_DELAY = 1

# 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

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
    """处理时间字段逻辑"""
    when = item.get("when", "")
    
    if not when.strip():
        when = item["created_at"]
    
    return {
        "when": format_date(when),
        "created_at": format_date(item["created_at"])
    }

def extract_5w1h(item):
    """处理单个新闻条目"""
    prompt = f"""Please extract the 5W1H information from the following news content and return the result in JSON format with the fields: what, when, where, why, who, how.
If a piece of information does not exist, set the corresponding field to an empty string. Don't add additional description.

title:{item['title']}
text:{item['text']}
NewsTime:{item['created_at']}
"""

    system_prompt = """
    You need to extract the 5w1h (what, when, where, why, who, how) from the news, please note that you can properly utilize your own knowledge base to extract, the extracted content must be the official name, not aliases and nicknames.
    At the same time the extraction of what needs to be concise, can summarize the main content of the news, why and how the same reason. 
    who must be the extraction of the main characters of the news, there can be more than one character,If there are multiple individuals, simply separate them with commas.The extraction of individuals must be based on personal names and not job titles. If only job titles are mentioned in the news, you are required to provide specific personal names based on your own understanding.
    when extraction must be accurate to the month and year and day,and You must return the date in the format of Year-Month-Day.ease note that "NewsTime" is not the time when the event occurred, but the time when the news was published. You can infer the time of the event based on this publication time. if there is no accurate time, please set to the empty string.
    The extraction of where must be accurate to a specific city or street, such as New York, Washington, the White House and so on, can not be a wide range of areas, such as the United States, Europe and so on. There can be more than one location, but it must be the main place where the news is happening. If the exact address does not exist, set it to an empty string.
    Also, if 5w1h does not exist, set the corresponding field to an empty string and do not add additional notes!
    final, returns results in JSON format strictly on demand.
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

    prompt = f"""Please convert the following information into its official full name.
who:{item['who'] or " "}
where:{item['where'] or " "}

"""
    system_prompt="""
    You are a data cleaning expert, specializing in extracting standardized names
    
    Cleaning Requirements:
    Person: Retain only the full names of individuals, removing titles and positions. If there are multiple names, separate them with commas.
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
            response_format={"type": "json_object"}
        )
        cleaned = json.loads(response.choices[0].message.content)
        # 保留原始值作为fallback
        item['who'] = cleaned.get('cleaned_who', item['who'])
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
        "text": (item["text"][:497] + "...") if len(item["text"]) > 500 else item["text"]
    } for item in data])
    
    columns_order = ['what', 'when', 'where', 'why', 'who', 'how', 'text', 'created_at']
    df[columns_order].to_excel(output_path, index=False, engine='openpyxl')

def save_to_json(data, output_path):
    """保存为JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 1. 数据读取
    news_data = read_news_data(INPUT_JSON)
    
    # 2. 5W1H提取
    extracted_data = pipeline_processing(news_data, extract_5w1h, "提取进度")
    
    # 3. 数据清洗
    final_data = pipeline_processing(extracted_data, clean_entities, "清洗进度")
    
    # 4. 结果保存
    save_to_xlsx(final_data, OUTPUT_XLSX)
    save_to_json(final_data, OUTPUT_JSON)
    
    print(f"\n处理完成！结果已保存至：\n- Excel文件: {OUTPUT_XLSX}\n- JSON文件: {OUTPUT_JSON}")
