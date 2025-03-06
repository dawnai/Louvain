"""
抽取新闻的5w1h
"""


import json
import openai
import pandas as pd
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
# 配置参数
INPUT_JSON = "./data/chunk_1.json"
OUTPUT_XLSX = "5w1h_results.xlsx"
OUTPUT_JSON = "5w1h_results.json"  # 新增JSON输出路径
OPENAI_API_KEY = "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c"
MODEL_NAME = "qwen2.5-72b-instruct"  # 可根据需要更改模型 deepseek-r1-distill-qwen-32b
BATCH_SIZE = 5
REQUEST_DELAY = 1

# 初始化OpenAI客户端
client = OpenAI(api_key=OPENAI_API_KEY,base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")#使用阿里云api

def read_news_data(file_path):
    """读取并解析JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_date(date_str):
    """将日期字符串统一格式化为YYYY-MM-DD格式"""
    if not date_str:
        return ""
    
    # 尝试多种日期格式解析
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",  # ISO格式
        "%Y/%m/%d",
        "%d %B %Y",            # 01 January 2023
        "%B %d, %Y",           # January 01, 2023
        "%Y%m%d"               # 20230101
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # 如果所有格式都失败，返回原始字符串前10个字符（适用于含时间的ISO格式）
    return date_str[:10] if len(date_str) >= 10 else date_str

def process_time_fields(item):
    """处理时间字段逻辑"""
    # 优先使用模型提取的时间
    when = item.get("when", "")
    
    # 当when为空时使用created_at
    if not when.strip():
        when = item["created_at"]
    
    # 统一格式化时间
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

    system_prompt=f"""
You are a professional information extraction assistant that returns results in JSON format strictly on demand.
    You need to extract the 5w1h (what, when, where, why, who, how) from the news, please note that you can properly utilize your own knowledge base to extract, the extracted content must be the official name, not aliases and nicknames.
    At the same time the extraction of what needs to be concise, can summarize the main content of the news, why and how the same reason. 
    who must be the extraction of the main characters of the news, there can be more than one character,If there are multiple individuals, simply separate them with commas.The extraction of individuals must be based on personal names and not job titles. If only job titles are mentioned in the news, you are required to provide specific personal names based on your own understanding.
    when extraction must be accurate to the month and year and day,and You must return the date in the format of Year-Month-Day.ease note that "NewsTime" is not the time when the event occurred, but the time when the news was published. You can infer the time of the event based on this publication time. if there is no accurate time, please set to the empty string.
    The extraction of where must be accurate to a specific city or street, such as New York, Washington, the White House and so on, can not be a wide range of areas, such as the United States, Europe and so on. There can be more than one location, but it must be the main place where the news is happening. If the exact address does not exist, set it to an empty string.
    Also, if 5w1h does not exist, set the corresponding field to an empty string and do not add additional notes!
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "system", "content": system_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"\n处理ID {item['id']} 时出错: {str(e)}")
        result = {key: "" for key in ["what", "when", "where", "why", "who", "how"]}
    
    # 处理时间字段
    time_data = process_time_fields({
        **item,
        **result
    })
    
    return {
        "id": item["id"],
        "title": item["title"],
        "created_at": time_data["created_at"],  # 格式化后的时间
        "text": item["text"],
        "what": result.get("what", ""),
        "when": time_data["when"],  # 格式化后的时间
        "where": result.get("where", ""),
        "why": result.get("why", ""),
        "who": result.get("who", ""),
        "how": result.get("how", ""),
    }

def process_batch(batch):
    """处理单个批次"""
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        return list(executor.map(extract_5w1h, batch))

def process_news(news_data):
    """批量处理所有新闻"""
    results = []
    
    total_batches = (len(news_data) + BATCH_SIZE - 1) // BATCH_SIZE
    with tqdm(total=len(news_data), desc="处理进度", unit="条") as progress_bar:
        for i in range(0, len(news_data), BATCH_SIZE):
            batch = news_data[i:i+BATCH_SIZE]
            batch_results = process_batch(batch)
            results.extend(batch_results)
            progress_bar.update(len(batch))
            time.sleep(REQUEST_DELAY)

    return results

def save_to_xlsx(data, output_path):
    """保存为Excel文件（包含截断文本）"""
    df = pd.DataFrame([{
        "what": item["what"],
        "when": item["when"],
        "where": item["where"],
        "why": item["why"],
        "who": item["who"],
        "how": item["how"],
        "text": item["text"][:500] + "..." if len(item["text"]) > 500 else item["text"],
        "created_at": item["created_at"]
    } for item in data])
    
    df = df[['what', 'when', 'where', 'why', 'who', 'how', 'text', 'created_at']]
    df.to_excel(output_path, index=False, engine='openpyxl')

def save_to_json(data, output_path):
    """保存为JSON文件（保留完整数据）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 1. 读取数据
    news_data = read_news_data(INPUT_JSON)
    
    # 2. 处理数据
    processed_data = process_news(news_data)
    
    # 3. 保存结果
    save_to_xlsx(processed_data, OUTPUT_XLSX)
    save_to_json(processed_data, OUTPUT_JSON)
    
    print(f"\n处理完成！结果已保存至：")
    print(f"- Excel文件: {OUTPUT_XLSX}")
    print(f"- JSON文件: {OUTPUT_JSON}")
