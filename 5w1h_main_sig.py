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
INPUT_JSON = "./data/waite_to_extrac/south_china_sea_news_1273.json"
OUTPUT_XLSX = "5w1h_results.xlsx"
OUTPUT_JSON = "5w1h_results.json"

API_KEY = "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c"
BASE_URL="https://llm.jnu.cn/v1"#暨大：https://llm.jnu.cn/v1 阿里云：https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME = "Qwen2.5-72B-Instruct"#暨大：Qwen2.5-72B-Instruct 阿里云：qwen2.5-72b-instruct

BATCH_SIZE = 5 #批处理
REQUEST_DELAY = 1

# 初始化OpenAI客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

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
    prompt = f"""
        title:{item['title']}
        text:{item['text']}
        NewsTime:{item['created_at']}
        """
    system_prompt = """
        ### 1.概述

​        您是一个顶级算法，旨在从新闻中提取5w1h结构化格式信息和事件牵涉的organization，以构建新闻事件知识图谱。

​        5w1h的内容包括：what、when、where、why、who、how。

​        目的是实现新闻知识图谱的简单性和清晰性，使其可供广大受众使用。

        ### 2.节点

​        **what**：确保可以总结新闻的主要内容，你可以适当参考title，确保尽可能精简。

​        **how**：确保可以总结新闻事件的处理手段以及结果，尽可能保持精简和详细

​        **why**：确保可以总结新闻事件发生的原因，尽可能保持精简和详细。

​        **who**：确保人物名字保持一致，如果某个人物(例如“John Doe”)在文本中多次提及，但使用不同的名称或代词(例如“Joe”、“he”),在整个知识图中始终使用该实体最完整的标识符，如果新闻事件中存在多个人物，请以逗号分隔开(例如:"唐纳德·特朗普,马丁·路德金,伊隆·马斯克"),请只抽取人物名字，并返回人物官方全称。**注意：**如果文中提及人物的所属机构和公司，那么需要抽取公司的全称，以"-"符号分隔开，例如（"伊隆·马斯克-特斯拉","Elias Costianes-Justice Department"）。如果text中只提及了公司或机构，请不要抽取，将其设置为空字符串，保证who中一定包含人的名字

​        **where**：确保是新闻事件中主要的发生地，并且精确到城市(例如“北京”、“上海”、“纽约”)，不要使用广泛地区名字(例如"美国"、“中国”、“欧洲”)。如果新闻事件中存在多个地名，请以逗号分隔开(例如："成都,北京,纽约")

​        **when**：确保是新闻事件发生的时间，如果文中没有提及时间，或是模糊时间(例如:"前天","后天","上周")，你可以根据新闻的发布时间(NewsTime)推理出事件实际发生时间，并按照 年-月-日格式返回(例如："2025-01-19")

​        **organization**：确保是新闻事件所牵涉的组织，而且必须返回组织全称（例如:“国际卫生组织”，“世界贸易组织”）。

​        **注意**：抽取结果应该是文本中的名称或人类可读的标识符，若是上述相关节点在新闻中没有提及，将其设置为空字符串，不必做任何说明。


        ### 3.共指解析

​        **维护实体一致性**:提取实体时，确保一致性至关重要，如果某个实体(例如“John Doe”)在文本中多次提及，但使用不同的名称或代词(例如“Joe”、“he”),在整个知识图中始终使用该实体最完整的标识符。请记住，知识图应该是连贯且易于理解的，因此保持实体引用的一致性至关重要

        ### 5.严格遵守

​        严格遵守节点抽取规则。不合规将导致终止。

        ### 6.返回格式

​        以JSON格式返回：{"what":"", "when":"", "where":"", "why":"", "who":"", "how":"","organization"}

        ### 7.示例

        ---

        **input**：
        *"title"*: "涨电费、下架酒 加拿大对美国出手了！",

        *"created_at"*: "2025-03-11",

        *"text"*: "当地时间3月10日，加拿大两个省份接连宣布两项举措，作为对美国持续威胁对加商品增收关税的报复性措施的一部分。输美电力 涨价！当地时间3月10日，加拿大经济第一大省——安大略省政府宣布，即日起开始对输美电力征收25%的关税，作为对美国总统特朗普对加拿大商品征收关税的报复措施的一部分。安大略省表示，新征关税将使输往美国的电力每兆瓦时增加约10加元。美国从安大略省输入电力的州包括纽约州、密歇根州和明尼苏达州等北部边境州，共有大约150万客户。安大略省省长道格·福特表示，这些州的家庭或企业每月的电费将增加100加元。福特补充说：“暂停部分关税，在最后一刻做出豁免——这些都行不通。我们需要一劳永逸地结束混乱局面。”此外，该省还表示，征税将给安大略省带来每天30万至40万加元的收入，这笔收入将用于支持工人和企业。美制酒精饮品 下架！当地时间3月10日，加拿大不列颠哥伦比亚省省长戴维·伊比宣布，将下架该省酒类商店内所有美国制造的酒精饮品。戴维·伊比表示，此举是为了回应美国持续威胁对进入该国的加拿大商品加征关税，也是为了回应美国总统特朗普有意重新划定加拿大与美国边界等。戴维·伊比称，正在通过立法工具，回击即将到来的加征钢铝关税。美对加关税征收3天后暂缓.加拿大报复关税却未完全停手.当地时间3月3日，美国总统特朗普表示，美国对墨西哥和加拿大商品加征25%的关税将于3月4日生效。特朗普称，对墨西哥和加拿大的关税没有达成共识的空间。当地时间3月4日，加拿大总理特鲁多宣布，对从美国进口的产品征收25%的报复性关税。特鲁多表示，加拿大将按计划对价值1550亿加元的美国商品分阶段加征25%关税，其中针对价值300亿加元商品的关税4日生效，对另外价值1250亿加元商品的关税将在21天后生效。根据加拿大财政部公布的两阶段关税清单，第一阶段涵盖橙汁、酒类、服装等消费品，第二阶段包括农产品、钢铁、电动汽车等战略物资。但在3月6日，美对加关税征收不满3天时，特朗普签署修正案，延缓了对墨加两国的关税征收。加拿大方面则在当天表示，加拿大4月2日前不会对美国商品征收第二阶段报复性关税。3月9日，加候任总理卡尼在当选执政党新领导人胜选演讲中表示，将保持对美国的报复性关税措施。卡尼在胜选演讲中说：“美国觊觎我们的资源，我们的水，我们的土地，甚至我们的国家。一旦得逞，将摧毁我们的生活。”卡尼说，他的政府将保持对美国的报复性关税措施。加拿大要团结起来，不能让美国总统特朗普得逞。"

        ---

        **output：**

        {
        "what": "加拿大两省实施对美报复性措施：安大略省对输美电力征税，不列颠哥伦比亚省下架美国酒精饮品",
        "when": "2025-03-10",
        "where": "安大略省,不列颠哥伦比亚省",
        "why": "回应美国对加拿大商品加征关税的威胁及特朗普的政策",
        "who": "道格·福特,戴维·伊比,唐纳德·特朗普,特鲁多",
        "how": "安大略省对输美电力征收25%关税,不列颠哥伦比亚省下架美国产酒精饮品",
        "organization":"安大略省政府"
        "category":"民生"
        }
        ---
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
        result = {key: "" for key in ["what", "when", "where", "why", "who", "how","organization"]}
    
    # 处理时间字段
    time_data = process_time_fields({**item, **result})
    
    return {
        "id": item["id"],
        "title": item["title"],
        "created_at": time_data["created_at"],
        "text": item["text"],
        **{k: v.strip() for k, v in result.items()}
    }


def process_batch(batch, processor):
    """通用批处理函数"""
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(processor, item) for item in batch]
        return [future.result() for future in as_completed(futures)]

def pipeline_processing(data, processor, desc):
    """处理流水线"""
    processed = []
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
    
    columns_order = ['what', 'when', 'where', 'why', 'who', 'how', 'text', 'created_at','title','organization']
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

    
    print(f"\n处理完成！结果已保存至：\n- Excel文件: {OUTPUT_XLSX}\n- JSON文件: {OUTPUT_JSON}")
