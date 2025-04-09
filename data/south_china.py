# """
# title去重
# """
# import json
# import re

# def remove_duplicate_news(json_data):
#     seen_titles = set()
#     unique_news = []
#     for item in json_data:
#         # 标准化标题：移除非字母数字字符并转为小写
#         normalized_title = re.sub(r'[^\w]', '', item['title'].lower())
#         if normalized_title not in seen_titles:
#             seen_titles.add(normalized_title)
#             unique_news.append(item)
#     print(len(unique_news))
#     return unique_news

# # 加载原始JSON数据（假设数据存储在'news.json'中）
# with open('./south_china/filtered_output.json', 'r',encoding='utf-8') as f:
#     original_data = json.load(f)

# # 去除重复项
# filtered_data = remove_duplicate_news(original_data)

# # 保存处理后的数据到新文件
# # with open('filtered_news.json', 'w') as f:
# #     json.dump(filtered_data, f, indent=2)

# print("重复标题已删除，结果保存至 filtered_news.json")

"""
使用大模型筛选和南海相关的新闻
"""
import json
from openai import OpenAI
from tqdm import tqdm  # 进度条工具
import time
CONFIG = {
    "api": {
        "key": "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c",
        "base_url": "https://llm.jnu.cn/v1",
        "model": "Qwen2.5-72B-Instruct",
        "retries": 3,
        "timeout": 30
    }
}
# 配置OpenAI API（需要提前获取API密钥）

client = OpenAI(api_key=CONFIG["api"]["key"], 
                            base_url=CONFIG["api"]["base_url"])
def is_south_china_sea_related(text, max_retries=3):
    """
    调用大模型判断新闻是否与南海相关
    :param text: 新闻文本（标题+正文）
    :param max_retries: 最大重试次数
    :return: True/False
    """
    prompt = """请严格判断以下新闻是否与中国南海地区（South China Sea）相关。相关标准包括：
1. 直接提及南海地理名称（如南沙群岛、西沙群岛等）
2. 涉及南海地区的军事活动、资源开发、领土争端
3. 包含中国与东南亚国家在南海的互动
4. 涉及国际法（如UNCLOS）对南海的裁决

请仅用"Yes"或"No"回答，不要添加任何解释。

新闻内容：
{}

判断结果：""".format(text[:5000])  # 限制文本长度控制成本

    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=CONFIG["api"]["model"],  # 推荐使用gpt-4效果更好
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.0,  # 保持确定性输出
                max_tokens=2
            )
            return response.choices[0].message.content.strip().lower() == "yes"
            
        except Exception as e:
            print(f"API调用失败: {str(e)}，5秒后重试...")
            time.sleep(5)
    
    return False  # 重试失败后默认视为不相关

def filter_news(input_file, output_file):
    # 读取原始数据
    with open(input_file, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    # 过滤新闻
    filtered_news = []
    for news in tqdm(news_data, desc="筛选进度"):
        # 合并标题和正文作为判断依据
        combined_text = f"{news['title']}\n{news['text']}"
        
        if is_south_china_sea_related(combined_text):
            filtered_news.append(news)

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_news, f, indent=2, ensure_ascii=False)

    print(f"\n完成！原始新闻数：{len(news_data)}，筛选后保留：{len(filtered_news)}")

# 执行筛选
if __name__ == "__main__":
    filter_news(
        input_file="./waite_to_extrac/south_china.json",  # 输入文件路径（去重后的数据）
        output_file="south_china_sea_news.json"  # 输出文件路径
    )

