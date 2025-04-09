"""
处理neo4j产生的json文件，只留下title、what、community_id
"""
import json
import os

# 指定包含 JSON 文件的文件夹路径
input_folder = "./finall_result"  # 替换为你的文件夹路径
output_file = "finall_result.json"  # 输出文件名

# 初始化一个空列表，用于存储所有处理后的数据
filtered_data = []

# 初始化 ID 计数器
id_counter = 1

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_file = os.path.join(input_folder, filename)

        # 打开并加载原始 JSON 数据
        with open(input_file, "r", encoding="utf-8-sig") as infile:
            data = json.load(infile)  # 假设文件是一个 JSON 数组

        # 提取所需的字段，并为每个条目分配一个 ID
        for item in data:
            filtered_item = {
                # "id": id_counter,  # 分配一个递增的整数 ID
                "title": item.get("title", ""),
                "what": item.get("what", ""),
                "community_id": item['n']['properties'].get("community_id", "")
            }
            filtered_data.append(filtered_item)
            id_counter += 1  # 更新 ID 计数器

        print(f"Processed {filename}")

# 将所有处理后的数据写入一个 JSON 文件
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(filtered_data, outfile, indent=4, ensure_ascii=False)

print(f"All filtered data has been saved to {output_file}")