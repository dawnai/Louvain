"""
将大型json文件进行拆分
"""

import json

# 打开原始 JSON 文件
with open('./chunk_1w.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 设置每个小文件的大小（例如每 10000 条记录一个文件）
chunk_size = 1000

# 遍历数据并分割成多个文件
for i in range(0, len(data), chunk_size):
    chunk = data[i:i + chunk_size]
    output_file_name = f'chunk_{i // chunk_size + 1}.json'
    with open(output_file_name, 'w', encoding='utf-8') as output_file:
        json.dump(chunk, output_file, ensure_ascii=False, indent=4)
    print(f'Created file: {output_file_name}')