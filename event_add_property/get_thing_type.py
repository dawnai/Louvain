import json

def get_thing_type(filename):
    """
    打开并读取 JSON 文件,这个json文件存储着事件的属性
    """
    # 打开并读取 JSON 文件
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
