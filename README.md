<img src="https://avatars.githubusercontent.com/u/61813006?v=4" style="zoom:50%;" />

# 新闻事件聚类  :newspaper: :newspaper_roll: :wave:

项目逻辑：

**原始新闻文档 ---> 抽取5w1h、organization、category ---> 存入neo4j数据库 --->louvain算法---> 聚类事件属性添加--->当日事件入库**

### :hand: 使用：

:one: 拉取项目：

```:two:
git clone https://github.com/dawnai/Louvain.git
```

:two: 使用conda创建虚拟环境，推荐python 10环境：

```
conda create -n py10 python=10
```

:three:准备代抽取的新闻，放置在data/waite_to_extrac/文件夹路径下

```
|-data
|---waite_to_extrac
|------news1.json
|------news2.json
```

:four:运行：

```
python main.py
```

运行结束后，可以在neo4j中查看抽取效果。



### :warning:注意事项： 

1、该项目必须使用json格式的新闻抽取文档，格式如下：

```json
[
  {
    "id": 1,
    "title": "",
    "created_at": "",
    "text": ""
  },
  {
    "id": 2,
    "title": "",
    "created_at": "",
    "text": ""
  }
]
```

2、main.py中配置文件说明：

```python
CONFIG = {
    "input_json": "./data/waite_to_extrac/2-3.json",#当日新闻json文件地址
    "output_files": {
        "excel": "./data/waite_to_neo4j/xlsx/2-3.xlsx",#抽取结果存放地址
        "json": "./data/waite_to_neo4j/json/2-3.json"
    },
    "api": {	#LLM配置
        "key": "sk-d0c3b3fe823c4fcfbe6a56a8a13c946c",
        "base_url": "https://llm.jnu.cn/v1",
        "model": "Qwen2.5-72B-Instruct",
        "retries": 3,	#失败重试次数
        "timeout": 30	#超时
    },
    "processing": {		#多5w1h抽取策略
        "batch_size": 5,
        "max_events_per_news": 3,	#每条新闻最多抽取多少5w1h
        "request_interval": 1,	#抽取间隔
        "text_truncate_length": 1000	#文本截断
    },
    "neo4j_config":{ 	#neo4j数据库配置
        "uri": "bolt://172.20.77.180:7687",
        "user": "neo4j",
        "password": "neo4j@openspg",
        "database": "today", 	#当日数据库
        "alldatabase":"allday"	#总数据库
    },
    "target_columns":['what', 'where', 'when', 'who', 'why', 'how', 'title','organization','news_id'],#用于判断xlsx表格中哪些列需要被抽取
    "config_louvain":{
        "semantic_threshold": 0.8,  #语义相似度阈值
        "embedding_uri":'https://embedding.jnu.cn/v1',#ollama地址http://172.20.71.112:11434 暨大：https://embedding.jnu.cn/v1
        "embedding_name":'bge-m3'#模型
    }
}
```

3、项目采用多线程进行实体抽取和数据处理工作



