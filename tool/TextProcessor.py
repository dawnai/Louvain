"""
配置嵌入向量编码器
切换为支持OpenAI接口的远程BGE-M3服务
服务地址：https://embedding.jnu.cn/v1
"""

import logging
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """支持OpenAI接口的远程BGE-M3嵌入生成器"""
    def __init__(self, 
                 embedding_uri: str = "https://embedding.jnu.cn/v1",
                 embedding_name: str = "bge-m3",
                 timeout: int = 60,
                 embedding_size: int = 1024,
                 max_workers: int = 10):
        self.model_name = embedding_name
        self.api_base = embedding_uri
        self.timeout = timeout
        self.embedding_size = embedding_size
        self.max_workers = max_workers  # 最大并行进程数

    @staticmethod
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10),
           stop=stop_after_attempt(3))
    def _get_single_embedding(text: str, model_name: str, api_base: str) -> np.ndarray:
        """获取单个文本的嵌入向量（静态方法以便序列化）"""
        client = OpenAI(api_key="API_KEY", base_url=api_base)
        try:
            response = client.embeddings.create(
                input=text,
                model=model_name
            )
            embedding = response.data[0].embedding
            return np.array(embedding)
        except Exception as e:
            logger.error(f"请求失败: {str(e)}")
            raise

    def get_embeddings(self, texts: list) -> np.ndarray:
        """批量获取文本嵌入向量"""
        embeddings = []
        failed_count = 0
        
        # 创建进程池
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 创建任务字典 {future: text}
            futures = {
                executor.submit(
                    self._get_single_embedding,
                    text,
                    self.model_name,
                    self.api_base
                ): text
                for text in texts
            }

            # 使用tqdm显示进度条
            with tqdm(total=len(texts), desc=f"生成嵌入({self.model_name})") as pbar:
                for future in as_completed(futures):
                    text = futures[future]
                    try:
                        emb = future.result()
                        if emb.shape != (self.embedding_size,):
                            raise ValueError(f"维度异常: {emb.shape}")
                        embeddings.append(emb)
                    except Exception as e:
                        logger.warning(f"文本处理失败: {text[:30]}... 错误: {str(e)}")
                        embeddings.append(np.zeros(self.embedding_size))
                        failed_count += 1
                    finally:
                        pbar.update(1)

        if failed_count > 0:
            logger.warning(f"总失败数: {failed_count}/{len(texts)}")
            
        return np.vstack(embeddings)

# 使用示例
if __name__ == "__main__":
    processor = TextProcessor(max_workers=4)  # 根据CPU核心数调整
    texts = ["自然语言处理", "深度学习模型"] * 10  # 测试数据
    embeddings = processor.get_embeddings(texts)
    print(f"生成的嵌入矩阵形状：{embeddings.shape}")
