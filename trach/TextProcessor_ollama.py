"""
配置嵌入向量编码器
这里我使用ollama部署的bge-m3，对中文字符编码挺好 
"""

import logging
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np
import requests
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TextProcessor:
    """Ollama本地部署的BGE-M3嵌入生成器"""
    def __init__(self, 
                 ollama_uri,
                 model_name,
                 timeout: int = 60,
                 instruction_prefix: str = "为这个句子生成表示，以用于检索相关文章："
                ):
        self.model_name = model_name
        self.base_url = ollama_uri
        self.timeout = timeout
        self.instruction_prefix = instruction_prefix
        self.embedding_size = 1024  #向量维度

    def _add_instruction(self, text: str) -> str:
        """添加模型需要的指令前缀"""
        return f"{self.instruction_prefix}{text}"

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10),
           stop=stop_after_attempt(3))
    def _get_single_embedding(self, text: str) -> np.ndarray:
        """获取单个文本的嵌入向量"""
        try:
            processed_text = self._add_instruction(text)
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": processed_text,
                    "options": {
                        "temperature": 0.0  # 确保确定性输出
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # 解析响应数据
            embedding = response.json().get("embedding", [])
            if not embedding:
                raise ValueError("响应中未找到嵌入向量")
                
            return np.array(embedding)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"处理文本时出错: {str(e)}")
            raise

    def get_embeddings(self, texts: list) -> np.ndarray:
        """批量获取文本嵌入向量"""
        embeddings = []
        failed_count = 0
        
        for text in tqdm(texts, desc=f"生成嵌入({self.model_name})"):
            try:
                emb = self._get_single_embedding(text)
                if emb.shape != (self.embedding_size,):
                    raise ValueError(f"维度异常: {emb.shape}")
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"文本处理失败: {text[:30]}... 错误: {str(e)}")
                embeddings.append(np.zeros(self.embedding_size))
                failed_count += 1
                
        if failed_count > 0:
            logger.warning(f"总失败数: {failed_count}/{len(texts)}")
            
        return np.vstack(embeddings)
