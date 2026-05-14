from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.embedder.embedder_base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """基于 SentenceTransformer 的文本向量化器。

    该类封装 SentenceTransformer 模型，用于将文本批量转换为向量表示，
    适用于语义检索、相似度计算等任务。

    当前默认模型为 all-MiniLM-L6-v2，是一个轻量级对称 embedding 模型，
    适用于中小规模 RAG 系统或本地部署场景。
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
    ) -> None:
        """初始化 SentenceTransformerEmbedder。

        Args:
            model_name (str): 使用的 SentenceTransformer 模型名称，
                例如 "all-MiniLM-L6-v2"。
            normalize_embeddings (bool): 是否对向量进行归一化，
                推荐开启（用于 cosine similarity）。

        Raises:
            RuntimeError: 当模型加载失败时抛出。
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        try:
            load_dotenv()
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(f"模型加载失败: {model_name}") from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表。

        该方法对输入文本进行批量 embedding，返回每个文本对应的向量表示。
        适用于文档向量化（如 Chunk embedding）。

        Args:
            texts (list[str]): 待向量化的文本列表。

        Returns:
            list[list[float]]: 每个文本对应的向量。

        Raises:
            TypeError:
                - texts 不是 list
                - texts 中包含非字符串元素
        """
        if not isinstance(texts, list):
            raise TypeError("texts 必须是 list[str] 类型")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts 中的元素必须全部是 str 类型")

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings.tolist()
