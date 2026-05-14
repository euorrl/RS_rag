from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.embedder.embedder_base import BaseEmbedder


class BGEEmbedder(BaseEmbedder):
    """基于 BGE 的向量化器（支持中英混合语义检索）。

    该类封装 BGE 系列模型（如 bge-small-zh、bge-m3），用于将文本和查询
    转换为向量表示，适用于语义检索（RAG）场景。

    支持：
    - 文档向量（passage embedding）
    - 查询向量（query embedding）
    - 可选 instruction prefix（query: / passage:）
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        normalize_embeddings: bool = True,
        use_instruction: bool = False,
    ) -> None:
        """初始化 BGEEmbedder。

        Args:
            model_name (str): 使用的 BGE 模型名称，
                例如 "BAAI/bge-small-zh-v1.5"、"BAAI/bge-m3"。
            normalize_embeddings (bool): 是否对向量进行归一化，
                推荐开启（用于 cosine similarity）。
            use_instruction (bool): 是否使用 instruction prefix，
                即在文本前添加 "query:" 或 "passage:"。

        Raises:
            RuntimeError: 当模型加载失败时抛出。
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.use_instruction = use_instruction

        try:
            load_dotenv()
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(f"BGE 模型加载失败: {model_name}") from exc

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表（用于文档 embedding）。

        该方法用于对 Chunk 文本进行批量向量化，通常用于构建向量数据库。

        Args:
            texts (list[str]): 待向量化的文本列表。

        Returns:
            list[list[float]]: 每个文本对应的向量。

        Raises:
            TypeError:
                - texts 不是 list
                - texts 中包含非字符串元素
        """
        # =========================
        # 输入校验
        # =========================
        if not isinstance(texts, list):
            raise TypeError("texts 必须是 list[str] 类型")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts 中的元素必须全部是 str 类型")

        if not texts:
            return []

        # =========================
        # instruction prefix（可选）
        # =========================
        if self.use_instruction:
            texts = [f"passage: {text}" for text in texts]

        # =========================
        # 执行 embedding
        # =========================
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """将查询文本转换为向量（用于检索）。

        该方法用于将用户输入的 query 转换为向量，
        用于向量数据库的相似度搜索。

        Args:
            query (str): 用户输入的查询文本。

        Returns:
            list[float]: 查询向量。

        Raises:
            TypeError: query 不是字符串
        """
        # =========================
        # 输入校验
        # =========================
        if not isinstance(query, str):
            raise TypeError("query 必须是 str 类型")

        if not query.strip():
            raise ValueError("query 不能为空字符串")

        # =========================
        # instruction prefix（可选）
        # =========================
        if self.use_instruction:
            query = f"query: {query}"

        # =========================
        # 执行 embedding
        # =========================
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embedding.tolist()
