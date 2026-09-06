from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List

# ---------------------------------------------------------------------------
# Shared model instance — imported by auditor.py, not reloaded
# ---------------------------------------------------------------------------
shared_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Thin LangChain-compatible wrapper around the shared ST model
# LangChain needs an object with .embed_documents() and .embed_query()
# ---------------------------------------------------------------------------
class SharedSTEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return shared_embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return shared_embedding_model.encode(text, convert_to_numpy=True).tolist()

embeddings = SharedSTEmbeddings()

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
vectorstore       = None
bm25_retriever    = None
ensemble_retriever = None

# ---------------------------------------------------------------------------
# Truth-doc embedding cache
# ---------------------------------------------------------------------------
# The knowledge base is fixed and small, so the same doc gets embedded over
# and over across requests for no reason. Cache by text so auditor.py can
# skip re-encoding a truth doc it has already seen. Real speed win (one
# fewer model call per request in the common case) with zero accuracy cost,
# since the embedding of a fixed string never changes.
# ---------------------------------------------------------------------------
_DOC_EMBEDDING_CACHE = {}

def get_cached_doc_embedding(text: str):
    """Return a cached sentence embedding for a KB doc, computing it once."""
    if text not in _DOC_EMBEDDING_CACHE:
        _DOC_EMBEDDING_CACHE[text] = shared_embedding_model.encode(
            text, convert_to_tensor=True
        )
    return _DOC_EMBEDDING_CACHE[text]

def initialize_retriever(docs):
    global vectorstore, bm25_retriever, ensemble_retriever

    client = QdrantClient(url="http://localhost:6333")

    # Only recreate collection if it doesn't already exist
    existing = [c.name for c in client.get_collections().collections]
    if "veracity_vault" in existing:
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="veracity_vault",
            embedding=embeddings,
        )
    else:
        vectorstore = QdrantVectorStore.from_documents(
            docs,
            embeddings,
            url="http://localhost:6333",
            collection_name="veracity_vault",
        )

    bm25_retriever   = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore.as_retriever()],
        weights=[0.3, 0.7]
    )
    return ensemble_retriever

def get_vectorstore():
    return vectorstore