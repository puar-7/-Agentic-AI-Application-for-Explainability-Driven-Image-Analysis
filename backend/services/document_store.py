from typing import List
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi
import pickle

class DocumentStore:
    """
    Handles document ingestion, chunking, embedding, and hybrid retrieval.

    Hybrid retrieval = Dense (FAISS) + Sparse (BM25)
    This implementation is intentionally explicit and deterministic.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_k: int = 6,
        dense_k: int = 6,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_k = bm25_k
        self.dense_k = dense_k

        # Dense embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name
        )

        # Stores
        self.vector_store = None

        # BM25 components (explicit, not via LangChain)
        self.bm25 = None
        self.bm25_documents = []


    def add_documents(self, new_documents):
        """
        Append new documents to existing FAISS + BM25 indexes.
        """
        if self.vector_store is None or self.bm25 is None:
            raise RuntimeError("Indexes must be loaded before appending.")

        # Chunk only NEW documents
        new_chunks = self.chunk_documents(new_documents)

        if not new_chunks:
            return

        # ---- Dense: FAISS append ----
        self.vector_store.add_documents(new_chunks)

        # ---- Sparse: extend corpus + rebuild BM25 ----
        self.bm25_documents.extend(new_chunks)
        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in self.bm25_documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
    # ------------------------------------------------------------------
    # Document Loading
    # ------------------------------------------------------------------

    def load_documents(self, file_paths: List[str]):
        """
        Load documents from disk.
        Supports .txt and .pdf files.
        """
        documents = []

        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {path}")

            documents.extend(loader.load())

        return documents

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def chunk_documents(self, documents):
        """
        Split documents into large, overlapping chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)

    # ------------------------------------------------------------------
    # Index Construction
    # ------------------------------------------------------------------

    def build_indexes(self, documents):
        """
        Build:
        1. FAISS vector index for dense retrieval
        2. BM25 index for keyword retrieval
        """
        chunks = self.chunk_documents(documents)

        if not chunks:
            raise ValueError("No document chunks created.")

        # ---------- Dense (FAISS) ----------
        self.vector_store = FAISS.from_documents(
            chunks,
            embedding=self.embeddings
        )

        # ---------- Sparse (BM25) ----------
        self.bm25_documents = chunks
        tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)


    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "vector_store": self.vector_store,
                "bm25": self.bm25,
                "bm25_documents": self.bm25_documents
            }, f)
    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        store = cls()
        store.vector_store = data["vector_store"]
        store.bm25 = data["bm25"]
        store.bm25_documents = data["bm25_documents"]
        return store        
    # ------------------------------------------------------------------
    # Hybrid Retrieval
    # ------------------------------------------------------------------

    def hybrid_retrieve(self, query: str):
        """
        Perform hybrid retrieval:
        - Dense similarity search (FAISS)
        - Sparse keyword search (BM25)
        - Merge results deterministically
        """
        if self.vector_store is None or self.bm25 is None:
            raise RuntimeError("Indexes not built. Call build_indexes() first.")

        # ---------- Dense Retrieval ----------
        dense_docs = self.vector_store.similarity_search(
            query,
            k=self.dense_k
        )

        # ---------- Sparse Retrieval ----------
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_bm25_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.bm25_k]

        sparse_docs = [self.bm25_documents[i] for i in top_bm25_indices]

        # ---------- Merge (deduplicate by content) ----------
        combined = {
            doc.page_content: doc
            for doc in dense_docs + sparse_docs
        }

        return list(combined.values())
    def reset(self):
        """
        Clears the in-memory state of the store.
        """
        self.vector_store = None
        self.bm25 = None
        self.bm25_documents = []
        print("[DocumentStore] In-memory index has been reset.")