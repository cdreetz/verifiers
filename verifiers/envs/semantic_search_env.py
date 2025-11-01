import os
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

import verifiers as vf


class SemanticSearchEnv(vf.ToolEnv):
    def __init__(
        self,
        collection_name: str = "documents",
        embed_model: str = "text-embedding-3-small",
        embed_base_url: str = "https://api.openai.com/v1",
        embed_api_key_var: str = "OPENAI_API_KEY",
        chroma_db_dir: str = ".chroma_db",
        chroma_server_port: int = 8000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.collection_name = collection_name
        self.chroma_db_dir = chroma_db_dir
        self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
            model_name=embed_model,
            api_base=embed_base_url,
            api_key=os.getenv(embed_api_key_var, "EMPTY")
        )
        self.setup_vector_db()
        self.prep_corpus()
        self.async_client = None

        self.chroma_server_port = chroma_server_port
        self.check_server_running()

        self.add_tool(self.search_documents)

    @abstractmethod
    def prep_corpus(self) -> None:
        """Prepare and load the corpus into the vector database.
        
        This method should:
        1. Load data from the source
        2. Upsert documents into the ChromaDB collection
        """
        raise NotImplementedError("Subclasses must implement prep_corpus")

    def setup_vector_db(self):
        """Persistent client for creating and adding documents during init"""
        self.setup_client = chromadb.PersistentClient(path=self.chroma_db_dir)
        self.collection = self.setup_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.emb_fn
        )
    
    async def get_async_client(self):
        """Async cient for querying via client-server"""
        if self.async_client is None:
            self.async_client = await chromadb.AsyncHttpClient(
                host="localhost",
                port=self.chroma_server_port
            )
        return self.async_client

    def check_server_running(self):
        try:
            client = chromadb.HttpClient(host="localhost", port=self.chroma_server_port)
            client.heartbeat()
        except Exception:
            raise RuntimeError(
                f"ChromaDB server is not running at localhost:{self.chroma_server_port}. "
                f"Please start the server: chroma run --path {self.chroma_db_dir}"
            ) from None

   # doing upsert as thats what was used in original wiki search env 
   # but pretty sure this could just be .add since we are only upserting new documents
   # but upsert is specifically for updating existing documents
    
    def upsert_documents(
        self,
        document_ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 500
    ) -> None:
        all_ids = document_ids
        existing: set[str] = set()
        
        for i in range(0, len(all_ids), batch_size):
            batch = all_ids[i : i + batch_size]
            got = self.collection.get(ids=batch)
            existing.update(got.get("ids", []))
        
        # Filter to only new documents
        to_upsert_indices = [i for i, doc_id in enumerate(document_ids) if doc_id not in existing]
        
        if to_upsert_indices:
            to_upsert_ids = [document_ids[i] for i in to_upsert_indices]
            to_upsert_docs = [documents[i] for i in to_upsert_indices]
            to_upsert_meta = [metadatas[i] for i in to_upsert_indices] if metadatas else None
            
            # Upsert in batches
            for i in range(0, len(to_upsert_ids), batch_size):
                end_idx = min(i + batch_size, len(to_upsert_ids))
                print(f"Upserting {end_idx - i} documents...")
                
                upsert_args = {
                    "ids": to_upsert_ids[i:end_idx],
                    "documents": to_upsert_docs[i:end_idx],
                }
                
                if to_upsert_meta:
                    upsert_args["metadatas"] = to_upsert_meta[i:end_idx]
                
                self.collection.upsert(**upsert_args)
    
    ##########################
    ## SEMANTIC SEARCH TOOL ##
    ##########################
    
    async def search_documents(
        self, 
        query: str,
        return_contents: bool = True,
        return_metadata: bool = True,
    ) -> list[dict]:
        """Search for relevant documents using embedding similarity.
        
        Args:
            query (str): The query to search for.
            
        Returns:
            list[dict]: A list of dicts with document_id, title, and optionally contents and/or metadata.
        """
        include = []
        if return_contents:
            include.append("documents")
        if return_metadata:
            include.append("metadatas")
        if not include:
            include = None
        
        async_client = await self.get_async_client()
        collection = await async_client.get_collection(
            name=self.collection_name,
            embedding_function=self.emb_fn
        )
        results = await collection.query(
            query_texts=[query],
            include=include,
            n_results=10
        )
        if not results:
            raise ValueError(f"No results found for query: {query}")
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "document_id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i] if return_metadata else None,
                "content": results["documents"][0][i] if return_contents else None,
            })
        return output