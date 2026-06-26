from typing import List, Dict, Any
import os
import json
import logging
import warnings
# Suppress HuggingFace warnings and progress bars
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Prevent huggingface_hub from warning about unauthenticated requests
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStoreManager:
    """
    An InMemory vector store to store and retrieve document chunks 
    from NPBS, NPedia, PhytoHub, and CARD.
    """
    
    def __init__(self, persist_directory: str = "./data/in_memory_index"):
        self.persist_directory = persist_directory
        print("Initializing Embeddings Model (this may take a moment to download if first time)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./models"
        )
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.index_file = os.path.join(self.persist_directory, "index.json")
        
        # Keep track of existing content to avoid duplicates
        self.seen_content = set()
        self.all_documents_metadata = []
        
        # Load existing index if it exists
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    docs = []
                    for d in data:
                        content = d.get("page_content", "").strip()
                        if content:
                            # Use a hash or just the string if it's not too large
                            if content not in self.seen_content:
                                docs.append(Document(page_content=content, metadata=d.get("metadata", {})))
                                self.all_documents_metadata.append({"page_content": content, "metadata": d.get("metadata", {})})
                                self.seen_content.add(content)
                    
                    if docs:
                        self.vector_store.add_documents(docs)
                        print(f"Loaded {len(docs)} unique documents from {self.index_file}")
            except Exception as e:
                print(f"Error loading in-memory index: {e}")
        else:
            print("Initialized empty InMemory VectorStoreManager.")

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Adds parsed documents to the in-memory store and saves to disk.
        """
        if not documents:
            return
            
        new_docs = []
        for doc, meta in zip(documents, metadatas):
            clean_doc = str(doc).strip()
            if clean_doc and clean_doc not in self.seen_content:
                new_docs.append(Document(page_content=clean_doc, metadata=meta))
                self.all_documents_metadata.append({"page_content": clean_doc, "metadata": meta})
                self.seen_content.add(clean_doc)
            
        if new_docs:
            self.vector_store.add_documents(new_docs)
            print(f"Added {len(new_docs)} new unique documents to InMemory store.")
        else:
            print("No new unique documents to add.")
            return

        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Save ALL unique documents to disk
        if self.all_documents_metadata:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.all_documents_metadata, f, indent=4)
            print(f"Saved total {len(self.all_documents_metadata)} documents to disk.")
        else:
            print("Warning: No documents found to save.")

    def retrieve(self, query: str, top_k: int = 5, source_filter: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves document chunks using dense vector similarity.
        """
        # Check if store is empty
        has_data = False
        for attr in ["store", "_dict", "data"]:
            if hasattr(self.vector_store, attr) and getattr(self.vector_store, attr):
                has_data = True
                break
                
        if not has_data:
            print("Vector store is empty.")
            return []
            
        def filter_func(doc):
            if source_filter and doc.metadata.get("source") != source_filter:
                return False
            # Filter out chunks that are too short or just N/A
            if len(doc.page_content) < 10 or ("N/A" in doc.page_content and len(doc.page_content) < 50):
                return False
            return True
            
        # similarity search
        try:
            docs = self.vector_store.similarity_search(query, k=top_k, filter=filter_func)
        except Exception as e:
            print(f"Retrieval error: {e}")
            docs = []
        
        retrieved_chunks = []
        for doc in docs:
            retrieved_chunks.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
                
        return retrieved_chunks

if __name__ == "__main__":
    # Simple test
    store = VectorStoreManager(persist_directory="./test_inmemory_db")
    
    # Dummy data
    store.add_documents(
        documents=["Curcumin is found in turmeric and has anti-inflammatory properties.", "Cardiovascular diseases are a class of diseases."],
        metadatas=[{"source": "NPedia", "url": "http://npedia/curcumin"}, {"source": "Wiki", "url": "http://wiki"}],
        ids=["doc_1", "doc_2"]
    )
    
    res = store.retrieve("What has anti-inflammatory properties?", top_k=1)
    print("Retrieval Result:", res)
