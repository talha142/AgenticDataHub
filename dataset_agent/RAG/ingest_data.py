import time
import uuid
import os
import warnings

# Suppress progress bars and warnings
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module='huggingface_hub')

from typing import List
from RAG.routing_agent import RoutingAgent, DataSource
from RAG.ingestion.parsers import ScientificDatabaseParser
from RAG.retrieval.vector_store import VectorStoreManager
from langchain_text_splitters import RecursiveCharacterTextSplitter

def run_ingestion_pipeline(urls: List[dict]):
    """
    Runs the automated data ingestion pipeline using real scientific databases.
    
    Args:
        urls: List of dicts with keys 'url', 'type' (pathway/compound/phytohub/card)
    """
    print("=== Starting Automated Data Ingestion Pipeline ===")
    
    # Initialize components
    router = RoutingAgent()
    parser = ScientificDatabaseParser()
    vector_store = VectorStoreManager(persist_directory="./data/in_memory_index")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    documents = []
    metadatas = []
    ids = []
    
    for entry in urls:
        url = entry["url"]
        data_type = entry["type"]
        print(f"\nProcessing: {url} (type: {data_type})")
        
        parsed_data = None
        
        try:
            if data_type == "phytohub":
                parsed_data = parser.parse_phytohub(url)
            elif data_type == "card":
                parsed_data = parser.parse_card(url)
            else:
                # Auto-detect using router
                routing_info = router.route_request(url)
                source = routing_info["source"]
                print(f"  Auto-detected source: {source}")
                
                if source == "PhytoHub":
                    parsed_data = parser.parse_phytohub(url)
                elif source == "CARD":
                    parsed_data = parser.parse_card(url)
                elif source == "KEGG":
                    parsed_data = parser.parse_kegg(url)
                elif source == "NPBS":
                    parsed_data = parser.parse_npbs(url)
                else:
                    print(f"  Skipping unknown source for URL: {url}")
                    continue
                    
        except Exception as e:
            print(f"  [Error] Failed to parse {url}: {e}")
            continue
            
        # Prepare the data for the Vector Store
        if parsed_data and parsed_data.get("text"):
            full_text = parsed_data["text"]
            chunks = text_splitter.split_text(full_text)
            
            for chunk in chunks:
                metadata = {
                    "source": parsed_data["source"],
                    "url": parsed_data["url"]
                }
                doc_id = str(uuid.uuid4())
                
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(doc_id)
                
            print(f"  Successfully extracted and chunked into {len(chunks)} chunk(s).")
        else:
            print(f"  Warning: No text content extracted from {url}")
            
        # Brief pause to respect server rate limits
        time.sleep(1)
    
    # Close the parser session
    parser.close()
        
    # Add all collected documents to the Vector Store
    if documents:
        print("\nSaving extracted documents to Vector Store...")
        vector_store.add_documents(documents, metadatas, ids)
        print(f"=== Ingestion Complete. Added {len(documents)} records. ===")
    else:
        print("\n=== Ingestion Complete. No valid documents were extracted. ===")

if __name__ == "__main__":
    # Real seed URLs from verified working databases
    seed_urls = [
        # PhytoHub (dietary phytochemicals & food sources)
        {"url": "https://phytohub.eu/search/compounds?query=epigallocatechin", "type": "phytohub"},
        {"url": "https://phytohub.eu/search/compounds?query=curcumin", "type": "phytohub"},
        {"url": "https://phytohub.eu/entries/PHUB000265", "type": "phytohub"},    # EGCG entry
        
        # CARD (antibiotic resistance ontology)
        {"url": "https://card.mcmaster.ca/ontology/36026", "type": "card"},       # OXA beta-lactamase
        {"url": "https://card.mcmaster.ca/ontology/36263", "type": "card"},       # NDM-1 (if exists)

        # NPBS (Natural Products)
        {"url": "https://biochemai.cstspace.cn/npbs/naturalproducts/", "type": "npbs"},
    ]
    
    run_ingestion_pipeline(seed_urls)
