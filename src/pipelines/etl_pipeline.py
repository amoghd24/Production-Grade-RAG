from zenml import pipeline, step
from typing import List, Dict, Any, Optional
from src.data_pipeline.integrated_collector import IntegratedDataCollector
from src.feature_pipeline.document_processor import DocumentProcessor
from src.feature_pipeline.vector_storage import create_vector_store
from src.models.schemas import Document, ProcessingStatus
from src.config.settings import settings
import asyncio
from datetime import datetime
import json

# Step 1: Data Collection
@step
def collect_data_step(notion_api_key: Optional[str] = None, notion_database_id: Optional[str] = None, max_crawl_pages: int = 100) -> List[Document]:
    collector = IntegratedDataCollector()
    print("[ZenML] Collecting data from Notion and web (including embedded links)...")
    documents = asyncio.run(collector.collect_all_data(
        notion_api_key=notion_api_key,
        notion_database_id=notion_database_id,
        max_crawl_pages=max_crawl_pages,
        include_web_crawling=True
    ))
    print(f"[ZenML] Collected {len(documents)} documents.")
    return documents

# Step 2: Document Processing
@step
def process_documents_step(documents: List[Document], quality_threshold: float = 0.5) -> List[Dict[str, Any]]:
    processor = DocumentProcessor()
    processed_documents = []
    print("[ZenML] Processing documents...")
    for i, document in enumerate(documents):
        processed_data = asyncio.run(processor.process_document(document))
        quality_score = processed_data.get("quality_score", 0.0)
        if quality_score >= quality_threshold:
            processed_documents.append(processed_data)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(documents)} documents")
    print(f"[ZenML] Processed {len(processed_documents)} documents (passed quality filter)")
    return processed_documents

# Step 3: Storage
@step
def store_data_step(documents: List[Dict[str, Any]], store_to_mongodb: bool = True, save_local_backup: bool = True) -> None:
    print("[ZenML] Storing processed data...")
    # Save local backup
    if save_local_backup:
        try:
            data_dir = settings.DATA_DIR / "processed"
            data_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = data_dir / f"second_brain_backup_{timestamp}.json"
            backup_data = {
                "metadata": {
                    "backup_timestamp": datetime.utcnow().isoformat(),
                    "total_documents": len(documents),
                    "total_chunks": sum(len(doc.get("chunks", [])) for doc in documents),
                },
                "documents": []
            }
            for doc_data in documents:
                doc_dict = doc_data["document"].dict() if hasattr(doc_data["document"], 'dict') else doc_data["document"]
                chunks_dict = [chunk.dict() if hasattr(chunk, 'dict') else chunk for chunk in doc_data.get("chunks", [])]
                backup_data["documents"].append({
                    "document": doc_dict,
                    "chunks": chunks_dict,
                    "quality_score": doc_data.get("quality_score", 0.0)
                })
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"[ZenML] Local backup saved: {backup_file}")
        except Exception as e:
            print(f"[ZenML] Failed to save local backup: {str(e)}")
    # Store to MongoDB
    if store_to_mongodb:
        try:
            vector_store = asyncio.run(create_vector_store(
                initialize=True,
                setup_indexes=True
            ))
            all_documents = []
            all_chunks = []
            for doc_data in documents:
                document = doc_data["document"]
                chunks = doc_data.get("chunks", [])
                document.quality_score = doc_data.get("quality_score", 0.0)
                document.processing_status = ProcessingStatus.COMPLETED
                all_documents.append(document)
                all_chunks.extend(chunks)
            if all_documents and all_chunks:
                storage_result = asyncio.run(vector_store.store_documents_with_embeddings(
                    documents=all_documents,
                    chunks=all_chunks
                ))
                print(f"[ZenML] Stored {len(storage_result.get('document_ids', []))} documents to MongoDB")
            else:
                print("[ZenML] No documents to store in MongoDB")
            asyncio.run(vector_store.close())
        except Exception as e:
            print(f"[ZenML] Failed to store to MongoDB: {str(e)}")

# ZenML Pipeline
@pipeline
def etl_pipeline():
    docs = collect_data_step()
    processed = process_documents_step(docs)
    store_data_step(processed)

if __name__ == "__main__":
    etl_pipeline()