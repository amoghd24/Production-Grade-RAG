from zenml import pipeline, step
from typing import List, Dict, Any, Optional, Set, Tuple
from src.data_pipeline.integrated_collector import NotionDataCollector, WebDataCollector, DocumentCombiner
from src.feature_pipeline.document_processor import DocumentProcessor
from src.feature_pipeline.vector_storage import create_vector_store
from src.models.schemas import Document, ProcessingStatus
from src.config.settings import settings
import asyncio
from datetime import datetime
import json

# Step 1: Collect Notion Documents
@step
def collect_notion_documents(
    notion_api_key: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    search_query: str = ""
) -> Tuple[List[Document], Set[str]]:
    """Step to collect documents from Notion"""
    collector = NotionDataCollector()
    notion_docs, embedded_urls = asyncio.run(collector.collect(
        api_key=notion_api_key,
        database_id=notion_database_id,
        search_query=search_query
    ))
    return notion_docs, set(embedded_urls)

# Step 2: Crawl Embedded Links
@step
def crawl_embedded_links(
    embedded_urls: Set[str],
    max_pages: int = 1000
) -> List[Document]:
    """Step to crawl embedded links"""
    if not embedded_urls:
        return []
        
    collector = WebDataCollector()
    crawled_docs = asyncio.run(collector.collect(
        urls=embedded_urls,
        max_pages=max_pages
    ))
    return crawled_docs

# Step 3: Combine Documents
@step
def combine_documents(
    notion_documents: List[Document],
    crawled_documents: List[Document]
) -> Tuple[List[Document], Dict[str, any]]:
    """Step to combine documents and generate statistics"""
    combiner = DocumentCombiner()
    all_docs, stats = combiner.combine(notion_documents, crawled_documents)
    return all_docs, stats

# Step 4: Process Documents
@step
def process_documents(
    documents: List[Document],
    quality_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Step to process documents"""
    processor = DocumentProcessor()
    processed_documents = []
    for document in documents:
        processed_data = asyncio.run(processor.process_document(document))
        quality_score = processed_data.get("quality_score", 0.0)
        if quality_score >= quality_threshold:
            processed_documents.append(processed_data)
    return processed_documents

# Step 5: Store Data
@step
def store_data(
    documents: List[Dict[str, Any]],
    store_to_mongodb: bool = True,
    save_local_backup: bool = True
) -> None:
    """Step to store processed data"""
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
        except Exception:
            pass
    
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
                asyncio.run(vector_store.store_documents_with_embeddings(
                    documents=all_documents,
                    chunks=all_chunks
                ))
            asyncio.run(vector_store.close())
        except Exception:
            pass

# Main Pipeline
@pipeline
def etl_pipeline(
    notion_api_key: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    max_crawl_pages: int = 1000,
    quality_threshold: float = 0.5,
    store_to_mongodb: bool = True,
    save_local_backup: bool = True
):
    # Step 1: Collect Notion documents
    notion_docs, embedded_urls = collect_notion_documents(
        notion_api_key=notion_api_key,
        notion_database_id=notion_database_id
    )
    
    # Step 2: Crawl embedded links
    crawled_docs = crawl_embedded_links(
        embedded_urls=embedded_urls,
        max_pages=max_crawl_pages
    )
    
    # Step 3: Combine documents
    all_docs, stats = combine_documents(notion_docs, crawled_docs)
    
    # Step 4: Process documents
    processed = process_documents(
        documents=all_docs,
        quality_threshold=quality_threshold
    )
    
    # Step 5: Store data
    store_data(
        documents=processed,
        store_to_mongodb=store_to_mongodb,
        save_local_backup=save_local_backup
    )

if __name__ == "__main__":
    etl_pipeline()