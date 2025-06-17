from zenml import pipeline, step
from typing import List, Dict, Any, Optional, Set, Tuple
from src.data_pipeline.integrated_collector import NotionDataCollector, WebDataCollector, DocumentCombiner
from src.feature_pipeline.document_processor import ContentCleaner, QualityScorer, DocumentChunker, EmbeddingGenerator, MarkdownConverter
from src.feature_pipeline.vector_storage import create_vector_store
from src.models.schemas import Document, DocumentChunk, ProcessingStatus, ContentSource
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
) -> List[Document]:
    """Step to combine documents."""
    combiner = DocumentCombiner()
    all_docs, _ = combiner.combine(notion_documents, crawled_documents)
    return all_docs

# Step 4a: Convert to Markdown
@step
def convert_to_markdown(documents: List[Document]) -> List[Document]:
    """Step to convert HTML content to Markdown format."""
    converter = MarkdownConverter()
    for doc in documents:
        if doc.source == ContentSource.WEB_CRAWL and doc.metadata.get("content_format") != "markdown":
            doc.content = converter.convert(doc.content)
            doc.metadata["content_format"] = "markdown"
    return documents

# Step 4b: Clean Content
@step
def clean_content(documents: List[Document]) -> List[Document]:
    cleaner = ContentCleaner()
    for doc in documents:
        doc.content = cleaner.clean(doc.content)
    return documents

# Step 4c: Compute Quality Score
@step
def compute_quality_score(documents: List[Document], quality_threshold: float = 0.5) -> List[Document]:
    scorer = QualityScorer()
    filtered_docs = []
    for doc in documents:
        doc.quality_score = scorer.score(doc)
        if doc.quality_score >= quality_threshold:
            filtered_docs.append(doc)
        else:
            doc.processing_status = ProcessingStatus.COMPLETED
    return filtered_docs

# Step 4d: Chunk Documents
@step
def chunk_document(documents: List[Document]) -> List[Dict[str, Any]]:
    chunker = DocumentChunker()
    chunked = []
    for doc in documents:
        chunks = chunker.chunk(doc)
        if not chunks:
            doc.processing_status = ProcessingStatus.COMPLETED
            continue
        chunked.append({
            "document": doc,
            "chunks": chunks,
            "quality_score": doc.quality_score
        })
    return chunked

# Step 4e: Generate Embeddings
@step
def generate_embeddings(chunked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    embedder = EmbeddingGenerator()
    processed_documents = []
    for doc_data in chunked_docs:
        document = doc_data["document"]
        chunks = doc_data["chunks"]
        if not chunks:
            document.processing_status = ProcessingStatus.COMPLETED
            continue
        chunks_with_embeddings = asyncio.run(embedder.generate(chunks))
        document.processing_status = ProcessingStatus.COMPLETED
        document.updated_at = datetime.utcnow()
        processed_data = {
            "document": document,
            "chunks": chunks_with_embeddings,
            "quality_score": doc_data["quality_score"],
            "processed": True,
            "chunk_count": len(chunks_with_embeddings)
        }
        processed_documents.append(processed_data)
    return processed_documents

# Step 5a: Store to MongoDB
@step
def store_to_mongodb(
    documents: List[Dict[str, Any]],
    should_store_to_mongodb: bool = True
) -> None:
    """Step to store processed data to MongoDB"""
    if should_store_to_mongodb:
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

# Step 5b: Save Local Backup
@step
def save_local_backup(
    documents: List[Dict[str, Any]],
    should_save_local_backup: bool = True
) -> None:
    """Step to save processed data as local backup"""
    if should_save_local_backup:
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

# Main Pipeline
@pipeline
def etl_pipeline(
    notion_api_key: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    max_crawl_pages: int = 1000,
    quality_threshold: float = 0.5,
    should_store_to_mongodb: bool = True,
    should_save_local_backup: bool = True
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
    combined_docs = combine_documents(
        notion_documents=notion_docs,
        crawled_documents=crawled_docs
    )
    
    # Step 4a: Convert to Markdown
    markdown_docs = convert_to_markdown(documents=combined_docs)
    
    # Step 4b: Clean content
    cleaned_docs = clean_content(documents=markdown_docs)
    
    # Step 4c: Compute quality score
    scored_docs = compute_quality_score(
        documents=cleaned_docs,
        quality_threshold=quality_threshold
    )
    
    # Step 4d: Chunk documents
    chunked_docs = chunk_document(documents=scored_docs)
    
    # Step 4e: Generate embeddings
    processed_docs = generate_embeddings(chunked_docs=chunked_docs)
    
    # Step 5a: Store to MongoDB
    store_to_mongodb(
        documents=processed_docs,
        should_store_to_mongodb=should_store_to_mongodb
    )
    
    # Step 5b: Save local backup
    save_local_backup(
        documents=processed_docs,
        should_save_local_backup=should_save_local_backup
    )

if __name__ == "__main__":
    etl_pipeline()