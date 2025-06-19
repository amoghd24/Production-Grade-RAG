from zenml import pipeline, step
from typing import List, Dict, Any, Optional, Annotated
from src.feature_pipeline.document_processor import ContentCleaner, DocumentChunker, EmbeddingGenerator
from src.feature_pipeline.vector_storage import create_vector_store
from src.models.schemas import Document, DocumentChunk, ProcessingStatus
from src.config.settings import settings
import asyncio
from datetime import datetime

# Step 1: Fetch Documents from MongoDB
@step
def fetch_from_mongodb(
    collection_name: str = "documents",
    limit: int = 1000
) -> Annotated[List[Document], "documents"]:
    """Step to fetch documents from MongoDB"""
    try:
        # Create a new event loop for this step
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        vector_store = loop.run_until_complete(create_vector_store(
            initialize=True,
            setup_indexes=False
        ))
        
        # Fetch documents using document repository
        documents = loop.run_until_complete(
            vector_store.document_repo.get_documents_by_filter(
                filters={}, 
                limit=limit
            )
        )
        
        loop.run_until_complete(vector_store.close())
        loop.close()
        
        print(f"Fetched {len(documents)} documents from MongoDB")
        return documents
        
    except Exception as e:
        print(f"Error fetching documents from MongoDB: {str(e)}")
        return []

# Step 2: Filter by Quality Score
@step
def filter_by_quality(
    documents: List[Document],
    content_quality_score_threshold: float = 0.5
) -> Annotated[List[Document], "filtered_documents"]:
    """Step to filter documents by quality score"""
    
    assert 0 <= content_quality_score_threshold <= 1, (
        "Content quality score threshold must be between 0 and 1"
    )
    
    filtered_docs = []
    for doc in documents:
        # Check if document has quality score and meets threshold
        if hasattr(doc, 'quality_score') and doc.quality_score is not None:
            if doc.quality_score >= content_quality_score_threshold:
                filtered_docs.append(doc)
        else:
            # If no quality score, include the document (backward compatibility)
            filtered_docs.append(doc)
    
    print(f"Filtered {len(documents)} -> {len(filtered_docs)} documents (threshold: {content_quality_score_threshold})")
    return filtered_docs

# Step 3: Clean Content
@step
def clean_content(documents: List[Document]) -> List[Document]:
    """Step to clean document content"""
    cleaner = ContentCleaner()
    for doc in documents:
        doc.content = cleaner.clean(doc.content)
    print(f"Cleaned content for {len(documents)} documents")
    return documents

# Step 4: Chunk Documents
@step
def chunk_documents(
    documents: List[Document]
) -> Annotated[List[Dict[str, Any]], "chunked_documents"]:
    """Step to chunk documents into smaller pieces"""
    chunker = DocumentChunker()  # No parameters needed - uses settings.CHUNK_SIZE
    chunked_data = []
    
    for doc in documents:
        chunks = chunker.chunk(doc)
        if chunks:
            chunked_data.append({
                "document": doc,
                "chunks": chunks,
                "quality_score": getattr(doc, 'quality_score', 0.0)
            })
        else:
            doc.processing_status = ProcessingStatus.COMPLETED
    
    total_chunks = sum(len(item["chunks"]) for item in chunked_data)
    print(f"Created {total_chunks} chunks from {len(documents)} documents")
    return chunked_data

# Step 5: Generate Embeddings
@step
def generate_embeddings(
    chunked_docs: List[Dict[str, Any]]
) -> Annotated[List[Dict[str, Any]], "embedded_documents"]:
    """Step to generate embeddings for document chunks"""
    embedder = EmbeddingGenerator()  # No parameters needed - uses settings
    processed_documents = []
    
    for doc_data in chunked_docs:
        document = doc_data["document"]
        chunks = doc_data["chunks"]
        
        if not chunks:
            document.processing_status = ProcessingStatus.COMPLETED
            continue
            
        # Generate embeddings for chunks
        chunks_with_embeddings = asyncio.run(embedder.generate(chunks))
        
        # Update document status
        document.processing_status = ProcessingStatus.COMPLETED
        document.updated_at = datetime.now()
        
        processed_data = {
            "document": document,
            "chunks": chunks_with_embeddings,
            "quality_score": doc_data["quality_score"],
            "processed": True,
            "chunk_count": len(chunks_with_embeddings)
        }
        processed_documents.append(processed_data)
    
    total_embeddings = sum(len(item["chunks"]) for item in processed_documents)
    print(f"Generated embeddings for {total_embeddings} chunks")
    return processed_documents

# Step 6: Store Chunks to MongoDB
@step
def store_chunks_to_mongodb(
    processed_docs: List[Dict[str, Any]],
    collection_name: str = "chunks"
) -> None:
    """Step to store processed chunks with embeddings to MongoDB"""
    try:
        # Create a new event loop for this step
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        vector_store = loop.run_until_complete(create_vector_store(
            initialize=True,
            setup_indexes=True
        ))
        
        # Prepare documents and chunks for storage
        all_documents = []
        all_chunks = []
        
        for doc_data in processed_docs:
            document = doc_data["document"]
            chunks = doc_data.get("chunks", [])
            
            # Ensure document has quality score
            document.quality_score = doc_data.get("quality_score", 0.0)
            document.processing_status = ProcessingStatus.COMPLETED
            
            all_documents.append(document)
            all_chunks.extend(chunks)
        
        if all_documents and all_chunks:
            # Store documents and chunks with embeddings
            result = loop.run_until_complete(
                vector_store.store_documents_with_embeddings(
                    documents=all_documents,
                    chunks=all_chunks
                )
            )
            print(f"Successfully stored {result['documents_stored']} documents and {result['chunks_stored']} chunks to MongoDB")
        
        loop.run_until_complete(vector_store.close())
        loop.close()
        
    except Exception as e:
        print(f"Error storing chunks to MongoDB: {str(e)}")

# Main RAG Feature Pipeline
@pipeline
def rag_pipeline(
    extract_collection_name: str = "documents",
    fetch_limit: int = 1000,
    load_collection_name: str = "chunks",
    content_quality_score_threshold: float = 0.5
):
    """
    RAG Feature Pipeline that processes documents for vector search.
    
    Args:
        extract_collection_name: MongoDB collection to fetch documents from
        fetch_limit: Maximum number of documents to process
        load_collection_name: MongoDB collection to store chunks to
        content_quality_score_threshold: Minimum quality score for documents
    """
    
    # Step 1: Fetch documents from MongoDB
    documents = fetch_from_mongodb(
        collection_name=extract_collection_name,
        limit=fetch_limit
    )
    
    # Step 2: Filter by quality score
    filtered_docs = filter_by_quality(
        documents=documents,
        content_quality_score_threshold=content_quality_score_threshold
    )
    
    # Step 3: Clean content
    cleaned_docs = clean_content(documents=filtered_docs)
    
    # Step 4: Chunk documents
    chunked_docs = chunk_documents(
        documents=cleaned_docs
    )
    
    # Step 5: Generate embeddings
    embedded_docs = generate_embeddings(
        chunked_docs=chunked_docs
    )
    
    # Step 6: Store chunks to MongoDB
    store_chunks_to_mongodb(
        processed_docs=embedded_docs,
        collection_name=load_collection_name
    )

if __name__ == "__main__":
    rag_pipeline() 