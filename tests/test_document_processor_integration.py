import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict

from src.data_pipeline.integrated_collector import IntegratedDataCollector
from src.feature_pipeline.document_processor import DocumentProcessor
from src.models.schemas import Document, ProcessingStatus

# Get Notion credentials from environment variables
NOTION_API_KEY = os.getenv("NOTION_SECRET_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

if not NOTION_API_KEY:
    pytest.skip("NOTION_SECRET_KEY environment variable not set")

@pytest.fixture
async def collector():
    """Create an IntegratedDataCollector instance."""
    return IntegratedDataCollector()

@pytest.fixture
async def processor():
    """Create a DocumentProcessor instance."""
    processor = DocumentProcessor()
    await processor.initialize_embedding_model()
    return processor

@pytest.mark.asyncio
async def test_process_notion_content(collector, processor):
    """Test processing of content from our Notion workspace."""
    # Step 1: Collect documents from our Notion workspace
    documents = await collector.collect_all_data(
        notion_api_key=NOTION_API_KEY,
        notion_database_id=NOTION_DATABASE_ID,
        max_crawl_pages=5,  # Limit to 5 pages for testing
        include_web_crawling=True
    )
    
    assert len(documents) > 0, "No documents were collected from Notion"
    
    # Step 2: Process each document
    processed_results = []
    for doc in documents:
        result = await processor.process_document(doc)
        processed_results.append(result)
        
        # Log the results for analysis
        print(f"\nProcessing results for: {doc.title}")
        print(f"Source: {doc.source}")
        print(f"Type: {doc.document_type}")
        print(f"Quality Score: {result['quality_score']:.3f}")
        print(f"Processed: {result['processed']}")
        print(f"Chunks: {len(result['chunks'])}")
        if result['chunks']:
            print(f"First chunk preview: {result['chunks'][0].content[:100]}...")
    
    # Step 3: Analyze results
    processed_count = sum(1 for r in processed_results if r["processed"])
    print(f"\nSummary:")
    print(f"Total documents: {len(documents)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Average quality score: {sum(r['quality_score'] for r in processed_results) / len(processed_results):.3f}")
    
    # Step 4: Verify processing results
    for result in processed_results:
        # Check if high-quality documents were processed
        if result["quality_score"] >= 0.5:
            assert result["processed"] is True, "High-quality document should be processed"
            assert len(result["chunks"]) > 0, "Processed document should have chunks"
            assert result["document"].processing_status == ProcessingStatus.COMPLETED
        
        # Check if low-quality documents were filtered
        if result["quality_score"] < 0.3:
            assert result["processed"] is False, "Low-quality document should be filtered"
            assert len(result["chunks"]) == 0, "Filtered document should have no chunks"
        
        # Verify chunk properties
        for chunk in result["chunks"]:
            assert chunk.document_id == result["document"].id
            assert chunk.embedding is not None
            assert len(chunk.embedding) > 0
            assert chunk.embedding_model == processor.model_name

@pytest.mark.asyncio
async def test_batch_processing_notion_content(collector, processor):
    """Test batch processing of content from our Notion workspace."""
    # Collect documents
    documents = await collector.collect_all_data(
        notion_api_key=NOTION_API_KEY,
        notion_database_id=NOTION_DATABASE_ID,
        max_crawl_pages=5,  # Limit to 5 pages for testing
        include_web_crawling=True
    )
    
    # Process in batch
    results = await processor.process_documents_batch(documents)
    
    # Verify batch processing results
    assert len(results) == len(documents)
    
    # Log batch processing results
    print("\nBatch Processing Results:")
    print(f"Total documents: {len(documents)}")
    print(f"Successfully processed: {sum(1 for r in results if r['processed'])}")
    print(f"Average quality score: {sum(r['quality_score'] for r in results) / len(results):.3f}")
    
    # Verify each result
    for result in results:
        assert "document" in result
        assert "quality_score" in result
        assert "processed" in result
        assert "chunks" in result
        
        if result["processed"]:
            assert len(result["chunks"]) > 0
            assert result["document"].processing_status == ProcessingStatus.COMPLETED

if __name__ == "__main__":
    if not NOTION_API_KEY:
        print("Error: NOTION_SECRET_KEY environment variable not set")
        exit(1)
    
    async def main():
        # Initialize collector and processor
        collector = IntegratedDataCollector()
        processor = DocumentProcessor()
        await processor.initialize_embedding_model()
        
        # Run the test
        await test_process_notion_content(collector, processor)
    
    # Run the async main function
    asyncio.run(main()) 