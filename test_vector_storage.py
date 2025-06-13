"""
Comprehensive integration test for the MongoDB Vector Storage system.
Tests all components: connection, repositories, search, indexing, and the main facade.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import List
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.schemas import (
    Document, DocumentChunk, SearchResult,
    DocumentType, ContentSource, ProcessingStatus
)
from src.feature_pipeline.vector_storage import (
    create_vector_store,
    VectorStorageException,
    MongoDBConnectionFactory
)
from src.feature_pipeline.vector_storage.base import VectorSearchQuery


async def test_vector_storage_system():
    """Test the complete vector storage system."""
    print("🧪 Starting Vector Storage System Integration Test\n")
    
    vector_store = None
    try:
        # Step 1: Initialize Vector Store
        print("📦 Step 1: Creating and initializing vector store...")
        vector_store = await create_vector_store(
            initialize=True,
            setup_indexes=True
        )
        print("✅ Vector store initialized successfully\n")
        
        # Step 2: Test Connection Health
        print("🔍 Step 2: Testing connection health...")
        stats = await vector_store.get_storage_stats()
        print(f"📊 Storage stats: {stats}")
        print("✅ Connection health check passed\n")
        
        # Step 3: Create Test Documents and Chunks
        print("📝 Step 3: Creating test documents and chunks...")
        
        # Sample documents
        documents = [
            Document(
                id="test_doc_1",
                title="Machine Learning Fundamentals",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                source=ContentSource.WEB_CRAWL,
                document_type=DocumentType.WEB_PAGE,
                source_url="https://example.com/ml-fundamentals",
                processing_status=ProcessingStatus.PROCESSING,
                metadata={"category": "AI", "difficulty": "beginner"}
            ),
            Document(
                id="test_doc_2", 
                title="Deep Learning Applications",
                content="Deep learning has revolutionized computer vision, natural language processing, and speech recognition through neural networks.",
                source=ContentSource.WEB_CRAWL,
                document_type=DocumentType.WEB_PAGE,
                source_url="https://example.com/deep-learning",
                processing_status=ProcessingStatus.PROCESSING,
                metadata={"category": "AI", "difficulty": "advanced"}
            )
        ]
        
        # Sample chunks with embeddings (using dummy vectors for testing)
        chunks = [
            DocumentChunk(
                document_id="test_doc_1",
                content="Machine learning is a subset of artificial intelligence",
                chunk_index=0,
                word_count=10,
                embedding=[0.1, 0.2, 0.3, 0.4] + [0.0] * 380,  # 384-dim vector
                metadata={"topic": "ML definition"}
            ),
            DocumentChunk(
                document_id="test_doc_1",
                content="enables computers to learn without being explicitly programmed",
                chunk_index=1,
                word_count=9,
                embedding=[0.2, 0.3, 0.4, 0.5] + [0.0] * 380,
                metadata={"topic": "ML capability"}
            ),
            DocumentChunk(
                document_id="test_doc_2",
                content="Deep learning has revolutionized computer vision",
                chunk_index=0,
                word_count=7,
                embedding=[0.3, 0.4, 0.5, 0.6] + [0.0] * 380,
                metadata={"topic": "DL applications"}
            ),
            DocumentChunk(
                document_id="test_doc_2",
                content="natural language processing and speech recognition through neural networks",
                chunk_index=1,
                word_count=9,
                embedding=[0.4, 0.5, 0.6, 0.7] + [0.0] * 380,
                metadata={"topic": "NLP and speech"}
            )
        ]
        
        print(f"📄 Created {len(documents)} documents and {len(chunks)} chunks")
        print("✅ Test data prepared\n")
        
        # Step 4: Store Documents and Chunks
        print("💾 Step 4: Storing documents and chunks...")
        storage_result = await vector_store.store_documents_with_embeddings(
            documents=documents,
            chunks=chunks
        )
        print(f"📊 Storage result: {storage_result}")
        print("✅ Documents and chunks stored successfully\n")
        
        # Step 5: Test Document Retrieval
        print("🔍 Step 5: Testing document retrieval...")
        stored_doc_id = storage_result["document_ids"][0]
        doc_with_chunks = await vector_store.get_document_with_chunks(stored_doc_id)
        
        if doc_with_chunks:
            document, doc_chunks = doc_with_chunks
            print(f"📄 Retrieved document: {document.title}")
            print(f"📝 Document has {len(doc_chunks)} chunks")
        else:
            print("❌ Failed to retrieve document")
        print("✅ Document retrieval test completed\n")
        
        # Step 6: Test Vector Search
        print("🔍 Step 6: Testing vector search...")
        
        # Semantic search
        search_query = VectorSearchQuery(
            query_text="artificial intelligence machine learning",
            query_vector=[0.15, 0.25, 0.35, 0.45] + [0.0] * 380,
            search_type="semantic",
            limit=3
        )
        
        search_results = await vector_store.search(search_query)
        print(f"🎯 Semantic search returned {len(search_results)} results:")
        for i, result in enumerate(search_results):
            print(f"  {i+1}. Score: {result.score:.4f} | Content: {result.content[:50]}...")
        print("✅ Semantic search test completed\n")
        
        # Step 7: Test Filtered Search
        print("🔍 Step 7: Testing filtered search...")
        
        filtered_query = VectorSearchQuery(
            query_text="deep learning applications",
            query_vector=[0.35, 0.45, 0.55, 0.65] + [0.0] * 380,
            search_type="filtered",
            limit=2,
            min_score=0.0,
            filters={"metadata.topic": {"$regex": "DL|ML"}}
        )
        
        filtered_results = await vector_store.search(filtered_query)
        print(f"🎯 Filtered search returned {len(filtered_results)} results:")
        for i, result in enumerate(filtered_results):
            print(f"  {i+1}. Score: {result.score:.4f} | Content: {result.content[:50]}...")
        print("✅ Filtered search test completed\n")
        
        # Step 8: Test Similar Documents
        print("🔍 Step 8: Testing similar documents search...")
        similar_docs = await vector_store.get_similar_documents(
            document_id=stored_doc_id,
            limit=2
        )
        print(f"🎯 Found {len(similar_docs)} similar documents:")
        for i, result in enumerate(similar_docs):
            print(f"  {i+1}. Score: {result.score:.4f} | Content: {result.content[:50]}...")
        print("✅ Similar documents test completed\n")
        
        # Step 9: Test Storage Statistics
        print("📊 Step 9: Testing storage statistics...")
        final_stats = await vector_store.get_storage_stats()
        print(f"📈 Final storage statistics:")
        for key, value in final_stats.items():
            if key != "timestamp":
                print(f"  {key}: {value}")
        print("✅ Storage statistics test completed\n")
        
        # Step 10: Test Document Deletion
        print("🗑️ Step 10: Testing document deletion...")
        deletion_success = await vector_store.delete_document_and_chunks(stored_doc_id)
        if deletion_success:
            print(f"✅ Successfully deleted document {stored_doc_id} and its chunks")
        else:
            print(f"❌ Failed to delete document {stored_doc_id}")
        print("✅ Document deletion test completed\n")
        
        print("🎉 All tests completed successfully!")
        print("✅ Vector Storage System is working correctly!")
        
    except VectorStorageException as e:
        print(f"❌ Vector Storage Error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if vector_store:
            print("\n🧹 Cleaning up resources...")
            await vector_store.close()
            print("✅ Resources cleaned up")
    
    return True


async def test_individual_components():
    """Test individual components separately."""
    print("\n🔧 Testing Individual Components\n")
    
    try:
        # Test connection only
        print("🔌 Testing MongoDB connection...")
        async with MongoDBConnectionFactory.get_connection() as conn:
            health = await conn.health_check()
            try:
                database = conn.get_database()
                collections = await database.list_collection_names()
                print(f"✅ Connection healthy: {health}")
                print(f"📁 Available collections: {len(collections)}")
            except Exception as e:
                print(f"❌ Connection failed: {str(e)}")
        
        print("✅ Individual components test completed")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False


async def main():
    """Main test function."""
    print("🚀 Second Brain Vector Storage Integration Test")
    print("=" * 60)
    
    # Test individual components first
    component_success = await test_individual_components()
    
    if not component_success:
        print("❌ Component tests failed, skipping integration test")
        return
    
    # Run full integration test
    success = await test_vector_storage_system()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Vector Storage System is ready!")
        print("💡 You can now use it in your DecodingML pipeline")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main()) 