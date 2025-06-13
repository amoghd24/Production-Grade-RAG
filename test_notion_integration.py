#!/usr/bin/env python3
"""
Test Notion Integration following DecodingML workflow.
Tests each step of the process like their Module 2.
"""

import os
import asyncio
from pathlib import Path


def print_step(step_num: int, title: str):
    """Print formatted step header."""
    print(f"\n{'='*60}")
    print(f"📋 STEP {step_num}: {title}")
    print('='*60)


async def test_step_1_environment():
    """Test Step 1: Environment Configuration (like their setup)"""
    print_step(1, "Environment Configuration")
    
    # Check for DecodingML naming first
    notion_key = os.getenv("NOTION_SECRET_KEY") or os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")
    
    print(f"🔑 NOTION_SECRET_KEY: {'✅ Set' if os.getenv('NOTION_SECRET_KEY') else '❌ Not set'}")
    print(f"🔑 NOTION_API_KEY (fallback): {'✅ Set' if os.getenv('NOTION_API_KEY') else '❌ Not set'}")
    print(f"📊 NOTION_DATABASE_ID: {'✅ Set' if database_id else '❌ Not set (optional)'}")
    
    if not notion_key:
        print("\n❌ Missing Notion credentials!")
        print("📖 Setup instructions:")
        print("1. Go to: https://www.notion.so/profile")
        print("2. Create integration following DecodingML tutorial")
        print("3. Set NOTION_SECRET_KEY environment variable")
        print("4. Share your database with the integration")
        return False
    
    print("✅ Environment configuration passed!")
    return True


async def test_step_2_notion_connection():
    """Test Step 2: Notion API Connection"""
    print_step(2, "Notion API Connection")
    
    try:
        from src.data_pipeline.notion_collector import NotionCollector
        
        collector = NotionCollector()
        print("🔌 Initializing Notion collector...")
        
        # Test basic connection
        pages = collector.search_pages()
        print(f"✅ Connection successful! Found {len(pages)} accessible pages")
        
        if len(pages) == 0:
            print("⚠️ No pages found. Make sure you've shared pages with your integration!")
            print("📖 To share pages: Go to page → '...' menu → 'Add connections' → Select your integration")
        
        return len(pages) > 0
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False


async def test_step_3_collect_notion_data():
    """Test Step 3: Collect Notion Data (like make collect-notion-data-pipeline)"""
    print_step(3, "Collect Notion Data")
    
    try:
        from src.data_pipeline.integrated_collector import IntegratedDataCollector
        
        print("📝 Collecting Notion documents...")
        collector = IntegratedDataCollector()
        documents = await collector.collect_all_data(
            notion_api_key=os.getenv("NOTION_API_KEY"),
            notion_database_id=os.getenv("NOTION_DATABASE_ID"),
            search_query="",
            max_crawl_pages=500  # Test with 500 pages
        )
        
        print(f"📊 Collection Results:")
        print(f"   Total documents: {len(documents)}")
        print(f"   Notion documents: {len(collector.notion_documents)}")
        print(f"   Crawled documents: {len(collector.crawled_documents)}")
        
        if documents:
            print(f"\n📄 Sample document: '{documents[0].title}'")
            print(f"   Source: {documents[0].source.value}")
            print(f"   Content length: {len(documents[0].content)} chars")
            
        print("✅ Data collection successful!")
        return len(documents) > 0
        
    except Exception as e:
        print(f"❌ Data collection failed: {str(e)}")
        return False


async def test_step_4_process_documents():
    """Test Step 4: Process Documents (quality scoring, chunking)"""
    print_step(4, "Process Documents")
    
    try:
        from src.feature_pipeline.document_processor import DocumentProcessor
        from src.data_pipeline.integrated_collector import collect_second_brain_data
        
        # Get some test documents
        documents, _ = await collect_second_brain_data(
            max_crawl_pages=1,
            include_web_crawling=False
        )
        
        if not documents:
            print("⚠️ No documents to process")
            return False
        
        processor = DocumentProcessor()
        test_doc = documents[0]
        
        print(f"🔄 Processing document: '{test_doc.title}'")
        processed = await processor.process_document(test_doc)
        
        print(f"📊 Processing Results:")
        print(f"   Quality score: {processed.get('quality_score', 0):.3f}")
        print(f"   Number of chunks: {len(processed.get('chunks', []))}")
        print(f"   Embeddings generated: {'✅' if any(c.embedding for c in processed.get('chunks', [])) else '❌'}")
        
        print("✅ Document processing successful!")
        return True
        
    except Exception as e:
        print(f"❌ Document processing failed: {str(e)}")
        return False


async def test_step_5_mongodb_storage():
    """Test Step 5: MongoDB Storage (like their ETL pipeline)"""
    print_step(5, "MongoDB Storage")
    
    try:
        from src.data_pipeline.etl_pipeline import run_second_brain_etl
        
        print("💾 Running mini ETL pipeline...")
        stats = await run_second_brain_etl(
            max_crawl_pages=500,  # Very small test
            quality_threshold=0.1  # Low threshold for testing
        )
        
        print(f"📊 ETL Results:")
        print(f"   Documents collected: {stats.get('total_documents_collected', 0)}")
        print(f"   Documents processed: {stats.get('documents_processed', 0)}")
        print(f"   Documents stored: {stats.get('documents_stored', 0)}")
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        print(f"   Errors: {len(stats.get('errors', []))}")
        
        if stats.get('documents_stored', 0) > 0:
            print("✅ MongoDB storage successful!")
            return True
        else:
            print("⚠️ No documents were stored to MongoDB")
            return False
            
    except Exception as e:
        print(f"❌ MongoDB storage failed: {str(e)}")
        return False


async def main():
    """Run all tests following DecodingML workflow."""
    print("🧠 Second Brain AI Assistant - Integration Test")
    print("Following DecodingML Module 2 workflow...")
    
    # Run each test step
    tests = [
        test_step_1_environment,
        test_step_2_notion_connection,
        test_step_3_collect_notion_data,
        test_step_4_process_documents,
        test_step_5_mongodb_storage
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
            
            if not result:
                print(f"\n⚠️ Test failed. Check the setup and try again.")
                break
                
        except Exception as e:
            print(f"\n❌ Test error: {str(e)}")
            results.append(False)
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 INTEGRATION TEST SUMMARY")
    print('='*60)
    
    test_names = [
        "Environment Setup",
        "Notion Connection", 
        "Data Collection",
        "Document Processing",
        "MongoDB Storage"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"Step {i+1}: {name:<20} {status}")
    
    total_passed = sum(results)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Your Notion integration is working!")
        print("📝 Next step: Run full ETL pipeline with your actual data")
    else:
        print(f"\n⚠️ {len(results) - total_passed} tests failed. Check the errors above.")
    
    return total_passed == len(results)


if __name__ == "__main__":
    asyncio.run(main()) 