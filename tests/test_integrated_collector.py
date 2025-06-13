#!/usr/bin/env python3
"""
Test script for IntegratedDataCollector functionality.
Tests Notion connection, MongoDB connection, and web crawling capabilities.
"""

import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.data_pipeline.integrated_collector import IntegratedDataCollector, collect_second_brain_data
from src.data_pipeline.notion_collector import NotionCollector
from src.data_pipeline.web_crawler import WebCrawler
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

async def test_integrated_collector() -> Dict[str, Any]:
    """
    Test the integrated collector functionality and return results.
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "tests": {},
        "errors": [],
        "crawled_content": {}  # New field to store crawled content
    }
    
    try:
        # Test Notion Connection
        try:
            notion_collector = NotionCollector()
            pages = notion_collector.search_pages()
            results["tests"]["notion"] = {
                "status": "success",
                "pages_found": len(pages)
            }
        except Exception as e:
            results["tests"]["notion"] = {
                "status": "failed",
                "error": str(e)
            }
            results["errors"].append(f"Notion test failed: {str(e)}")

        # Test MongoDB Connection
        try:
            mongo_url = os.getenv('MONGODB_URL')
            if not mongo_url:
                raise ValueError("MONGODB_URL environment variable not set")
            
            client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            results["tests"]["mongodb"] = {
                "status": "success"
            }
        except Exception as e:
            results["tests"]["mongodb"] = {
                "status": "failed",
                "error": str(e)
            }
            results["errors"].append(f"MongoDB test failed: {str(e)}")

        # Test Full Integrated Collector
        try:
            collector = IntegratedDataCollector()
            documents, stats = await collect_second_brain_data(
                notion_api_key=os.getenv('NOTION_API_KEY'),
                notion_database_id=os.getenv('NOTION_DATABASE_ID'),
                max_crawl_pages=5,  # Limit for testing
                include_web_crawling=True
            )
            
            results["tests"]["integrated_collector"] = {
                "status": "success",
                "documents_collected": len(documents),
                "stats": stats
            }
            
            # Store all collected documents content
            for doc in documents:
                results["crawled_content"][str(doc.source_url)] = {
                    "title": doc.title,
                    "content": doc.content,
                    "word_count": doc.word_count,
                    "source": doc.source.value,
                    "document_type": doc.document_type.value,
                    "metadata": doc.metadata
                }
            
        except Exception as e:
            results["tests"]["integrated_collector"] = {
                "status": "failed",
                "error": str(e)
            }
            results["errors"].append(f"Integrated collector test failed: {str(e)}")

    except Exception as e:
        results["errors"].append(f"Overall test failed: {str(e)}")

    return results

def save_test_results(results: Dict[str, Any], output_file: str = "test_results.json") -> None:
    """
    Save test results to a JSON file.
    """
    try:
        # Create a copy of results to modify for saving
        save_results = results.copy()
        
        # If content is too large, truncate it in the saved file
        if "crawled_content" in save_results:
            for url, content in save_results["crawled_content"].items():
                if len(content["content"]) > 10000:  # Truncate content longer than 1000 chars
                    content["content"] = content["content"][:1000] + "... [truncated]"
        
        with open(output_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"✅ Test results saved to {output_file}")
        
        # Save full content to a separate file if there's crawled content
        if results["crawled_content"]:
            full_content_file = output_file.replace(".json", "_full_content.json")
            with open(full_content_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Full content saved to {full_content_file}")
            
    except Exception as e:
        print(f"❌ Error saving test results: {str(e)}")

def print_test_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of the test results.
    """
    print("\n=== Test Summary ===")
    for test_name, test_result in results["tests"].items():
        status = "✅" if test_result["status"] == "success" else "❌"
        print(f"{status} {test_name}: {test_result['status']}")
    
    if results["errors"]:
        print("\n=== Errors ===")
        for error in results["errors"]:
            print(f"❌ {error}")

if __name__ == "__main__":
    # Run tests
    results = asyncio.run(test_integrated_collector())
    
    # Save results
    save_test_results(results)
    
    # Print summary
    print_test_summary(results) 