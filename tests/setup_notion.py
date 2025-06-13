#!/usr/bin/env python3
"""
Simple script to test Notion API connection.
"""

import os
import sys
from pathlib import Path
import asyncio
from src.data_pipeline.notion_collector import NotionCollector

def test_notion_connection():
    """Test the Notion API connection and return status."""
    try:
        async def test():
            try:
                collector = NotionCollector()
                pages = collector.search_pages()
                return len(pages)
            except Exception as e:
                return str(e)
        
        result = asyncio.run(test())
        
        if isinstance(result, int):
            print(f"✅ Connected to Notion. Found {result} accessible pages.")
            return True
        else:
            print(f"❌ Notion connection failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing connection: {str(e)}")
        return False

if __name__ == "__main__":
    test_notion_connection() 