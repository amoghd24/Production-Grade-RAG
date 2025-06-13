#!/usr/bin/env python3
"""
Setup script for Notion API integration.
Helps configure the Second Brain AI Assistant with Notion workspace.
"""

import os
import sys
from pathlib import Path


def print_header():
    print("🧠" + "="*60)
    print("    Second Brain AI Assistant - Notion Setup")
    print("="*60)


def print_instructions():
    print("\n📖 Setup Instructions:")
    print("="*30)
    print("1. Create a Notion Integration:")
    print("   → Go to: https://www.notion.com/my-integrations")
    print("   → Click '+ New integration'")
    print("   → Enter name: 'Second Brain AI Assistant'")
    print("   → Select your workspace")
    print("   → Copy the 'Internal Integration Secret'")
    
    print("\n2. Give Integration Permissions:")
    print("   → Go to any Notion page you want to include")
    print("   → Click the '...' menu (top-right)")
    print("   → Scroll to '+ Add connections'")
    print("   → Find and select your integration")
    print("   → Confirm access to page and child pages")
    
    print("\n3. Optional - Get Database ID:")
    print("   → Open your Notion database in browser")
    print("   → Copy the database ID from URL:")
    print("   → URL format: notion.so/[workspace]/[DATABASE_ID]?v=...")
    print("   → Database ID is the 32-character string")


def create_env_file():
    """Create or update .env file with Notion configuration."""
    env_path = Path(".env")
    
    print(f"\n⚙️ Configuration:")
    print("="*20)
    
    # Get API key
    api_key = input("🔑 Enter your Notion API Secret (Integration Token): ").strip()
    if not api_key:
        print("❌ API key is required!")
        return False
    
    # Get optional database ID
    database_id = input("📊 Enter Database ID (optional, press Enter to skip): ").strip()
    
    # Prepare environment variables
    env_content = []
    
    # Read existing .env if it exists
    if env_path.exists():
        with open(env_path, 'r') as f:
            existing_lines = f.readlines()
        
        # Keep non-Notion variables
        for line in existing_lines:
            if not line.strip().startswith(('NOTION_API_KEY', 'NOTION_DATABASE_ID')):
                env_content.append(line.rstrip())
    
    # Add Notion configuration
    env_content.append(f"NOTION_API_KEY={api_key}")
    if database_id:
        env_content.append(f"NOTION_DATABASE_ID={database_id}")
    
    # Write to .env file
    with open(env_path, 'w') as f:
        f.write('\n'.join(env_content) + '\n')
    
    print(f"✅ Configuration saved to {env_path}")
    return True


def test_notion_connection():
    """Test the Notion API connection."""
    print("\n🧪 Testing Notion Connection:")
    print("="*35)
    
    try:
        # Import and test
        import asyncio
        from src.data_pipeline.notion_collector import NotionCollector
        
        async def test():
            try:
                collector = NotionCollector()
                pages = collector.search_pages()
                return len(pages)
            except Exception as e:
                return str(e)
        
        result = asyncio.run(test())
        
        if isinstance(result, int):
            print(f"✅ Connection successful! Found {result} accessible pages")
            return True
        else:
            print(f"❌ Connection failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing connection: {str(e)}")
        return False


def run_data_collection_demo():
    """Run a demo of the data collection."""
    print("\n🚀 Running Data Collection Demo:")
    print("="*40)
    
    try:
        import asyncio
        from src.data_pipeline.integrated_collector import collect_second_brain_data
        
        async def demo():
            documents, stats = await collect_second_brain_data(
                max_crawl_pages=5,  # Limit for demo
                include_web_crawling=True
            )
            return documents, stats
        
        documents, stats = asyncio.run(demo())
        
        print(f"📊 Demo Results:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Notion documents: {stats['notion_documents']}")
        print(f"   Crawled documents: {stats['crawled_documents']}")
        print(f"   Total words: {stats['total_word_count']:,}")
        
        if documents:
            print(f"\n📄 Sample document: '{documents[0].title}'")
            print(f"   Content length: {len(documents[0].content)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        return False


def main():
    """Main setup flow."""
    print_header()
    print_instructions()
    
    # Step 1: Configure environment
    if not create_env_file():
        return
    
    # Step 2: Test connection
    if not test_notion_connection():
        print("\n⚠️ Connection test failed. Please check your configuration.")
        return
    
    # Step 3: Ask if user wants to run demo
    print("\n" + "="*60)
    run_demo = input("🎯 Run data collection demo? (y/N): ").lower().strip()
    
    if run_demo == 'y':
        if run_data_collection_demo():
            print("\n✅ Demo completed successfully!")
        else:
            print("\n⚠️ Demo encountered issues. Check your configuration.")
    
    print("\n🎉 Setup completed!")
    print("\n📝 Next steps:")
    print("   → Test collection: python -m src.data_pipeline.integrated_collector")
    print("   → Check data: Look in data/ directory after collection")
    print("   → Continue with: Training pipeline (Lesson 3)")


if __name__ == "__main__":
    main() 