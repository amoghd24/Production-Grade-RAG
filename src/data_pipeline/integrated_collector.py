"""
Integrated data collection pipeline for the Second Brain AI Assistant.
Combines Notion API collection with web crawling as described in DecodingML course.
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.data_pipeline.notion_collector import collect_notion_documents, NotionCollector
from src.data_pipeline.web_crawler import WebCrawler
from src.models.schemas import Document, ProcessingStatus
from src.utils.logger import LoggerMixin


class IntegratedDataCollector(LoggerMixin):
    """
    Integrated data collector following DecodingML methodology:
    1. Collect Notion documents
    2. Extract embedded links
    3. Crawl embedded links
    4. Combine all documents
    """
    
    def __init__(self):
        """Initialize the integrated collector."""
        self.notion_documents: List[Document] = []
        self.crawled_documents: List[Document] = []
        self.all_documents: List[Document] = []
    
    async def collect_all_data(
        self,
        notion_api_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,
        notion_search_query: str = "",
        max_crawl_pages: int = 100,
        crawl_embedded_links: bool = True
    ) -> List[Document]:
        """
        Complete data collection pipeline following DecodingML methodology.
        
        Args:
            notion_api_key: Notion API key
            notion_database_id: Specific database to query
            notion_search_query: Search filter for Notion pages
            max_crawl_pages: Maximum pages to crawl
            crawl_embedded_links: Whether to crawl embedded links
            
        Returns:
            List of all collected documents
        """
        self.logger.info("🚀 Starting integrated data collection pipeline...")
        
        # Step 1: Collect Notion documents
        await self._collect_notion_data(
            api_key=notion_api_key,
            database_id=notion_database_id,
            search_query=notion_search_query
        )
        
        # Step 2: Crawl embedded links if enabled
        if crawl_embedded_links and self.notion_documents:
            await self._crawl_embedded_links(max_pages=max_crawl_pages)
        
        # Step 3: Combine all documents
        self._combine_documents()
        
        self.logger.info(f"✅ Data collection complete! Total documents: {len(self.all_documents)}")
        self.logger.info(f"📄 Notion documents: {len(self.notion_documents)}")
        self.logger.info(f"🌐 Crawled documents: {len(self.crawled_documents)}")
        
        return self.all_documents
    
    async def _collect_notion_data(
        self,
        api_key: Optional[str] = None,
        database_id: Optional[str] = None,
        search_query: str = ""
    ) -> None:
        """Collect documents from Notion workspace."""
        self.logger.info("📝 Step 1: Collecting Notion documents...")
        
        try:
            documents, embedded_urls = await collect_notion_documents(
                api_key=api_key,
                search_query=search_query,
                database_id=database_id
            )
            
            self.notion_documents = documents
            self.embedded_urls = embedded_urls
            
            self.logger.info(f"✅ Collected {len(documents)} Notion documents")
            self.logger.info(f"🔗 Found {len(embedded_urls)} embedded URLs")
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting Notion data: {str(e)}")
            self.notion_documents = []
            self.embedded_urls = []
    
    async def _crawl_embedded_links(self, max_pages: int = 100) -> None:
        """Crawl all embedded links found in Notion documents."""
        if not hasattr(self, 'embedded_urls') or not self.embedded_urls:
            self.logger.info("ℹ️ No embedded URLs to crawl")
            return
        
        self.logger.info(f"🕷️ Step 2: Crawling {len(self.embedded_urls)} embedded URLs...")
        
        # Limit URLs if too many
        urls_to_crawl = self.embedded_urls[:max_pages]
        if len(self.embedded_urls) > max_pages:
            self.logger.info(f"⚠️ Limiting crawl to {max_pages} URLs (found {len(self.embedded_urls)})")
        
        try:
            async with WebCrawler() as crawler:
                crawled_documents = await crawler.crawl_multiple_urls(urls_to_crawl)
                self.crawled_documents = crawled_documents
                
                self.logger.info(f"✅ Successfully crawled {len(crawled_documents)} web pages")
                
        except Exception as e:
            self.logger.error(f"❌ Error crawling embedded links: {str(e)}")
            self.crawled_documents = []
    
    def _combine_documents(self) -> None:
        """Combine Notion and crawled documents into single collection."""
        self.logger.info("🔗 Step 3: Combining all documents...")
        
        # Start with Notion documents
        all_docs = self.notion_documents.copy()
        
        # Add crawled documents
        all_docs.extend(self.crawled_documents)
        
        # Update processing status
        for doc in all_docs:
            doc.processing_status = ProcessingStatus.COMPLETED
            doc.updated_at = datetime.utcnow()
        
        self.all_documents = all_docs
        
        self.logger.info(f"✅ Combined {len(all_docs)} total documents")
    
    def get_collection_stats(self) -> Dict[str, any]:
        """Get statistics about the collected data."""
        stats = {
            "total_documents": len(self.all_documents),
            "notion_documents": len(self.notion_documents),
            "crawled_documents": len(self.crawled_documents),
            "total_word_count": sum(doc.word_count or 0 for doc in self.all_documents),
            "sources": {},
            "document_types": {},
            "collection_timestamp": datetime.utcnow().isoformat()
        }
        
        # Count by source
        for doc in self.all_documents:
            source = doc.source.value
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        # Count by document type
        for doc in self.all_documents:
            doc_type = doc.document_type.value
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
        
        return stats


# Convenience functions for easy use
async def collect_second_brain_data(
    notion_api_key: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    max_crawl_pages: int = 100,
    include_web_crawling: bool = True
) -> Tuple[List[Document], Dict[str, any]]:
    """
    Main function to collect all Second Brain data.
    
    Args:
        notion_api_key: Notion API key
        notion_database_id: Specific database to collect from
        max_crawl_pages: Maximum pages to crawl
        include_web_crawling: Whether to crawl embedded links
        
    Returns:
        Tuple of (documents, collection_stats)
    """
    collector = IntegratedDataCollector()
    
    documents = await collector.collect_all_data(
        notion_api_key=notion_api_key,
        notion_database_id=notion_database_id,
        max_crawl_pages=max_crawl_pages,
        crawl_embedded_links=include_web_crawling
    )
    
    stats = collector.get_collection_stats()
    
    return documents, stats


# CLI for testing the complete pipeline
async def main():
    """Test the integrated data collection pipeline."""
    import os
    
    # Get environment variables
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_database_id = os.getenv("NOTION_DATABASE_ID")
    
    if not notion_api_key:
        print("❌ Please set NOTION_API_KEY environment variable")
        print("📖 Follow setup instructions:")
        print("1. Go to https://www.notion.com/my-integrations")
        print("2. Create a new integration")
        print("3. Copy the API secret")
        print("4. Give integration access to your pages")
        return
    
    print("🚀 Testing complete Second Brain data collection...")
    print(f"🔑 Using API key: {notion_api_key[:20]}...")
    if notion_database_id:
        print(f"📊 Target database: {notion_database_id}")
    
    # Collect all data
    documents, stats = await collect_second_brain_data(
        notion_api_key=notion_api_key,
        notion_database_id=notion_database_id,
        max_crawl_pages=50,  # Limit for testing
        include_web_crawling=True
    )
    
    # Display results
    print("\n" + "="*50)
    print("📊 COLLECTION RESULTS")
    print("="*50)
    print(f"Total documents collected: {stats['total_documents']}")
    print(f"Notion documents: {stats['notion_documents']}")
    print(f"Crawled web pages: {stats['crawled_documents']}")
    print(f"Total word count: {stats['total_word_count']:,}")
    
    print(f"\n📈 Sources breakdown:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count}")
    
    print(f"\n📝 Document types:")
    for doc_type, count in stats['document_types'].items():
        print(f"  {doc_type}: {count}")
    
    if documents:
        print(f"\n📄 Sample document:")
        doc = documents[0]
        print(f"Title: {doc.title}")
        print(f"Source: {doc.source.value}")
        print(f"Type: {doc.document_type.value}")
        print(f"Content length: {len(doc.content)} chars")
        print(f"Word count: {doc.word_count}")
        
        if hasattr(doc.metadata, 'embedded_links'):
            print(f"Embedded links: {len(doc.metadata.get('embedded_links', []))}")
    
    print("\n✅ Collection test completed!")


if __name__ == "__main__":
    asyncio.run(main()) 