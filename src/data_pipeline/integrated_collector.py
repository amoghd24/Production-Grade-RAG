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
from src.data_pipeline.web_crawler import BrowserConfig, CrawlerRunConfig, MemoryAdaptiveDispatcher, AsyncWebCrawler


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
        notion_api_key: str,
        notion_database_id: str,
        search_query: str = "",
        max_crawl_pages: int = 1000000
    ) -> List[Document]:
        """Collect data from all sources."""
        try:
            # Step 1: Collect Notion data
            await self._collect_notion_data(
                api_key=notion_api_key,
                database_id=notion_database_id,
                search_query=search_query
            )
            
            # Step 2: Crawl embedded links if any found
            if self.embedded_urls:
                await self._crawl_embedded_links(max_pages=max_crawl_pages)
            
            # Step 3: Combine all documents
            self._combine_documents()
            
            return self.all_documents
            
        except Exception as e:
            raise Exception(f"Error in data collection: {str(e)}")
    
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
    
    async def _crawl_embedded_links(self, max_pages: int = 1000000) -> None:
        """Crawl embedded links from Notion documents."""
        if not self.embedded_urls:
            return

        try:
            browser_config = BrowserConfig(headless=True, verbose=False)
            run_config = CrawlerRunConfig(cache_mode="BYPASS", stream=False)
            dispatcher = MemoryAdaptiveDispatcher(
                memory_threshold_percent=70.0,
                check_interval=1.0,
                max_session_permit=10
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun_many(
                    urls=self.embedded_urls[:max_pages],
                    config=run_config,
                    dispatcher=dispatcher
                )
                
                for result in results:
                    if result.get('success'):
                        doc = await crawler.crawl_url(result['url'])
                        if doc:
                            self.crawled_documents.append(doc)
                    else:
                        self.logger.error(f"Failed to crawl {result['url']}: {result.get('error_message')}")
                
        except Exception as e:
            raise Exception(f"Error crawling embedded links: {str(e)}")
    
    def _combine_documents(self) -> None:
        """Combine all collected documents."""
        self.all_documents = []
        self.all_documents.extend(self.notion_documents)
        self.all_documents.extend(self.crawled_documents)
    
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
        max_crawl_pages=max_crawl_pages
    )
    
    stats = collector.get_collection_stats()
    
    return documents, stats
