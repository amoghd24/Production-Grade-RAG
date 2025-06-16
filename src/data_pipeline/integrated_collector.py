"""
Restructured data collection pipeline for the Second Brain AI Assistant.
Each class handles a specific responsibility in the data collection process.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime

from src.data_pipeline.notion_collector import collect_notion_documents, NotionCollector
from src.data_pipeline.web_crawler import WebCrawler, BrowserConfig, CrawlerRunConfig, MemoryAdaptiveDispatcher, AsyncWebCrawler
from src.models.schemas import Document, ProcessingStatus
from src.utils.logger import LoggerMixin

class NotionDataCollector(LoggerMixin):
    """Handles collection of documents from Notion workspace"""
    
    def __init__(self):
        self.notion_documents: List[Document] = []
        self.embedded_urls: Set[str] = set()
    
    async def collect(
        self,
        api_key: Optional[str] = None,
        database_id: Optional[str] = None,
        search_query: str = ""
    ) -> Tuple[List[Document], Set[str]]:
        """Collect documents from Notion workspace"""
        self.logger.info("📝 Collecting Notion documents...")
        
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
            
            return self.notion_documents, self.embedded_urls
            
        except Exception as e:
            self.logger.error(f"❌ Error collecting Notion data: {str(e)}")
            return [], set()

class WebDataCollector(LoggerMixin):
    """Handles crawling of embedded links"""
    
    def __init__(self):
        self.crawled_documents: List[Document] = []
    
    async def collect(
        self,
        urls: Set[str],
        max_pages: int = 1000000
    ) -> List[Document]:
        """Crawl embedded links from documents"""
        if not urls:
            return []

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
                    urls=list(urls)[:max_pages],
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
                
            return self.crawled_documents
                
        except Exception as e:
            self.logger.error(f"Error crawling embedded links: {str(e)}")
            return []

class DocumentCombiner(LoggerMixin):
    """Handles combining documents and generating statistics"""
    
    def __init__(self):
        self.all_documents: List[Document] = []
        self.notion_documents: List[Document] = []
        self.crawled_documents: List[Document] = []
    
    def combine(
        self,
        notion_documents: List[Document],
        crawled_documents: List[Document]
    ) -> Tuple[List[Document], Dict[str, any]]:
        """Combine documents and generate statistics"""
        self.notion_documents = notion_documents
        self.crawled_documents = crawled_documents
        self.all_documents = []
        self.all_documents.extend(notion_documents)
        self.all_documents.extend(crawled_documents)
        
        stats = self._generate_stats()
        return self.all_documents, stats
    
    def _generate_stats(self) -> Dict[str, any]:
        """Generate statistics about the collected data"""
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
