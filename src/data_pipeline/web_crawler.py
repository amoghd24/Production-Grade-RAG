"""
Web crawler for the Second Brain AI Assistant.
Implements async web crawling using Crawl4AI for optimal LLM-friendly content extraction.
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from crawl4ai import AsyncWebCrawler as Crawl4aiCrawler
from src.models.schemas import Document, ContentSource, DocumentType, ProcessingStatus
from src.models.WebCrawler import BrowserConfig, CrawlerRunConfig
from src.utils.logger import LoggerMixin

class CrawlerMonitor:
    """Monitor for crawler operations."""
    def __init__(self, display_mode: str = "DETAILED"):
        self.display_mode = display_mode

class MemoryAdaptiveDispatcher:
    """Dispatcher for managing concurrent crawler operations."""
    def __init__(
        self,
        memory_threshold_percent: float = 70.0,
        check_interval: float = 1.0,
        max_session_permit: int = 10,
        monitor: Optional[CrawlerMonitor] = None
    ):
        self.memory_threshold_percent = memory_threshold_percent
        self.check_interval = check_interval
        self.max_session_permit = max_session_permit
        self.monitor = monitor or CrawlerMonitor()

class AsyncWebCrawler(LoggerMixin):
    """
    Async web crawler using Crawl4AI for LLM-optimized content extraction.
    Maintains backward compatibility with the previous interface.
    """
    
    def __init__(self, config: Optional[BrowserConfig] = None):
        """Initialize the web crawler."""
        self.config = config or BrowserConfig()
        self._crawl4ai_crawler = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._crawl4ai_crawler = Crawl4aiCrawler(
            headless=self.config.headless,
            verbose=self.config.verbose
        )
        await self._crawl4ai_crawler.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._crawl4ai_crawler:
            await self._crawl4ai_crawler.__aexit__(exc_type, exc_val, exc_tb)
            self._crawl4ai_crawler = None
    
    def _convert_to_document(self, crawl_result, original_url: str) -> Optional[Document]:
        """Convert Crawl4AI result to our Document schema."""
        try:
            if not crawl_result.success:
                self.logger.warning(f"Crawl failed for {original_url}: {crawl_result.error_message}")
                return None
            
            # Extract title from metadata or use a default
            title = "Unknown Title"
            if hasattr(crawl_result, 'metadata') and crawl_result.metadata:
                title = crawl_result.metadata.get('title', title) or title
            
            # If no title in metadata, try to extract from markdown
            if title == "Unknown Title" and crawl_result.markdown:
                lines = crawl_result.markdown.split('\n')
                for line in lines:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        if title:  # Make sure it's not empty
                            break
            
            # Fallback to URL
            if not title or title == "Unknown Title":
                title = original_url
            
            # Ensure title is never None or empty
            if not title:
                title = f"Web Page - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Use Crawl4AI's markdown output (optimized for LLMs)
            content = crawl_result.markdown or ""
            
            document = Document(
                id=f"web_{datetime.utcnow().timestamp()}",
                title=title,
                content=content,
                source=ContentSource.WEB_CRAWL,
                source_url=original_url,
                document_type=DocumentType.WEB_PAGE,
                processing_status=ProcessingStatus.COMPLETED,
                metadata={
                    "crawled_at": datetime.utcnow().isoformat(),
                    "url": original_url,
                    "content_format": "markdown",
                    "status_code": getattr(crawl_result, 'status_code', None),
                    "crawl4ai_metadata": getattr(crawl_result, 'metadata', {}),
                    "links_found": len(getattr(crawl_result, 'links', {})),
                    "images_found": len(getattr(crawl_result, 'media', {}))
                },
                word_count=len(content.split()) if content else 0
            )
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error converting crawl result for {original_url}: {str(e)}")
            return None

    async def crawl_url(self, url: str) -> Optional[Document]:
        """Crawl a single URL using Crawl4AI."""
        try:
            self.logger.info(f"Crawling URL with Crawl4AI: {url}")
            
            if not self._crawl4ai_crawler:
                raise RuntimeError("Crawler not initialized. Use async context manager.")
            
            # Crawl with Crawl4AI
            result = await self._crawl4ai_crawler.arun(url=url)
            
            # Convert to our Document format
            document = self._convert_to_document(result, url)
            
            if document:
                self.logger.info(f"✅ Successfully crawled {url} - {len(document.content)} chars")
            else:
                self.logger.error(f"❌ Failed to crawl {url}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            return None

    async def crawl_urls(self, urls: List[str]) -> List[Document]:
        """Crawl multiple URLs using Crawl4AI batch processing."""
        try:
            if not urls:
                return []
            
            self.logger.info(f"🚀 Batch crawling {len(urls)} URLs with Crawl4AI")
            
            if not self._crawl4ai_crawler:
                raise RuntimeError("Crawler not initialized. Use async context manager.")
            
            documents = []
            
            # Crawl URLs sequentially for now (can optimize later with arun_many)
            for url in urls:
                try:
                    result = await self._crawl4ai_crawler.arun(url=url)
                    document = self._convert_to_document(result, url)
                    if document:
                        documents.append(document)
                except Exception as e:
                    self.logger.error(f"Failed to crawl {url}: {str(e)}")
                    continue
            
            self.logger.info(f"✅ Successfully crawled {len(documents)}/{len(urls)} URLs")
            return documents
                
        except Exception as e:
            self.logger.error(f"Error in batch crawling: {str(e)}")
            return []
    
    async def arun_many(
        self,
        urls: List[str],
        config: CrawlerRunConfig,
        dispatcher: MemoryAdaptiveDispatcher
    ) -> List[Any]:
        """Run batch crawling - maintaining compatibility with old interface."""
        documents = await self.crawl_urls(urls)
        
        # Convert to old format for compatibility
        results = []
        for i, url in enumerate(urls):
            if i < len(documents) and documents[i]:
                results.append({
                    "url": url,
                    "success": True,
                    "error_message": None
                })
            else:
                results.append({
                    "url": url,
                    "success": False,
                    "error_message": "Failed to crawl"
                })
        
        return results