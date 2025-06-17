"""
Web crawler for the Second Brain AI Assistant.
Implements async web crawling with proper context management and batch processing.
"""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from src.models.schemas import Document, ContentSource, DocumentType, ProcessingStatus
from src.utils.logger import LoggerMixin
from src.feature_pipeline.document_processor import MarkdownConverter

@dataclass
class BrowserConfig:
    """Browser configuration for web crawling."""
    headless: bool = True
    verbose: bool = False

@dataclass
class CrawlerRunConfig:
    """Configuration for crawler runs."""
    cache_mode: str = "BYPASS"
    stream: bool = False

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
    Async web crawler with proper context management and batch processing support.
    """
    
    def __init__(self, config: Optional[BrowserConfig] = None):
        """Initialize the web crawler."""
        self.config = config or BrowserConfig()
        self._session = None
        self.markdown_converter = MarkdownConverter()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _create_session(self):
        """Create a new session for crawling."""
        import aiohttp
        
        # Create session with timeout and retry settings
        timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds total timeout
        connector = aiohttp.TCPConnector(
            limit=10,  # Max concurrent connections
            ttl_dns_cache=300,  # DNS cache TTL
            ssl=False  # Disable SSL verification for testing
        )
        
        return aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
    
    async def crawl_url(self, url: str) -> Optional[Document]:
        """Crawl a single URL."""
        try:
            self.logger.info(f"Crawling URL: {url}")
            
            # Create a new session if none exists
            if not self._session:
                self._session = await self._create_session()
            
            # Fetch the page content
            async with self._session.get(url) as response:
                response.raise_for_status()
                html = await response.text()
            
            # Extract title and content
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get title
            title = soup.title.string if soup.title else url
            
            # Convert HTML to Markdown
            markdown_content = self.markdown_converter.convert(str(soup))
            
            document = Document(
                id=f"web_{datetime.utcnow().timestamp()}",
                title=title,
                content=markdown_content,
                source=ContentSource.WEB_CRAWL,
                source_url=url,
                document_type=DocumentType.WEB_PAGE,
                processing_status=ProcessingStatus.COMPLETED,
                metadata={
                    "crawled_at": datetime.utcnow().isoformat(),
                    "url": url,
                    "content_format": "markdown"
                },
                word_count=len(markdown_content.split())
            )
            
            self.logger.info(f"Successfully crawled {url}")
            return document
            
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            return None
    
    async def crawl_urls(self, urls: List[str]) -> List[Document]:
        """Crawl multiple URLs using batch processing."""
        try:
            browser_config = BrowserConfig(
                headless=self.config.headless,
                verbose=self.config.verbose
            )
            
            run_config = CrawlerRunConfig(
                cache_mode="BYPASS",
                stream=False
            )
            
            dispatcher = MemoryAdaptiveDispatcher(
                memory_threshold_percent=70.0,
                check_interval=1.0,
                max_session_permit=10,
                monitor=CrawlerMonitor()
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun_many(
                    urls=urls,
                    config=run_config,
                    dispatcher=dispatcher
                )
                
                documents = []
                for result in results:
                    if result.success:
                        doc = await crawler.crawl_url(result.url)
                        if doc:
                            documents.append(doc)
                
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
        """Run batch crawling with proper configuration."""
        # Implement batch crawling logic here
        # This is a placeholder implementation
        results = []
        for url in urls:
            try:
                doc = await self.crawl_url(url)
                results.append({
                    "url": url,
                    "success": doc is not None,
                    "error_message": None if doc else "Failed to crawl"
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "success": False,
                    "error_message": str(e)
                })
        return results

class WebCrawler(LoggerMixin):
    """
    Synchronous wrapper around AsyncWebCrawler for backward compatibility.
    """
    
    def __init__(self):
        """Initialize the web crawler."""
        self._crawler = AsyncWebCrawler()
    
    async def crawl_url(self, url: str) -> Optional[Document]:
        """Crawl a single URL."""
        async with self._crawler as crawler:
            return await crawler.crawl_url(url)
    
    async def crawl_urls(self, urls: List[str]) -> List[Document]:
        """Crawl multiple URLs."""
        async with self._crawler as crawler:
            return await crawler.crawl_urls(urls) 