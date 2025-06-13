"""
Web crawler module for the Second Brain AI Assistant.
Uses Crawl4AI to extract and process web content.
"""

import asyncio
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime

from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from src.models.schemas import Document, DocumentType, ContentSource, CrawlJob, ProcessingStatus
from src.config.settings import settings


class WebCrawler:
    """
    Web crawler for collecting content from web pages.
    Uses Crawl4AI for robust content extraction.
    """
    
    def __init__(self):
        """Initialize the web crawler."""
        self.crawler = None
    
    async def __aenter__(self):
        """Initialize the crawler when entering the context."""
        self.crawler = AsyncWebCrawler()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the context."""
        if self.crawler:
            await self.crawler.aclose()
    
    async def crawl_url(self, url: str) -> Optional[Document]:
        """
        Crawl a URL and extract its content.
        
        Args:
            url: The URL to crawl
            
        Returns:
            Document object containing the crawled content, or None if crawling failed
        """
        try:
            if not self.crawler:
                self.crawler = AsyncWebCrawler()
            result = await self.crawler.arun(url=url)
            if not result or not getattr(result, "markdown", None):
                print(f"No content found at {url}")
                return None
            return Document(
                title=getattr(result, "title", url),
                content=result.markdown,
                source=ContentSource.WEB_CRAWL,
                source_url=url,
                document_type=DocumentType.WEB_PAGE,
                metadata={
                    "crawl_date": datetime.now().isoformat()
                }
            )
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            return None

    async def crawl_urls(self, urls: List[str]) -> List[Document]:
        """Crawl multiple URLs and return a list of Document objects."""
        documents = []
        
        async with self:
            for url in urls:
                doc = await self.crawl_url(url)
                if doc:
                    documents.append(doc)
        
        return documents
    
    def extract_links(self, content: str, base_url: str) -> List[str]:
        """
        Extract links from HTML content.
        
        Args:
            content: HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        # Simple regex-based link extraction
        link_pattern = r'href=[\'"]([^\'"]+)[\'"]'
        matches = re.findall(link_pattern, content, re.IGNORECASE)
        
        absolute_links = []
        for link in matches:
            # Skip anchor links, mailto, etc.
            if link.startswith(('#', 'mailto:', 'tel:', 'javascript:')):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(base_url, link)
            absolute_links.append(absolute_url)
        
        return list(set(absolute_links))  # Remove duplicates
    
    def filter_urls_by_domain(self, urls: List[str], allowed_domains: List[str]) -> List[str]:
        """
        Filter URLs to only include allowed domains.
        
        Args:
            urls: List of URLs to filter
            allowed_domains: List of allowed domain names
            
        Returns:
            Filtered list of URLs
        """
        if not allowed_domains:
            return urls
        
        filtered_urls = []
        for url in urls:
            domain = urlparse(url).netloc.lower()
            if any(allowed_domain.lower() in domain for allowed_domain in allowed_domains):
                filtered_urls.append(url)
        
        return filtered_urls
    
    async def crawl_website(
        self,
        start_url: str,
        max_pages: int = 10,
        allowed_domains: Optional[List[str]] = None,
        depth_limit: int = 3
    ) -> CrawlJob:
        """
        Crawl an entire website starting from a URL.
        
        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            allowed_domains: List of allowed domains
            depth_limit: Maximum crawl depth
            
        Returns:
            CrawlJob with results
        """
        crawl_job = CrawlJob(
            start_url=start_url,
            max_pages=max_pages,
            allowed_domains=allowed_domains or [],
            status=ProcessingStatus.PROCESSING,
            started_at=datetime.utcnow()
        )
        
        try:
            print(f"Starting website crawl from: {start_url}")
            
            urls_to_crawl = [start_url]
            crawled_urls = set()
            documents = []
            
            for depth in range(depth_limit):
                if not urls_to_crawl or len(crawled_urls) >= max_pages:
                    break
                
                print(f"Crawling depth {depth + 1}, {len(urls_to_crawl)} URLs")
                
                # Crawl current batch
                batch_documents = await self.crawl_urls(urls_to_crawl[:max_pages - len(crawled_urls)])
                documents.extend(batch_documents)
                
                # Update crawled URLs
                for url in urls_to_crawl:
                    crawled_urls.add(url)
                    crawl_job.crawled_urls.append(url)
                
                # Extract new URLs for next depth
                next_urls = set()
                for doc in batch_documents:
                    if doc.content:
                        links = self.extract_links(doc.content, doc.url)
                        filtered_links = self.filter_urls_by_domain(links, allowed_domains)
                        next_urls.update(filtered_links)
                
                # Remove already crawled URLs
                urls_to_crawl = list(next_urls - crawled_urls)
                
                # Update progress
                crawl_job.pages_crawled = len(crawled_urls)
                crawl_job.pages_processed = len(documents)
            
            # Update final status
            crawl_job.status = ProcessingStatus.COMPLETED
            crawl_job.completed_at = datetime.utcnow()
            crawl_job.pages_crawled = len(crawled_urls)
            crawl_job.pages_processed = len(documents)
            
            print(f"Website crawl completed. Crawled {len(documents)} pages successfully")
            
        except Exception as e:
            print(f"Website crawl failed: {str(e)}")
            crawl_job.status = ProcessingStatus.FAILED
            crawl_job.completed_at = datetime.utcnow()
        
        return crawl_job 