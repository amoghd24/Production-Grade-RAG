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
from src.utils.logger import LoggerMixin


class WebCrawler(LoggerMixin):
    """
    Web crawler for collecting content from web pages.
    Uses Crawl4AI for robust content extraction.
    """
    
    def __init__(self):
        """Initialize the web crawler."""
        self.visited_urls: Set[str] = set()
    
    async def crawl_url(self, url: str) -> Optional[Document]:
        """
        Crawl a single URL and extract content.
        Args:
            url: URL to crawl
        Returns:
            Document object or None if crawling failed
        """
        try:
            self.logger.info(f"Crawling URL: {url}")
            async with AsyncWebCrawler(
                verbose=True,
                headless=True,
                user_agent=settings.USER_AGENT
            ) as crawler:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=50,  # Minimum words for content to be considered
                    extraction_strategy=LLMExtractionStrategy(
                        provider="openai/gpt-4o-mini",
                        api_token=settings.OPENAI_API_KEY,
                        instruction="Extract the main content, remove navigation, ads, and irrelevant elements."
                    )
                )
            if result.success:
                # Extract metadata
                metadata = {
                    "url": url,
                    "title": result.metadata.get("title", ""),
                    "description": result.metadata.get("description", ""),
                    "keywords": result.metadata.get("keywords", []),
                    "author": result.metadata.get("author", ""),
                    "published_date": result.metadata.get("published_date", ""),
                    "word_count": len(result.cleaned_html.split()) if result.cleaned_html else 0,
                    "crawl_timestamp": datetime.utcnow().isoformat(),
                    "status_code": result.status_code,
                }
                document = Document(
                    title=result.metadata.get("title", urlparse(url).path),
                    content=result.markdown or result.cleaned_html or "",
                    source=ContentSource.WEB_CRAWL,
                    source_url=url,
                    document_type=DocumentType.WEB_PAGE,
                    metadata=metadata,
                    word_count=metadata["word_count"],
                    processing_status=ProcessingStatus.COMPLETED
                )
                self.visited_urls.add(url)
                self.logger.info(f"Successfully crawled: {url}")
                return document
            else:
                self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                return None
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            return None

    async def crawl_multiple_urls(self, urls: List[str]) -> List[Document]:
        """
        Crawl multiple URLs concurrently.
        Args:
            urls: List of URLs to crawl
        Returns:
            List of successfully crawled documents
        """
        self.logger.info(f"Starting to crawl {len(urls)} URLs")
        documents = []
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
        async with AsyncWebCrawler(
            verbose=True,
            headless=True,
            user_agent=settings.USER_AGENT
        ) as crawler:
            async def bounded_crawl(url):
                async with semaphore:
                    try:
                        result = await crawler.arun(
                            url=url,
                            word_count_threshold=50,
                            extraction_strategy=LLMExtractionStrategy(
                                provider="openai/gpt-4o-mini",
                                api_token=settings.OPENAI_API_KEY,
                                instruction="Extract the main content, remove navigation, ads, and irrelevant elements."
                            )
                        )
                        if result.success:
                            metadata = {
                                "url": url,
                                "title": result.metadata.get("title", ""),
                                "description": result.metadata.get("description", ""),
                                "keywords": result.metadata.get("keywords", []),
                                "author": result.metadata.get("author", ""),
                                "published_date": result.metadata.get("published_date", ""),
                                "word_count": len(result.cleaned_html.split()) if result.cleaned_html else 0,
                                "crawl_timestamp": datetime.utcnow().isoformat(),
                                "status_code": result.status_code,
                            }
                            document = Document(
                                title=result.metadata.get("title", urlparse(url).path),
                                content=result.markdown or result.cleaned_html or "",
                                source=ContentSource.WEB_CRAWL,
                                source_url=url,
                                document_type=DocumentType.WEB_PAGE,
                                metadata=metadata,
                                word_count=metadata["word_count"],
                                processing_status=ProcessingStatus.COMPLETED
                            )
                            self.visited_urls.add(url)
                            self.logger.info(f"Successfully crawled: {url}")
                            return document
                        else:
                            self.logger.error(f"Failed to crawl {url}: {result.error_message}")
                            return None
                    except Exception as e:
                        self.logger.error(f"Error crawling {url}: {str(e)}")
                        return None
            tasks = [bounded_crawl(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Document):
                    documents.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Error crawling {urls[i]}: {str(result)}")
        self.logger.info(f"Successfully crawled {len(documents)} out of {len(urls)} URLs")
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
            self.logger.info(f"Starting website crawl from: {start_url}")
            
            urls_to_crawl = [start_url]
            crawled_urls = set()
            documents = []
            
            for depth in range(depth_limit):
                if not urls_to_crawl or len(crawled_urls) >= max_pages:
                    break
                
                self.logger.info(f"Crawling depth {depth + 1}, {len(urls_to_crawl)} URLs")
                
                # Crawl current batch
                batch_documents = await self.crawl_multiple_urls(urls_to_crawl[:max_pages - len(crawled_urls)])
                documents.extend(batch_documents)
                
                # Update crawled URLs
                for url in urls_to_crawl:
                    crawled_urls.add(url)
                    crawl_job.crawled_urls.append(url)
                
                # Extract new URLs for next depth
                next_urls = set()
                for doc in batch_documents:
                    if doc.content:
                        links = self.extract_links(doc.content, doc.source_url)
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
            
            self.logger.info(f"Website crawl completed. Crawled {len(documents)} pages successfully")
            
        except Exception as e:
            self.logger.error(f"Website crawl failed: {str(e)}")
            crawl_job.status = ProcessingStatus.FAILED
            crawl_job.completed_at = datetime.utcnow()
        
        return crawl_job 