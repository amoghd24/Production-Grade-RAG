"""
Notion API collector for the Second Brain AI Assistant.
Collects documents from Notion workspace and converts them to standardized format.
Follows the DecodingML methodology for data collection.
"""

import asyncio
import re
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.models.schemas import Document, DocumentType, ContentSource, ProcessingStatus
from src.config.settings import settings
from src.utils.logger import LoggerMixin


class NotionCollector(LoggerMixin):
    """
    Collects documents from Notion workspace using the Notion API.
    Converts pages and database entries to standardized Document format.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Notion collector."""
        # Try DecodingML naming first, then fallback
        self.api_key = api_key or settings.NOTION_SECRET_KEY or settings.NOTION_API_KEY
        if not self.api_key:
            raise ValueError("Notion API key is required. Set NOTION_SECRET_KEY or NOTION_API_KEY environment variable.")
        
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(self.headers)
        
        self.collected_urls: Set[str] = set()
    
    def search_pages(self, query: str = "", page_size: int = 100) -> List[Dict[str, Any]]:
        """
        Search for pages in the Notion workspace.
        
        Args:
            query: Search query (empty returns all accessible pages)
            page_size: Number of results per page
            
        Returns:
            List of page objects
        """
        url = f"{self.base_url}/search"
        all_pages = []
        
        payload = {
            "page_size": page_size,
            "filter": {
                "property": "object",
                "value": "page"
            }
        }
        
        if query:
            payload["query"] = query
        
        has_more = True
        next_cursor = None
        
        while has_more:
            if next_cursor:
                payload["start_cursor"] = next_cursor
            
            try:
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                all_pages.extend(data.get("results", []))
                has_more = data.get("has_more", False)
                next_cursor = data.get("next_cursor")
                
                self.logger.info(f"Retrieved {len(data.get('results', []))} pages")
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error searching pages: {str(e)}")
                break
        
        self.logger.info(f"Total pages found: {len(all_pages)}")
        return all_pages
    
    def get_database_entries(self, database_id: str, page_size: int = 100) -> List[Dict[str, Any]]:
        """
        Get all entries from a Notion database.
        
        Args:
            database_id: ID of the Notion database
            page_size: Number of results per page
            
        Returns:
            List of database entry objects
        """
        url = f"{self.base_url}/databases/{database_id}/query"
        all_entries = []
        
        payload = {"page_size": page_size}
        has_more = True
        next_cursor = None
        
        while has_more:
            if next_cursor:
                payload["start_cursor"] = next_cursor
            
            try:
                response = self.session.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                all_entries.extend(data.get("results", []))
                has_more = data.get("has_more", False)
                next_cursor = data.get("next_cursor")
                
                self.logger.info(f"Retrieved {len(data.get('results', []))} database entries")
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error querying database {database_id}: {str(e)}")
                break
        
        self.logger.info(f"Total database entries: {len(all_entries)}")
        return all_entries
    
    def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Get the content of a specific page.
        
        Args:
            page_id: ID of the Notion page
            
        Returns:
            Page content data
        """
        url = f"{self.base_url}/blocks/{page_id}/children"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            self.logger.info(f"Page content data: {data}")  # Add this line to see the raw data
            return data
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting page content for {page_id}: {str(e)}")
            return {}
    
    def extract_text_from_rich_text(self, rich_text: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion rich text objects."""
        if not rich_text:
            return ""
        
        text_parts = []
        for item in rich_text:
            if item.get("type") == "text":
                text = item.get("text", {}).get("content", "")
                # Handle links in text
                if item.get("href"):
                    self.logger.info(f"Found link in rich text: {text} -> {item['href']}")
                    text = f"[{text}]({item['href']})"
                text_parts.append(text)
        
        return "".join(text_parts)
    
    def get_table_rows(self, table_id: str) -> List[Dict[str, Any]]:
        """
        Get the rows of a table block.
        
        Args:
            table_id: ID of the table block
            
        Returns:
            List of table row objects
        """
        url = f"{self.base_url}/blocks/{table_id}/children"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            self.logger.info(f"Table rows data: {data}")
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting table rows for {table_id}: {str(e)}")
            return []
    
    def block_to_markdown(self, block: Dict[str, Any]) -> str:
        """
        Convert a Notion block to Markdown format.
        
        Args:
            block: Notion block object
            
        Returns:
            Markdown representation of the block
        """
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})
        
        # Debug log the block type and data
        self.logger.info(f"Processing block type: {block_type}")
        
        if block_type == "table":
            # Get table rows
            table_id = block.get("id")
            rows = self.get_table_rows(table_id)
            self.logger.info(f"Table rows: {rows}")
            
            # Process table content
            table_content = []
            for row in rows:
                row_content = []
                cells = row.get("table_row", {}).get("cells", [])
                self.logger.info(f"Row cells: {cells}")
                
                for cell in cells:
                    self.logger.info(f"Processing cell content: {cell}")
                    cell_text = self.extract_text_from_rich_text(cell)
                    self.logger.info(f"Extracted cell text: {cell_text}")
                    row_content.append(cell_text)
                
                table_content.append("| " + " | ".join(row_content) + " |")
            
            # Add header separator
            if table_content:
                header = table_content[0]
                separator = "|" + "|".join(["---" for _ in header.split("|")[1:-1]]) + "|"
                table_content.insert(1, separator)
            
            markdown_table = "\n".join(table_content) + "\n\n"
            self.logger.info(f"Generated markdown table: {markdown_table}")
            return markdown_table
        
        elif block_type == "paragraph":
            rich_text = block_data.get("rich_text", [])
            self.logger.info(f"Paragraph rich text: {rich_text}")
            text = self.extract_text_from_rich_text(rich_text)
            return f"{text}\n\n"
        
        elif block_type == "heading_1":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            return f"# {text}\n\n"
        
        elif block_type == "heading_2":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            return f"## {text}\n\n"
        
        elif block_type == "heading_3":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            return f"### {text}\n\n"
        
        elif block_type == "bulleted_list_item":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            return f"- {text}\n"
        
        elif block_type == "numbered_list_item":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            return f"1. {text}\n"
        
        elif block_type == "to_do":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            checked = "x" if block_data.get("checked", False) else " "
            return f"- [{checked}] {text}\n"
        
        elif block_type == "code":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            language = block_data.get("language", "")
            return f"```{language}\n{text}\n```\n\n"
        
        elif block_type == "quote":
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            return f"> {text}\n\n"
        
        elif block_type == "divider":
            return "---\n\n"
        
        else:
            # For unsupported block types, extract any available text
            text = self.extract_text_from_rich_text(block_data.get("rich_text", []))
            return f"{text}\n" if text else ""
    
    def convert_page_to_markdown(self, page_data: Dict[str, Any], content_data: Dict[str, Any]) -> str:
        """
        Convert a Notion page to Markdown format.
        
        Args:
            page_data: Page metadata
            content_data: Page content blocks
            
        Returns:
            Markdown representation of the page
        """
        markdown_parts = []
        
        # Add title
        title = self.get_page_title(page_data)
        if title:
            markdown_parts.append(f"# {title}\n\n")
        
        # Convert content blocks
        blocks = content_data.get("results", [])
        for block in blocks:
            markdown_parts.append(self.block_to_markdown(block))
        
        return "".join(markdown_parts).strip()
    
    def get_page_title(self, page_data: Dict[str, Any]) -> str:
        """Extract title from page data."""
        properties = page_data.get("properties", {})
        
        # Look for title property
        for prop_name, prop_data in properties.items():
            if prop_data.get("type") == "title":
                title_parts = prop_data.get("title", [])
                return self.extract_text_from_rich_text(title_parts)
        
        # Fallback to page URL or ID
        return page_data.get("url", page_data.get("id", "Untitled"))
    
    def extract_links_from_content(self, markdown_content: str) -> List[str]:
        """
        Extract all URLs from markdown content.
        
        Args:
            markdown_content: Markdown text content
            
        Returns:
            List of URLs found in the content
        """
        # Debug log the content being processed
        self.logger.info(f"Extracting links from content: {markdown_content[:200]}...")
        
        # Pattern to match markdown links [text](url) and plain URLs
        link_patterns = [
            r'\[([^\]]+)\]\(([^)]+)\)',  # Markdown links
            r'https?://[^\s\)]+',        # Plain URLs
        ]
        
        urls = []
        for pattern in link_patterns:
            matches = re.findall(pattern, markdown_content)
            if pattern.startswith(r'\['):
                # Extract URLs from markdown link tuples
                urls.extend([match[1] for match in matches])
            else:
                # Direct URL matches
                urls.extend(matches)
        
        # Debug log found URLs
        self.logger.info(f"Found URLs: {urls}")
        
        # Clean and validate URLs
        clean_urls = []
        for url in urls:
            # Remove any trailing punctuation or brackets
            url = url.strip('.,;:!?)]')
            # Remove any markdown link text if present
            if '](' in url:
                url = url.split('](')[-1]
            if url and self.is_valid_url(url):
                clean_urls.append(url)
        
        # Debug log cleaned URLs
        self.logger.info(f"Cleaned URLs: {clean_urls}")
        
        return list(set(clean_urls))  # Remove duplicates
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not a Notion internal link."""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in ['http', 'https'] and
                parsed.netloc and
                'notion.so' not in parsed.netloc  # Skip Notion internal links
            )
        except Exception:
            return False
    
    def page_to_document(self, page_data: Dict[str, Any]) -> Optional[Document]:
        """
        Convert a Notion page to a Document object.
        
        Args:
            page_data: Notion page data
            
        Returns:
            Document object or None if conversion failed
        """
        try:
            page_id = page_data.get("id", "")
            if not page_id:
                return None
            
            # Get page content
            content_data = self.get_page_content(page_id)
            if not content_data:
                return None
            
            # Convert to markdown
            markdown_content = self.convert_page_to_markdown(page_data, content_data)
            if not markdown_content.strip():
                return None
            
            # Extract metadata
            title = self.get_page_title(page_data)
            page_url = page_data.get("url", "")
            
            # Extract embedded links for crawling
            embedded_links = self.extract_links_from_content(markdown_content)
            
            # Create document
            document = Document(
                id=f"notion_{page_id}",
                title=title,
                content=markdown_content,
                source=ContentSource.NOTION,
                source_url=page_url,
                document_type=DocumentType.NOTION_PAGE,
                processing_status=ProcessingStatus.COMPLETED,
                metadata={
                    "notion_id": page_id,
                    "notion_url": page_url,
                    "embedded_links": embedded_links,
                    "created_time": page_data.get("created_time", ""),
                    "last_edited_time": page_data.get("last_edited_time", ""),
                    "collection_timestamp": datetime.utcnow().isoformat(),
                    "properties": page_data.get("properties", {}),
                },
                word_count=len(markdown_content.split())
            )
            
            # Store links for web crawling
            self.collected_urls.update(embedded_links)
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error converting page to document: {str(e)}")
            return None
    
    def collect_workspace_documents(
        self, 
        search_query: str = "",
        database_id: Optional[str] = None
    ) -> List[Document]:
        """
        Collect all documents from the Notion workspace.
        
        Args:
            search_query: Optional search query to filter pages
            database_id: Optional specific database ID to query
            
        Returns:
            List of Document objects
        """
        documents = []
        
        try:
            if database_id:
                # Query specific database
                self.logger.info(f"Collecting from database: {database_id}")
                pages = self.get_database_entries(database_id)
            else:
                # Search all accessible pages
                self.logger.info("Collecting from all accessible pages")
                pages = self.search_pages(search_query)
            
            self.logger.info(f"Processing {len(pages)} pages...")
            
            for i, page_data in enumerate(pages):
                document = self.page_to_document(page_data)
                if document:
                    documents.append(document)
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(pages)} pages")
            
            self.logger.info(f"Successfully collected {len(documents)} documents")
            self.logger.info(f"Found {len(self.collected_urls)} embedded URLs for crawling")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error collecting workspace documents: {str(e)}")
            return []
    
    def get_embedded_urls(self) -> List[str]:
        """Get all URLs found in collected documents for web crawling."""
        return list(self.collected_urls)


# Async wrapper for integration with existing pipeline
async def collect_notion_documents(
    api_key: Optional[str] = None,
    search_query: str = "",
    database_id: Optional[str] = None
) -> tuple[List[Document], List[str]]:
    """
    Async wrapper for collecting Notion documents.
    
    Args:
        api_key: Notion API key
        search_query: Search query for filtering
        database_id: Specific database ID to query
        
    Returns:
        Tuple of (documents, embedded_urls)
    """
    collector = NotionCollector(api_key)
    documents = collector.collect_workspace_documents(search_query, database_id)
    embedded_urls = collector.get_embedded_urls()
    
    return documents, embedded_urls
