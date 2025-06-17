"""
Document processing pipeline for the Second Brain AI Assistant.
Follows the DecodingML approach: chunk documents, compute quality scores,
and prepare them for MongoDB vector storage.
"""

import asyncio
from typing import List, Dict, Optional, Any
import re
from datetime import datetime
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.models.schemas import Document, DocumentChunk, ProcessingStatus
from src.config.settings import settings
from src.utils.logger import LoggerMixin

class ContentCleaner(LoggerMixin):
    """Cleans and normalizes document content."""
    def clean(self, content: str) -> str:
        if not content:
            return ""
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'Cookie Policy|Privacy Policy|Terms of Service', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Subscribe to.*newsletter', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Follow us on.*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'&\w+;', '', content)
        self.logger.debug("Content cleaned.")
        return content.strip()

class QualityScorer(LoggerMixin):
    """Computes quality score for a document."""
    def score(self, document: Document) -> float:
        if not document.content:
            return 0.0
        score = 0.0
        word_count = len(document.content.split())
        if word_count > 100:
            length_score = min(1.0, word_count / 1000)
            score += 0.2 * length_score
        structure_score = 0.0
        if document.title and len(document.title.strip()) > 5:
            structure_score += 0.3
        header_count = len(re.findall(r'^#{1,6}\s+.+', document.content, re.MULTILINE))
        if header_count > 0:
            structure_score += 0.4
        list_count = len(re.findall(r'^\s*[-*+]\s+.+', document.content, re.MULTILINE))
        if list_count > 0:
            structure_score += 0.3
        score += 0.15 * min(1.0, structure_score)
        structure_elements = len(re.findall(r'^#{1,6}\s+.+|^\s*[-*+]\s+.+|^\s*\d+\.\s+.+', document.content, re.MULTILINE))
        richness_score = min(1.0, structure_elements / 10)
        score += 0.25 * richness_score
        sentences = re.split(r'[.!?]+', document.content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            readability_score = max(0, 1 - abs(avg_sentence_length - 17) / 17)
            score += 0.2 * readability_score
        credibility_score = 0.0
        if document.source_url:
            url_str = str(document.source_url).lower()
            credible_domains = [
                'yahoo', 'org', 'realmadrid', 'psg', 'inter', 'porto', 'benfica',
                'apple', 'ai', 'nvidia', 'google', 'microsoft', 'amazon', 'meta',
            ]
            if any(domain in url_str for domain in credible_domains):
                credibility_score = 0.8
            elif any(domain in url_str for domain in ['com', 'net']):
                credibility_score = 0.6
            else:
                credibility_score = 0.4
        else:
            credibility_score = 0.5
        score += 0.2 * credibility_score
        final_score = max(0.0, min(1.0, score))
        self.logger.debug(f"Quality score for '{document.title}': {final_score:.3f}")
        return final_score

class DocumentChunker(LoggerMixin):
    """Splits documents into chunks for vector storage."""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.cleaner = ContentCleaner()
    def chunk(self, document: Document) -> List[DocumentChunk]:
        if not document.content:
            return []
        cleaned_content = self.cleaner.clean(document.content)
        chunk_texts = self.text_splitter.split_text(cleaned_content)
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            if len(chunk_text.strip()) < 50:
                continue
            chunk_id = hashlib.md5(
                f"{document.id}_{idx}_{chunk_text[:100]}".encode()
            ).hexdigest()
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document.id or "",
                content=chunk_text.strip(),
                chunk_index=idx,
                word_count=len(chunk_text.split()),
                start_char=0,
                end_char=len(chunk_text)
            )
            chunks.append(chunk)
        self.logger.info(f"Created {len(chunks)} chunks for document '{document.title}'")
        return chunks

class EmbeddingGenerator(LoggerMixin):
    """Generates embeddings for document chunks."""
    def __init__(self):
        self.embedding_model = None
        self.model_name = settings.EMBEDDING_MODEL
    async def initialize(self):
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
    async def generate(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        if not chunks:
            return chunks
        await self.initialize()
        texts = [chunk.content for chunk in chunks]
        self.logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
            chunk.embedding_model = self.model_name
        self.logger.info("Embeddings generated successfully")
        return chunks

class MarkdownConverter(LoggerMixin):
    """Converts HTML content to Markdown format."""
    
    def __init__(self):
        try:
            import html2text
            self.converter = html2text.HTML2Text()
            self.converter.ignore_links = False
            self.converter.ignore_images = False
            self.converter.ignore_emphasis = False
            self.converter.body_width = 0  # No wrapping
        except ImportError:
            self.logger.error("html2text package not found. Please install it using: pip install html2text")
            raise

    def convert(self, html_content: str) -> str:
        """Convert HTML content to Markdown format."""
        try:
            markdown_content = self.converter.handle(html_content)
            return markdown_content.strip()
        except Exception as e:
            self.logger.error(f"Error converting HTML to Markdown: {str(e)}")
            return html_content  # Return original content if conversion fails
