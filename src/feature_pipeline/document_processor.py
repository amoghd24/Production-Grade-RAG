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


class DocumentProcessor(LoggerMixin):
    """
    Process documents following the DecodingML methodology:
    1. Clean and normalize content
    2. Compute quality scores using LLMs
    3. Chunk documents for vector storage
    4. Generate embeddings
    5. Prepare for MongoDB vector index
    """
    
    def __init__(self):
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embedding model
        self.embedding_model = None
        self.model_name = settings.EMBEDDING_MODEL
        
    async def initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
    
    def clean_content(self, content: str) -> str:
        """
        Clean and normalize document content.
        
        Args:
            content: Raw document content
            
        Returns:
            Cleaned content string
        """
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Remove common artifacts from web scraping
        content = re.sub(r'Cookie Policy|Privacy Policy|Terms of Service', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Subscribe to.*newsletter', '', content, flags=re.IGNORECASE)
        content = re.sub(r'Follow us on.*', '', content, flags=re.IGNORECASE)
        
        # Remove HTML artifacts that might remain
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'&\w+;', '', content)
        
        return content.strip()
    
    def compute_quality_score(self, document: Document) -> float:
        """
        Compute quality score for a document using heuristics and LLMs.
        Following DecodingML methodology.
        
        Args:
            document: Document to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not document.content:
            return 0.0
        
        score = 0.0
        
        # Content length score (20% weight)
        word_count = len(document.content.split())
        if word_count > 100:
            length_score = min(1.0, word_count / 1000)  # Normalize to 1000 words
            score += 0.2 * length_score
        
        # Structure score (15% weight)
        structure_score = 0.0
        if document.title and len(document.title.strip()) > 5:
            structure_score += 0.3
        
        # Check for headers (markdown style)
        header_count = len(re.findall(r'^#{1,6}\s+.+', document.content, re.MULTILINE))
        if header_count > 0:
            structure_score += 0.4
        
        # Check for lists
        list_count = len(re.findall(r'^\s*[-*+]\s+.+', document.content, re.MULTILINE))
        if list_count > 0:
            structure_score += 0.3
        
        score += 0.15 * min(1.0, structure_score)
        
        # Content richness score (25% weight)
        richness_score = 0.0
        
        # Check for content structure and formatting
        # Headers, lists, and paragraphs indicate well-structured content
        structure_elements = len(re.findall(r'^#{1,6}\s+.+|^\s*[-*+]\s+.+|^\s*\d+\.\s+.+', document.content, re.MULTILINE))
        richness_score = min(1.0, structure_elements / 10)  # Normalize to 10 elements
        
        score += 0.25 * richness_score
        
        # Readability score (20% weight)
        sentences = re.split(r'[.!?]+', document.content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Ideal sentence length is around 15-20 words
            readability_score = max(0, 1 - abs(avg_sentence_length - 17) / 17)
            score += 0.2 * readability_score
        
        # Source credibility score (20% weight)
        credibility_score = 0.0
        if document.source_url:
            url_str = str(document.source_url).lower()
            # Higher score for educational/technical domains
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
            credibility_score = 0.5  # Neutral for uploaded content
        
        score += 0.2 * credibility_score
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, score))
        
        self.logger.debug(f"Quality score for '{document.title}': {final_score:.3f}")
        return final_score
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Split document into chunks for vector storage.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        if not document.content:
            return []
        
        # Clean content before chunking
        cleaned_content = self.clean_content(document.content)
        
        # Split into chunks
        chunk_texts = self.text_splitter.split_text(cleaned_content)
        
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            if len(chunk_text.strip()) < 50:  # Skip very small chunks
                continue
            
            # Generate chunk ID
            chunk_id = hashlib.md5(
                f"{document.id}_{idx}_{chunk_text[:100]}".encode()
            ).hexdigest()
            
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document.id or "",
                content=chunk_text.strip(),
                chunk_index=idx,
                word_count=len(chunk_text.split()),
                start_char=0,  # Would need more complex logic for exact positions
                end_char=len(chunk_text)
            )
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks for document '{document.title}'")
        return chunks
    
    async def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            return chunks
        
        await self.initialize_embedding_model()
        
        # Extract texts for batch processing
        texts = [chunk.content for chunk in chunks]
        
        self.logger.info(f"Generating embeddings for {len(texts)} chunks")
        
        # Generate embeddings in batch
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
            chunk.embedding_model = self.model_name
        
        self.logger.info("Embeddings generated successfully")
        return chunks
    
    async def process_document(self, document: Document) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Args:
            document: Document to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            self.logger.info(f"Processing document: {document.title}")
            
            # Step 1: Compute quality score
            quality_score = self.compute_quality_score(document)
            document.quality_score = quality_score
            
            # Step 2: Check if document meets quality threshold
            min_quality_threshold = 0.3  # Configurable threshold
            if quality_score < min_quality_threshold:
                self.logger.warning(
                    f"Document '{document.title}' below quality threshold: {quality_score:.3f}"
                )
                document.processing_status = ProcessingStatus.COMPLETED
                return {
                    "document": document,
                    "chunks": [],
                    "quality_score": quality_score,
                    "processed": False,
                    "reason": "Below quality threshold"
                }
            
            # Step 3: Chunk document
            chunks = self.chunk_document(document)
            
            if not chunks:
                self.logger.warning(f"No valid chunks created for document '{document.title}'")
                document.processing_status = ProcessingStatus.COMPLETED
                return {
                    "document": document,
                    "chunks": [],
                    "quality_score": quality_score,
                    "processed": False,
                    "reason": "No valid chunks"
                }
            
            # Step 4: Generate embeddings
            chunks_with_embeddings = await self.generate_embeddings(chunks)
            
            # Step 5: Update document status
            document.processing_status = ProcessingStatus.COMPLETED
            document.updated_at = datetime.utcnow()
            
            result = {
                "document": document,
                "chunks": chunks_with_embeddings,
                "quality_score": quality_score,
                "processed": True,
                "chunk_count": len(chunks_with_embeddings)
            }
            
            self.logger.info(
                f"Successfully processed document '{document.title}' -> "
                f"{len(chunks_with_embeddings)} chunks, quality: {quality_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document '{document.title}': {str(e)}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            
            return {
                "document": document,
                "chunks": [],
                "quality_score": 0.0,
                "processed": False,
                "error": str(e)
            }
    
    async def process_documents_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Processing batch of {len(documents)} documents")
        
        results = []
        for document in documents:
            result = await self.process_document(document)
            results.append(result)
        
        successful = sum(1 for r in results if r["processed"])
        self.logger.info(f"Batch processing completed: {successful}/{len(documents)} successful")
        
        return results
