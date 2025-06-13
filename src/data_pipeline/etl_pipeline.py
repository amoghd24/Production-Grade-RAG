"""
Complete ETL Pipeline for Second Brain AI Assistant.
Implements Lesson 2 of DecodingML course: Data collection → Processing → Storage
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from src.data_pipeline.integrated_collector import IntegratedDataCollector
from src.feature_pipeline.document_processor import DocumentProcessor
from src.feature_pipeline.vector_storage import create_vector_store
from src.models.schemas import Document, ProcessingStatus
from src.config.settings import settings
from src.utils.logger import LoggerMixin


class SecondBrainETLPipeline(LoggerMixin):
    """
    Complete ETL Pipeline following DecodingML methodology:
    1. Extract: Collect from Notion + Crawl embedded links
    2. Transform: Clean, chunk, compute quality scores, generate embeddings
    3. Load: Store to MongoDB vector database
    """
    
    def __init__(self):
        """Initialize the ETL pipeline."""
        self.collector = IntegratedDataCollector()
        self.processor = DocumentProcessor()
        self.vector_store = None
        
        # Pipeline statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_documents_collected": 0,
            "documents_processed": 0,
            "documents_stored": 0,
            "total_chunks": 0,
            "errors": []
        }
    
    async def run_complete_pipeline(
        self,
        notion_api_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,
        max_crawl_pages: int = 100,
        quality_threshold: float = 0.5,
        store_to_mongodb: bool = True,
        save_local_backup: bool = True
    ) -> Dict[str, any]:
        """
        Run the complete ETL pipeline.
        
        Args:
            notion_api_key: Notion API key
            notion_database_id: Specific database to collect from
            max_crawl_pages: Maximum pages to crawl
            quality_threshold: Minimum quality score for storage
            store_to_mongodb: Whether to store in MongoDB vector database
            save_local_backup: Whether to save local JSON backup
            
        Returns:
            Pipeline execution statistics
        """
        self.stats["start_time"] = datetime.utcnow()
        
        try:
            self.logger.info("🚀 Starting Second Brain ETL Pipeline...")
            
            # Step 1: Extract - Data Collection
            documents = await self._extract_data(
                notion_api_key=notion_api_key,
                notion_database_id=notion_database_id,
                max_crawl_pages=max_crawl_pages
            )
            
            # Step 2: Transform - Process Documents
            processed_documents = await self._transform_data(
                documents=documents,
                quality_threshold=quality_threshold
            )
            
            # Step 3: Load - Store Data
            if store_to_mongodb or save_local_backup:
                await self._load_data(
                    documents=processed_documents,
                    store_to_mongodb=store_to_mongodb,
                    save_local_backup=save_local_backup
                )
            
            self.stats["end_time"] = datetime.utcnow()
            self.stats["duration"] = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            
            self.logger.info("✅ ETL Pipeline completed successfully!")
            self._log_pipeline_summary()
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"❌ ETL Pipeline failed: {str(e)}")
            self.stats["errors"].append(str(e))
            self.stats["end_time"] = datetime.utcnow()
            raise
    
    async def _extract_data(
        self,
        notion_api_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,
        max_crawl_pages: int = 100
    ) -> List[Document]:
        """Extract data from Notion and web crawling."""
        self.logger.info("📥 EXTRACT: Starting data collection...")
        
        try:
            documents = await self.collector.collect_all_data(
                notion_api_key=notion_api_key,
                notion_database_id=notion_database_id,
                max_crawl_pages=max_crawl_pages,
                crawl_embedded_links=True
            )
            
            self.stats["total_documents_collected"] = len(documents)
            self.logger.info(f"✅ Collected {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ Data extraction failed: {str(e)}")
            self.stats["errors"].append(f"Extraction error: {str(e)}")
            return []
    
    async def _transform_data(
        self,
        documents: List[Document],
        quality_threshold: float = 0.5
    ) -> List[Dict[str, any]]:
        """Transform documents: clean, chunk, score, embed."""
        self.logger.info("🔄 TRANSFORM: Processing documents...")
        
        processed_documents = []
        
        for i, document in enumerate(documents):
            try:
                # Process document using existing processor
                processed_data = await self.processor.process_document(document)
                
                # Apply quality filter
                quality_score = processed_data.get("quality_score", 0.0)
                if quality_score >= quality_threshold:
                    processed_documents.append(processed_data)
                    self.stats["documents_processed"] += 1
                    self.stats["total_chunks"] += len(processed_data.get("chunks", []))
                else:
                    self.logger.debug(f"Filtered out document '{document.title}' (quality: {quality_score:.3f})")
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                self.logger.error(f"Error processing document '{document.title}': {str(e)}")
                self.stats["errors"].append(f"Processing error for '{document.title}': {str(e)}")
        
        self.logger.info(f"✅ Processed {len(processed_documents)} documents (passed quality filter)")
        return processed_documents
    
    async def _load_data(
        self,
        documents: List[Dict[str, any]],
        store_to_mongodb: bool = True,
        save_local_backup: bool = True
    ) -> None:
        """Load processed data to storage systems."""
        self.logger.info("💾 LOAD: Storing processed data...")
        
        # Save local backup
        if save_local_backup:
            await self._save_local_backup(documents)
        
        # Store to MongoDB vector database
        if store_to_mongodb:
            await self._store_to_mongodb(documents)
    
    async def _save_local_backup(self, documents: List[Dict[str, any]]) -> None:
        """Save documents as local JSON backup."""
        try:
            # Ensure data directory exists
            data_dir = settings.DATA_DIR / "processed"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup file with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = data_dir / f"second_brain_backup_{timestamp}.json"
            
            # Prepare serializable data
            backup_data = {
                "metadata": {
                    "backup_timestamp": datetime.utcnow().isoformat(),
                    "total_documents": len(documents),
                    "total_chunks": sum(len(doc.get("chunks", [])) for doc in documents),
                    "pipeline_stats": self.stats
                },
                "documents": []
            }
            
            # Serialize documents
            for doc_data in documents:
                doc_dict = doc_data["document"].dict() if hasattr(doc_data["document"], 'dict') else doc_data["document"]
                chunks_dict = [chunk.dict() if hasattr(chunk, 'dict') else chunk for chunk in doc_data.get("chunks", [])]
                
                backup_data["documents"].append({
                    "document": doc_dict,
                    "chunks": chunks_dict,
                    "quality_score": doc_data.get("quality_score", 0.0)
                })
            
            # Write to file
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 Local backup saved: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save local backup: {str(e)}")
            self.stats["errors"].append(f"Backup error: {str(e)}")
    
    async def _store_to_mongodb(self, documents: List[Dict[str, any]]) -> None:
        """Store documents and chunks to MongoDB vector database."""
        try:
            # Initialize vector store
            self.vector_store = await create_vector_store(
                initialize=True,
                setup_indexes=True
            )
            
            # Prepare documents and chunks for storage
            all_documents = []
            all_chunks = []
            
            for doc_data in documents:
                document = doc_data["document"]
                chunks = doc_data.get("chunks", [])
                
                # Update document with quality score
                document.quality_score = doc_data.get("quality_score", 0.0)
                document.processing_status = ProcessingStatus.COMPLETED
                
                all_documents.append(document)
                all_chunks.extend(chunks)
            
            # Store in MongoDB
            if all_documents and all_chunks:
                storage_result = await self.vector_store.store_documents_with_embeddings(
                    documents=all_documents,
                    chunks=all_chunks
                )
                
                self.stats["documents_stored"] = len(storage_result.get("document_ids", []))
                self.logger.info(f"✅ Stored {self.stats['documents_stored']} documents to MongoDB")
            else:
                self.logger.warning("⚠️ No documents to store in MongoDB")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store to MongoDB: {str(e)}")
            self.stats["errors"].append(f"MongoDB storage error: {str(e)}")
        finally:
            # Clean up vector store connection
            if self.vector_store:
                try:
                    await self.vector_store.close()
                except:
                    pass
    
    def _log_pipeline_summary(self) -> None:
        """Log pipeline execution summary."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 SECOND BRAIN ETL PIPELINE SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Duration: {self.stats.get('duration', 0):.2f} seconds")
        self.logger.info(f"Documents collected: {self.stats['total_documents_collected']}")
        self.logger.info(f"Documents processed: {self.stats['documents_processed']}")
        self.logger.info(f"Documents stored: {self.stats['documents_stored']}")
        self.logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        self.logger.info(f"Errors encountered: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            self.logger.info("\n❌ Errors:")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                self.logger.info(f"   {error}")
        
        self.logger.info("="*60)


# Convenience functions
async def run_second_brain_etl(
    notion_api_key: Optional[str] = None,
    notion_database_id: Optional[str] = None,
    max_crawl_pages: int = 100,
    quality_threshold: float = 0.5
) -> Dict[str, any]:
    """
    Run the complete Second Brain ETL pipeline.
    
    Args:
        notion_api_key: Notion API key
        notion_database_id: Specific database to collect from
        max_crawl_pages: Maximum pages to crawl
        quality_threshold: Minimum quality score for storage
        
    Returns:
        Pipeline execution statistics
    """
    pipeline = SecondBrainETLPipeline()
    
    return await pipeline.run_complete_pipeline(
        notion_api_key=notion_api_key,
        notion_database_id=notion_database_id,
        max_crawl_pages=max_crawl_pages,
        quality_threshold=quality_threshold,
        store_to_mongodb=True,
        save_local_backup=True
    )


# CLI for running the complete pipeline
async def main():
    """Run the complete ETL pipeline from command line."""
    import os
    
    print("🧠 Second Brain AI Assistant - ETL Pipeline")
    print("="*50)
    
    # Get configuration
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_database_id = os.getenv("NOTION_DATABASE_ID")
    
    if not notion_api_key:
        print("❌ NOTION_API_KEY environment variable not set!")
        print("📖 Run: python setup_notion.py")
        return
    
    print(f"🔑 Using API key: {notion_api_key[:20]}...")
    if notion_database_id:
        print(f"📊 Target database: {notion_database_id}")
    
    # Run pipeline
    try:
        stats = await run_second_brain_etl(
            notion_api_key=notion_api_key,
            notion_database_id=notion_database_id,
            max_crawl_pages=50,  # Reasonable limit
            quality_threshold=0.3  # Lower threshold for initial collection
        )
        
        print("\n🎉 ETL Pipeline completed successfully!")
        print(f"📈 Final stats: {stats['documents_stored']} documents stored")
        
    except Exception as e:
        print(f"\n❌ ETL Pipeline failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main())) 