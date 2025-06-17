"""
MongoDB connection management using Motor async driver.
Implements Factory pattern for creating connections and manages connection lifecycle.
Follows Single Responsibility Principle - only handles database connections.
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.config.settings import settings
from src.utils.logger import LoggerMixin
from .base import IDatabaseConnection, ConnectionError


class MongoDBConnection(IDatabaseConnection, LoggerMixin):
    """
    MongoDB connection manager using Motor async driver.
    Implements connection pooling, health checks, and graceful error handling.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database to use
            **kwargs: Additional Motor client options
        """
        self.connection_string = connection_string or settings.MONGODB_URL
        self.database_name = database_name or settings.DATABASE_NAME
        
        # Motor client options for production use
        self.client_options = {
            "maxPoolSize": kwargs.get("max_pool_size", 50),
            "minPoolSize": kwargs.get("min_pool_size", 5),
            "maxIdleTimeMS": kwargs.get("max_idle_time_ms", 60000),
            "serverSelectionTimeoutMS": kwargs.get("server_selection_timeout_ms", 30000),
            "connectTimeoutMS": kwargs.get("connect_timeout_ms", 30000),
            "socketTimeoutMS": kwargs.get("socket_timeout_ms", 30000),
            "waitQueueTimeoutMS": kwargs.get("wait_queue_timeout_ms", 30000),
            "retryWrites": True,
            "retryReads": True,
            **kwargs
        }
        
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """
        Establish connection to MongoDB Atlas.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.logger.info(f"Connecting to MongoDB: {self.database_name}")
            
            # Create Motor client with options
            self._client = AsyncIOMotorClient(
                self.connection_string,
                **self.client_options
            )
            
            # Test connection
            await self._client.admin.command('ping')
            
            # Get database reference
            self._database = self._client[self.database_name]
            
            self._is_connected = True
            self.logger.info("Successfully connected to MongoDB")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise ConnectionError(f"MongoDB connection failed: {str(e)}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            raise ConnectionError(f"Unexpected connection error: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Close MongoDB connection gracefully."""
        if self._client is not None:
            self.logger.info("Disconnecting from MongoDB")
            self._client.close()
            self._client = None
            self._database = None
            self._is_connected = False
            self.logger.info("Disconnected from MongoDB")
    
    async def health_check(self) -> bool:
        """
        Check if MongoDB connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if self._client is None or not self._is_connected:
                return False
            
            # Ping database with timeout
            await asyncio.wait_for(
                self._client.admin.command('ping'),
                timeout=2.0
            )
            return True
            
        except Exception as e:
            self.logger.warning(f"MongoDB health check failed: {str(e)}")
            return False
    
    def get_client(self) -> AsyncIOMotorClient:
        """
        Get the Motor client instance.
        
        Returns:
            AsyncIOMotorClient instance
            
        Raises:
            ConnectionError: If not connected
        """
        if self._client is None or not self._is_connected:
            raise ConnectionError("Not connected to MongoDB. Call connect() first.")
        return self._client
    
    def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get the database instance.
        
        Returns:
            AsyncIOMotorDatabase instance
            
        Raises:
            ConnectionError: If not connected
        """
        if self._database is None or not self._is_connected:
            raise ConnectionError("Not connected to MongoDB. Call connect() first.")
        return self._database
    
    def get_collection(self, collection_name: str):
        """
        Get a collection instance.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            AsyncIOMotorCollection instance
        """
        database = self.get_database()
        return database[collection_name]
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected to MongoDB."""
        return self._is_connected
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class MongoDBConnectionFactory:
    """
    Factory class for creating MongoDB connections.
    Implements Factory pattern for dependency injection and testing.
    """
    
    @staticmethod
    def create_connection(
        connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        **kwargs
    ) -> MongoDBConnection:
        """
        Create a new MongoDB connection instance.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Database name
            **kwargs: Additional connection options
            
        Returns:
            MongoDBConnection instance
        """
        return MongoDBConnection(
            connection_string=connection_string,
            database_name=database_name,
            **kwargs
        )
    
    @staticmethod
    @asynccontextmanager
    async def get_connection(
        connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        **kwargs
    ):
        """
        Get a MongoDB connection as an async context manager.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Database name
            **kwargs: Additional connection options
            
        Yields:
            MongoDBConnection instance
        """
        connection = MongoDBConnectionFactory.create_connection(
            connection_string, database_name, **kwargs
        )
        
        try:
            await connection.connect()
            yield connection
        finally:
            await connection.disconnect()


# Global connection instance for dependency injection
_global_connection: Optional[MongoDBConnection] = None


async def get_mongodb_connection() -> MongoDBConnection:
    """
    Get the global MongoDB connection instance.
    Implements Singleton pattern for connection reuse.
    
    Returns:
        MongoDBConnection instance
    """
    global _global_connection
    
    if _global_connection is None:
        _global_connection = MongoDBConnectionFactory.create_connection()
        await _global_connection.connect()
    
    elif not _global_connection.is_connected:
        await _global_connection.connect()
    
    return _global_connection


async def close_mongodb_connection() -> None:
    """Close the global MongoDB connection."""
    global _global_connection
    
    if _global_connection:
        await _global_connection.disconnect()
        _global_connection = None


# Test function
async def test_connection():
    """Test MongoDB connection."""
    async with MongoDBConnectionFactory.get_connection() as conn:
        health = await conn.health_check()
        print(f"Connection health: {health}")
        
        # Test basic operation
        db = conn.get_database()
        collections = await db.list_collection_names()
        print(f"Available collections: {collections}")


if __name__ == "__main__":
    asyncio.run(test_connection()) 