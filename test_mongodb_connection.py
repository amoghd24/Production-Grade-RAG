import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mongodb_connection():
    """
    Test basic MongoDB connection with detailed error reporting
    """
    try:
        # Get MongoDB URL from environment variable
        mongo_url = os.getenv('MONGODB_URL')
        if not mongo_url:
            logger.error("❌ MONGODB_URL environment variable not set")
            return False

        logger.info("🔌 Testing MongoDB connection...")
        logger.info(f"📝 Connection URL: {mongo_url[:20]}...{mongo_url[-20:]}")  # Show only parts of URL for security

        # Log SSL version information
        logger.info(f"🔒 SSL Version: {ssl.OPENSSL_VERSION}")
        logger.info(f"🔒 SSL Protocol Version: {ssl.PROTOCOL_TLS}")

        # Create client with explicit SSL settings
        client = MongoClient(
            mongo_url,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            tls=True,
            tlsAllowInvalidCertificates=False
        )

        # Test the connection
        client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB!")
        return True

    except ConnectionFailure as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        return False
    except ServerSelectionTimeoutError as e:
        logger.error(f"❌ Server selection timeout: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting MongoDB Connection Test")
    success = test_mongodb_connection()
    if success:
        logger.info("✅ Test completed successfully")
    else:
        logger.error("❌ Test failed")
        sys.exit(1) 