import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def test_mongodb_connection():
    """
    Test basic MongoDB connection
    """
    try:
        mongo_url = os.getenv('MONGODB_URL')
        if not mongo_url:
            print("❌ MONGODB_URL environment variable not set")
            return False

        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ Successfully connected to MongoDB!")
        return True

    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_mongodb_connection() 