import asyncio
from src.data_pipeline.integrated_collector import IntegratedDataCollector
import os

async def main():
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_database_id = os.getenv("NOTION_DATABASE_ID")

    collector = IntegratedDataCollector()
    # Step 1: Collect Notion data
    await collector._collect_notion_data(
        api_key=notion_api_key,
        database_id=notion_database_id,
        search_query=""
    )
    print(f"Step 1: Collected {len(collector.notion_documents)} Notion documents")
    print(f"Step 1: Found {len(getattr(collector, 'embedded_urls', []))} embedded URLs")
    if getattr(collector, 'embedded_urls', []):
        print("Sample embedded URL:", collector.embedded_urls[0])

    # Step 2: Crawl embedded links (if any)
    if getattr(collector, 'embedded_urls', []):
        # For testing, limit to 10 URLs
        test_urls = collector.embedded_urls[:10]
        print(f"\nTesting with first 10 URLs out of {len(collector.embedded_urls)} total URLs")
        await collector._crawl_embedded_links(max_pages=10)
        print(f"Step 2: Crawled {len(collector.crawled_documents)} web documents")
        if collector.crawled_documents:
            print("Sample crawled doc title:", collector.crawled_documents[0].title)
            print("Sample crawled doc content (first 200 chars):", collector.crawled_documents[0].content[:200])
    else:
        print("No URLs to crawl.")

    # Step 3: Combine
    collector._combine_documents()
    print(f"Step 3: Combined total documents: {len(collector.all_documents)}")
    sources = [doc.source.value for doc in collector.all_documents]
    print("Sources in combined docs:", sources)

if __name__ == "__main__":
    asyncio.run(main())
    