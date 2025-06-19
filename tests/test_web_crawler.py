import asyncio
from src.data_pipeline.web_crawler import AsyncWebCrawler

async def test_crawler():
    async with AsyncWebCrawler() as crawler:
        doc = await crawler.crawl_url("https://en.wikipedia.org/wiki/Chelsea_F.C.")
        if doc:
            print("Title:", doc.title)
            print("Content length:", len(doc.content))
            print("Sample content:", doc.content[:100000])
        else:
            print("Crawling failed.")

if __name__ == "__main__":
    asyncio.run(test_crawler()) 