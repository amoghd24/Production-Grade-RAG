import asyncio
from src.data_pipeline.web_crawler import WebCrawler

async def test_crawler():
    crawler = WebCrawler()
    doc = await crawler.crawl_url("https://en.wikipedia.org/wiki/Chelsea_F.C.")
    if doc:
        print("Title:", doc.title)
        print("Content length:", len(doc.content))
        print("Sample content:", doc.content[:300])
    else:
        print("Crawling failed.")

if __name__ == "__main__":
    asyncio.run(test_crawler()) 