"""
Web Crawler configuration models for the Second Brain AI Assistant.
This module defines all configuration models used for web crawling operations.
"""

from dataclasses import dataclass


@dataclass
class BrowserConfig:
    """Browser configuration for web crawling."""
    headless: bool = True
    verbose: bool = False


@dataclass
class CrawlerRunConfig:
    """Configuration for crawler runs."""
    cache_mode: str = "BYPASS"
    stream: bool = False 