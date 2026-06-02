"""Semantic Scholar paper harvester with fault tolerance."""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

from .config import config


logger = logging.getLogger(__name__)


class DownloadLog:
    """Manages download history to prevent duplicates."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.downloaded: Dict[str, dict] = self._load()
    
    def _load(self) -> Dict[str, dict]:
        """Load existing download log."""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save(self):
        """Persist download log to disk."""
        with open(self.log_path, 'w') as f:
            json.dump(self.downloaded, f, indent=2)
    
    def is_downloaded(self, doi: str) -> bool:
        """Check if DOI already downloaded."""
        return doi in self.downloaded
    
    def mark_downloaded(self, doi: str, metadata: dict):
        """Mark DOI as downloaded with metadata."""
        self.downloaded[doi] = {
            **metadata,
            "timestamp": time.time()
        }
        self.save()


class SemanticScholarHarvester:
    """Harvests open-access papers from Semantic Scholar."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self):
        self.session = requests.Session()
        if config.s2_api_key:
            self.session.headers.update({"x-api-key": config.s2_api_key})
        
        self.download_log = DownloadLog(
            config.logs_dir / "download_log.json"
        )
        
        # Setup logging
        log_file = config.logs_dir / "harvester.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(
            min=config.retry_wait_min,
            max=config.retry_wait_max
        ),
        retry=retry_if_exception_type((
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        ))
    )
    def _search_papers(
        self,
        query: str,
        limit: int = 100,
        fields: Optional[List[str]] = None
    ) -> List[dict]:
        """Search for papers with retry logic."""
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "year",
                "authors", "openAccessPdf", "externalIds"
            ]
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields),
            "openAccessPdf": ""  # Only open access
        }
        
        logger.info(f"Searching for: {query}")
        response = self.session.get(url, params=params, timeout=30)
        
        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited. Waiting {retry_after}s")
            time.sleep(retry_after)
            raise requests.exceptions.HTTPError("Rate limited", response=response)
        
        response.raise_for_status()
        data = response.json()
        
        papers = data.get("data", [])
        logger.info(f"Found {len(papers)} papers")
        return papers
    
    @staticmethod
    def _sanitize_filename(doi: str) -> str:
        """Convert DOI to safe filename."""
        return doi.replace("/", "_").replace(":", "_")
    
    @staticmethod
    def _validate_pdf(file_path: Path) -> bool:
        """Verify file is a valid PDF by checking magic numbers."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except Exception as e:
            logger.error(f"PDF validation failed for {file_path}: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type((
            requests.exceptions.RequestException,
        ))
    )
    def _download_pdf(self, url: str, output_path: Path) -> bool:
        """Download PDF with retry logic."""
        response = self.session.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Validate downloaded file
        if not self._validate_pdf(output_path):
            output_path.unlink()
            raise ValueError(f"Downloaded file is not a valid PDF: {url}")
        
        return True
    
    def harvest(
        self,
        query: Optional[str] = None,
        max_papers: Optional[int] = None
    ) -> int:
        """
        Harvest papers from Semantic Scholar.
        
        Args:
            query: Search query (defaults to config)
            max_papers: Maximum papers to download (defaults to config)
        
        Returns:
            Number of papers successfully downloaded
        """
        query = query or config.search_query
        max_papers = max_papers or config.s2_max_results
        
        # Search for papers
        papers = self._search_papers(query, limit=max_papers)
        
        # Filter for open access with valid PDFs
        downloadable = [
            p for p in papers
            if p.get("openAccessPdf") and p["openAccessPdf"].get("url")
        ]
        
        logger.info(f"Found {len(downloadable)} downloadable papers")
        
        downloaded_count = 0
        
        for paper in tqdm(downloadable, desc="Downloading papers"):
            try:
                # Extract DOI or use paper ID
                doi = None
                if paper.get("externalIds"):
                    doi = paper["externalIds"].get("DOI")
                
                if not doi:
                    doi = paper["paperId"]
                
                # Check if already downloaded
                if self.download_log.is_downloaded(doi):
                    logger.debug(f"Skipping already downloaded: {doi}")
                    continue
                
                # Download PDF
                pdf_url = paper["openAccessPdf"]["url"]
                filename = self._sanitize_filename(doi) + ".pdf"
                output_path = config.pdfs_downloaded / filename
                
                logger.info(f"Downloading: {paper.get('title', 'Unknown')}")
                self._download_pdf(pdf_url, output_path)
                
                # Log successful download
                self.download_log.mark_downloaded(doi, {
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "filename": filename,
                    "pdf_url": pdf_url
                })
                
                downloaded_count += 1
                
                # Politeness delay
                delay = random.uniform(*config.s2_rate_limit_delay)
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Failed to download {paper.get('paperId')}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {downloaded_count} new papers")
        return downloaded_count


def main():
    """CLI entry point for harvester."""
    harvester = SemanticScholarHarvester()
    count = harvester.harvest()
    print(f"\n✓ Downloaded {count} papers to {config.pdfs_downloaded}")


if __name__ == "__main__":
    main()
