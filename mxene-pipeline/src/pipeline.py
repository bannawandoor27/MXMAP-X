"""Main pipeline orchestrator for end-to-end processing."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .config import config
from .extractor import MXeneExtractor
from .harvester import SemanticScholarHarvester


logger = logging.getLogger(__name__)


class MXenePipeline:
    """Orchestrates the complete data extraction pipeline."""
    
    def __init__(self):
        self.harvester = SemanticScholarHarvester()
        self.extractor = MXeneExtractor()
        
        # Setup logging
        log_file = config.logs_dir / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _get_unprocessed_pdfs(self) -> List[Path]:
        """Find PDFs that haven't been processed yet."""
        # Get all PDFs from both directories
        downloaded_pdfs = list(config.pdfs_downloaded.glob("*.pdf"))
        manual_pdfs = list(config.pdfs_manual.glob("*.pdf"))
        all_pdfs = downloaded_pdfs + manual_pdfs
        
        # Get already processed files
        processed_files = {
            json.loads(p.read_text()).get("paper_doi", "")
            for p in config.processed_json.glob("*.json")
        }
        
        # Filter unprocessed
        unprocessed = [
            pdf for pdf in all_pdfs
            if pdf.stem not in processed_files
        ]
        
        logger.info(
            f"Found {len(unprocessed)} unprocessed PDFs "
            f"({len(all_pdfs)} total, {len(processed_files)} already processed)"
        )
        
        return unprocessed
    
    def harvest_papers(
        self,
        query: Optional[str] = None,
        max_papers: Optional[int] = None
    ) -> int:
        """
        Step 1: Download papers from Semantic Scholar.
        
        Args:
            query: Search query (defaults to config)
            max_papers: Maximum papers to download
        
        Returns:
            Number of papers downloaded
        """
        logger.info("=" * 60)
        logger.info("STEP 1: HARVESTING PAPERS")
        logger.info("=" * 60)
        
        count = self.harvester.harvest(query=query, max_papers=max_papers)
        
        logger.info(f"Harvesting complete: {count} new papers downloaded")
        return count
    
    def extract_data(self, pdf_paths: Optional[List[Path]] = None) -> int:
        """
        Step 2: Extract structured data from PDFs.
        
        Args:
            pdf_paths: Specific PDFs to process (defaults to all unprocessed)
        
        Returns:
            Number of successful extractions
        """
        logger.info("=" * 60)
        logger.info("STEP 2: EXTRACTING DATA")
        logger.info("=" * 60)
        
        if pdf_paths is None:
            pdf_paths = self._get_unprocessed_pdfs()
        
        if not pdf_paths:
            logger.info("No PDFs to process")
            return 0
        
        success_count = 0
        
        for pdf_path in tqdm(pdf_paths, desc="Extracting data"):
            try:
                # Extract data
                result = self.extractor.extract_from_pdf(pdf_path)
                
                if result and result.experiments:
                    # Save to JSON
                    output_path = config.processed_json / f"{pdf_path.stem}.json"
                    self.extractor.save_result(result, output_path)
                    success_count += 1
                else:
                    logger.warning(f"No data extracted from {pdf_path.name}")
            
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(
            f"Extraction complete: {success_count}/{len(pdf_paths)} successful"
        )
        return success_count
    
    def aggregate_results(self, output_file: str = "output_dataset.csv") -> pd.DataFrame:
        """
        Step 3: Aggregate all JSON results into a single dataset.
        
        Args:
            output_file: Output CSV filename
        
        Returns:
            Aggregated DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 3: AGGREGATING RESULTS")
        logger.info("=" * 60)
        
        json_files = list(config.processed_json.glob("*.json"))
        
        if not json_files:
            logger.warning("No JSON files found to aggregate")
            return pd.DataFrame()
        
        all_experiments = []
        
        for json_file in tqdm(json_files, desc="Loading results"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract experiments
                for exp in data.get("experiments", []):
                    # Add metadata
                    exp["source_filename"] = json_file.stem
                    exp["paper_doi"] = data.get("paper_doi")
                    exp["paper_title"] = data.get("paper_title")
                    exp["extraction_confidence"] = data.get("extraction_confidence")
                    exp["extraction_timestamp"] = datetime.fromtimestamp(
                        json_file.stat().st_mtime
                    ).isoformat()
                    exp["model_used"] = config.ollama_model
                    
                    all_experiments.append(exp)
            
            except Exception as e:
                logger.error(f"Failed to load {json_file.name}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(all_experiments)
        
        if df.empty:
            logger.warning("No experiments found in JSON files")
            return df
        
        # Reorder columns for better readability
        priority_cols = [
            "mxene_type",
            "electrolyte",
            "areal_capacitance_mf_cm2",
            "specific_capacitance_f_g",
            "volumetric_capacitance_f_cm3",
            "specific_surface_area_m2_g",
            "synthesis_method",
            "source_filename",
            "paper_doi"
        ]
        
        # Put priority columns first
        other_cols = [col for col in df.columns if col not in priority_cols]
        ordered_cols = [col for col in priority_cols if col in df.columns] + other_cols
        df = df[ordered_cols]
        
        # Save to CSV
        output_path = config.base_dir / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"Aggregated {len(df)} experiments from {len(json_files)} papers")
        logger.info(f"Saved to {output_path}")
        
        # Print summary statistics
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics of the dataset."""
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal experiments: {len(df)}")
        print(f"Unique papers: {df['source_filename'].nunique()}")
        
        if 'mxene_type' in df.columns:
            print(f"\nMXene types:")
            print(df['mxene_type'].value_counts().head(10))
        
        if 'areal_capacitance_mf_cm2' in df.columns:
            cap_data = df['areal_capacitance_mf_cm2'].dropna()
            if not cap_data.empty:
                print(f"\nAreal Capacitance (mF/cm²):")
                print(f"  Mean: {cap_data.mean():.2f}")
                print(f"  Median: {cap_data.median():.2f}")
                print(f"  Range: {cap_data.min():.2f} - {cap_data.max():.2f}")
        
        print("\n" + "=" * 60)
    
    def run_full_pipeline(
        self,
        harvest: bool = True,
        extract: bool = True,
        aggregate: bool = True,
        max_papers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run the complete pipeline end-to-end.
        
        Args:
            harvest: Whether to harvest new papers
            extract: Whether to extract data from PDFs
            aggregate: Whether to aggregate results
            max_papers: Maximum papers to harvest
        
        Returns:
            Final aggregated DataFrame
        """
        logger.info("=" * 60)
        logger.info("STARTING MXENE DATA PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Ollama model: {config.ollama_model}")
        logger.info(f"Ollama URL: {config.ollama_base_url}")
        
        # Step 1: Harvest
        if harvest:
            self.harvest_papers(max_papers=max_papers)
        
        # Step 2: Extract
        if extract:
            self.extract_data()
        
        # Step 3: Aggregate
        df = pd.DataFrame()
        if aggregate:
            df = self.aggregate_results()
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return df


def main():
    """CLI entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MXene Data Extraction Pipeline"
    )
    parser.add_argument(
        "--harvest-only",
        action="store_true",
        help="Only harvest papers, don't extract"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract from existing PDFs"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing JSON files"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Maximum number of papers to harvest"
    )
    
    args = parser.parse_args()
    
    pipeline = MXenePipeline()
    
    if args.harvest_only:
        pipeline.harvest_papers(max_papers=args.max_papers)
    elif args.extract_only:
        pipeline.extract_data()
    elif args.aggregate_only:
        pipeline.aggregate_results()
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(max_papers=args.max_papers)


if __name__ == "__main__":
    main()
