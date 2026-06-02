#!/usr/bin/env python3
"""
Extract structured MXene experimental data from a corpus of PDFs.

Multi-chunk extraction: processes ALL text chunks per paper (not just the first),
merges experiments, and deduplicates by (mxene_type, electrolyte, capacitance).

Usage:
    cd MXMAP-X/mxene-pipeline
    source venv/bin/activate
    python ../scripts/extract_corpus.py [--delay 5] [--reprocess]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "mxene-pipeline"

sys.path.insert(0, str(PIPELINE_DIR))

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("extract_corpus")


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_pdfs(corpus_dir: Path) -> list[Path]:
    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    log.info(f"Found {len(pdfs)} PDFs in {corpus_dir}")
    return pdfs


def dedup_experiments(experiments: list[dict]) -> list[dict]:
    """
    Deduplicate experiments that are identical or near-identical.
    
    Key = (mxene_type, electrolyte, areal_capacitance, specific_capacitance)
    """
    seen = set()
    deduped = []
    for exp in experiments:
        key = (
            (exp.get("mxene_type") or "").strip().lower(),
            (exp.get("electrolyte") or "").strip().lower(),
            exp.get("areal_capacitance_mf_cm2"),
            exp.get("specific_capacitance_f_g"),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(exp)
    return deduped


def extract_paper_multichunk(
    pdf_path: Path,
    extractor,
    cleaner,
    delay_between_chunks: float = 3.0,
) -> dict | None:
    """
    Process ALL chunks of a paper and merge experiments.
    
    Returns the merged ExtractionResult dict or None.
    """
    try:
        processed = cleaner.process(pdf_path)
        chunks = processed["chunks"]
        log.info(
            f"  Text: {processed['cleaned_length']} chars, {len(chunks)} chunk(s)"
        )

        all_experiments = []
        paper_title = None
        paper_doi = pdf_path.stem

        for i, chunk in enumerate(chunks):
            log.info(f"  → Chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            result = extractor.extract_from_text(chunk)

            if result and result.experiments:
                for exp in result.experiments:
                    all_experiments.append(exp.model_dump(by_alias=True))
                if result.paper_title and not paper_title:
                    paper_title = result.paper_title
                log.info(f"    ✓ {len(result.experiments)} experiment(s)")
            else:
                log.info(f"    – no experiments in this chunk")

            # Small gap between chunks (less than between papers)
            if i < len(chunks) - 1 and delay_between_chunks > 0:
                time.sleep(delay_between_chunks)

        if not all_experiments:
            return None

        # Deduplicate across chunks
        unique_experiments = dedup_experiments(all_experiments)
        log.info(
            f"  Total: {len(all_experiments)} raw → "
            f"{len(unique_experiments)} unique experiments"
        )

        return {
            "experiments": unique_experiments,
            "paper_title": paper_title,
            "paper_doi": paper_doi,
            "extraction_confidence": "medium",
            "num_chunks_processed": len(chunks),
        }

    except Exception as exc:
        log.error(f"  ✗ Error: {exc}")
        return None


# ── Main extraction loop ──────────────────────────────────────────────────────

def run_extraction(
    corpus_dir: Path,
    output_csv: Path,
    json_dir: Path,
    model: str,
    delay_seconds: float = 5.0,
    reprocess: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:

    os.environ.setdefault("OLLAMA_MODEL", model)
    os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

    from src.config import config
    from src.extractor import MXeneExtractor
    from src.cleaner import PDFCleaner

    config.ollama_model = model
    json_dir.mkdir(parents=True, exist_ok=True)

    # Optionally clear old results for re-extraction
    if reprocess:
        old_files = list(json_dir.glob("*.json"))
        if old_files:
            log.info(f"--reprocess: deleting {len(old_files)} old JSON results")
            for f in old_files:
                f.unlink()

    extractor = MXeneExtractor()
    cleaner = PDFCleaner()

    pdfs = collect_pdfs(corpus_dir)

    # Load existing results for skip logic
    existing: dict[str, dict] = {}
    for jf in json_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text())
            existing[data.get("paper_doi") or jf.stem] = data
        except Exception:
            pass

    skipped = 0
    success = 0
    failed = 0

    for pdf_path in tqdm(pdfs, desc="Extracting"):
        if limit is not None and success >= limit:
            log.info(f"Reached limit of {limit} successful extractions. Stopping.")
            break

        stem = pdf_path.stem

        if stem in existing:
            skipped += 1
            continue

        log.info(f"[PROCESS] {pdf_path.name}")
        merged = extract_paper_multichunk(
            pdf_path, extractor, cleaner,
            delay_between_chunks=2.0,
        )

        if merged and merged["experiments"]:
            out_path = json_dir / f"{stem}.json"
            out_path.write_text(json.dumps(merged, indent=2))
            existing[stem] = merged
            success += 1
            log.info(
                f"  ✓ Saved {len(merged['experiments'])} experiment(s)"
            )
        else:
            log.warning(f"  ✗ No experiments from {pdf_path.name}")
            failed += 1

        # Cool-down between papers
        if delay_seconds > 0:
            log.info(f"  ⏸  Cooling {delay_seconds:.0f}s...")
            time.sleep(delay_seconds)

    log.info(
        f"\nDone — success: {success}, skipped: {skipped}, failed: {failed}"
    )

    # ── Aggregate to CSV ──────────────────────────────────────────────────────
    all_rows = []
    for stem, data in existing.items():
        for exp in data.get("experiments", []):
            exp["source_filename"] = stem
            exp["paper_doi"] = data.get("paper_doi", stem)
            exp["paper_title"] = data.get("paper_title")
            exp["extraction_confidence"] = data.get("extraction_confidence")
            exp["extraction_timestamp"] = datetime.now().isoformat()
            exp["model_used"] = model
            all_rows.append(exp)

    if not all_rows:
        log.warning("No experiments — CSV will be empty.")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(all_rows)
        priority = [
            "mxene_type",
            "electrolyte",
            "areal_capacitance_mf_cm2",
            "specific_capacitance_f_g",
            "volumetric_capacitance_f_cm3",
            "ssa_m2_g",
            "synthesis_method",
            "annealing_temperature_c",
            "scan_rate_mv_s",
            "current_density_a_g",
            "cycling_stability",
            "num_cycles",
            "source_filename",
            "paper_doi",
            "paper_title",
            "extraction_confidence",
            "model_used",
        ]
        other = [c for c in df.columns if c not in priority]
        df = df[[c for c in priority if c in df.columns] + other]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("CORPUS DATASET SUMMARY")
    print("=" * 70)
    print(f"  Total experiments : {len(df)}")
    print(f"  Unique papers     : {df['source_filename'].nunique() if not df.empty else 0}")
    if not df.empty and "mxene_type" in df.columns:
        print(f"\n  MXene types:")
        for v, c in df["mxene_type"].value_counts().head(10).items():
            print(f"    {v}: {c}")
    if not df.empty and "areal_capacitance_mf_cm2" in df.columns:
        cap = df["areal_capacitance_mf_cm2"].dropna()
        if not cap.empty:
            print(f"\n  Areal capacitance (mF/cm²):")
            print(f"    mean={cap.mean():.1f}  median={cap.median():.1f}  "
                  f"range=[{cap.min():.1f}, {cap.max():.1f}]")
    print(f"\n  Output: {output_csv}")
    print("=" * 70)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract MXene dataset from a corpus of PDFs (multi-chunk)."
    )
    parser.add_argument(
        "--corpus-dir", type=Path,
        default=PROJECT_ROOT / "Readings" / "ML_for_Mxenes_corpus",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data" / "corpus_dataset.csv",
    )
    parser.add_argument(
        "--json-dir", type=Path,
        default=PROJECT_ROOT / "mxene-pipeline" / "data" / "processed_json",
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5:7b",
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Seconds between papers (default 5). Use 0 to go full speed.",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Delete existing JSONs and re-extract everything with multi-chunk.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after extracting this many successful papers.",
    )
    args = parser.parse_args()

    if not args.corpus_dir.exists():
        log.error(f"Corpus dir not found: {args.corpus_dir}")
        sys.exit(1)

    log.info(f"Corpus : {args.corpus_dir}")
    log.info(f"Output : {args.output}")
    log.info(f"Model  : {args.model}")
    log.info(f"Delay  : {args.delay}s  |  reprocess={args.reprocess} | limit={args.limit}")

    run_extraction(
        corpus_dir=args.corpus_dir,
        output_csv=args.output,
        json_dir=args.json_dir,
        model=args.model,
        delay_seconds=args.delay,
        reprocess=args.reprocess,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
