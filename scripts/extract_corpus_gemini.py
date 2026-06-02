#!/usr/bin/env python3
"""
Automate MXene experimental data extraction using Gemini Pro.
Uses the new `google-genai` SDK, native PDF uploads, and Structured Outputs 
to guarantee the exact JSON format without parsing errors.

Requirements:
    pip install google-genai pydantic tqdm pandas
    export GEMINI_API_KEY="your_api_key_here"

Usage:
    python scripts/extract_corpus_gemini.py [--limit 5]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Try to import the new SDK
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    print("Error: Missing google-genai library.")
    print("Please run: pip install google-genai pydantic tqdm pandas")
    sys.exit(1)

from pydantic import BaseModel, Field
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = PROJECT_ROOT / "mxene-pipeline"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("extract_gemini")


# ── Pydantic Schema for Structured Outputs ────────────────────────────────────

class MXeneExperiment(BaseModel):
    mxene_type: str | None = Field(
        default=None, description="Type of MXene material (e.g., Ti3C2Tx, V2CTx) or null"
    )
    electrolyte: str | None = Field(
        default=None, description="Electrolyte composition (e.g., H2SO4, KOH) or null"
    )
    areal_capacitance_mf_cm2: float | None = Field(
        default=None, description="Areal capacitance in mF/cm²"
    )
    volumetric_capacitance_f_cm3: float | None = Field(
        default=None, description="Volumetric capacitance in F/cm³"
    )
    specific_capacitance_f_g: float | None = Field(
        default=None, description="Specific capacitance in F/g"
    )
    ssa_m2_g: float | None = Field(
        default=None, description="Specific surface area in m²/g from BET analysis"
    )
    annealing_temperature_c: float | None = Field(
        default=None, description="Annealing temperature in Celsius"
    )
    scan_rate_mv_s: float | None = Field(
        default=None, description="Scan rate in mV/s for CV measurements"
    )
    current_density_a_g: float | None = Field(
        default=None, description="Current density in A/g for GCD measurements"
    )
    cycling_stability: float | None = Field(
        default=None, description="Capacitance retention percentage after cycling"
    )
    num_cycles: int | None = Field(
        default=None, description="Number of charge-discharge cycles tested"
    )
    synthesis_method: str | None = Field(
        default=None, description="Synthesis method (e.g., HF etching, MILD)"
    )

class ExtractionResult(BaseModel):
    experiments: list[MXeneExperiment] = Field(
        description="List of extracted experimental data points"
    )
    paper_title: str | None = Field(default=None)
    extraction_confidence: str | None = Field(
        default="high", description="high, medium, or low"
    )


# ── Extraction Logic ──────────────────────────────────────────────────────────

PROMPT = """You are a materials science data extraction expert specializing in MXene supercapacitor research.

Read the attached scientific paper PDF in its entirety.
Extract ALL experimental data points about the electrochemical performance of the MXene devices.

CRITICAL INSTRUCTIONS:
1. Because you are reading the entire document, the materials used (like MXene type and electrolyte) are often mentioned in the "Materials and Methods" section, while the capacitance and cycling performance are mentioned in the "Results" section. You MUST connect these pieces of information together for each extracted experiment. Do NOT leave mxene_type or electrolyte as null if they are mentioned anywhere in the paper.
2. If an explicit value is missing from the entire document, return null (not 0).
3. Be precise with units to match the requested fields. Convert everything to pure numbers.
4. Extract multiple experiments if the paper tests various conditions (e.g., varying annealing temp or trying different electrolytes).
"""

def extract_from_pdf(
    client: genai.Client, 
    pdf_path: Path, 
    model_name: str = "gemini-2.5-pro"
) -> dict | None:
    """Upload PDF, extract data via Gemini, and delete the file."""
    uploaded_file = None
    try:
        # 1. Upload the PDF to the Gemini File API
        log.info(f"  Uploading {pdf_path.name} to Gemini...")
        uploaded_file = client.files.upload(file=str(pdf_path))
        
        # 2. Call the model with Structured Outputs
        log.info(f"  Extracting with {model_name}...")
        response = client.models.generate_content(
            model=model_name,
            contents=[uploaded_file, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult,
                temperature=0.0,
            ),
        )
        
        if not response.text:
            log.warning("  ⚠ Empty response from Gemini")
            return None
            
        # Parse the guaranteed JSON
        data = json.loads(response.text)
        data["paper_doi"] = pdf_path.stem
        return data

    except APIError as e:
        log.error(f"  Gemini API Error: {e.message}")
        if e.code == 429:
            raise  # Bubble up rate limits to trigger retry
        return None
    except Exception as e:
        log.error(f"  Unexpected Error: {e}")
        return None
    finally:
        # Cleanup: Delete the file from Google's servers
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except:
                pass


def run_extraction(
    corpus_dir: Path,
    output_csv: Path,
    json_dir: Path,
    model_name: str,
    limit: int | None = None,
) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY environment variable is not set!")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    json_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    if not pdfs:
        log.error(f"No PDFs found in {corpus_dir}")
        return

    # Load existing to resume
    existing = {}
    for jf in json_dir.glob("*.gemini.json"):
        try:
            data = json.loads(jf.read_text())
            existing[data.get("paper_doi") or jf.name.replace(".gemini.json", "")] = data
        except:
            pass
            
    success = 0
    tqdm_pdfs = tqdm(pdfs, desc="Extracting via Gemini")
    
    for pdf_path in tqdm_pdfs:
        if limit and success >= limit:
            break
            
        stem = pdf_path.stem
        if stem in existing:
            continue

        log.info(f"\n[PROCESS] {pdf_path.name}")
        
        # Exponential backoff for rate limits
        max_retries = 5
        base_delay = 10
        result = None
        
        for attempt in range(max_retries):
            try:
                result = extract_from_pdf(client, pdf_path, model_name=model_name)
                break  # Success
            except APIError as e:
                if e.code == 429: # Resource Exhausted (Throttling)
                    delay = base_delay * (2 ** attempt)
                    log.warning(f"  [429 Throttled] Sleeping for {delay}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(delay)
                else:
                    break

        if result and result.get("experiments"):
            out_path = json_dir / f"{stem}.gemini.json"
            out_path.write_text(json.dumps(result, indent=2))
            existing[stem] = result
            success += 1
            log.info(f"  ✓ Saved {len(result['experiments'])} experiment(s)")
            
            # Anti-throttling base gap for free tier (approx 15 RPM limit)
            time.sleep(4) 
        else:
            log.warning(f"  ✗ Failed to extract from {pdf_path.name}")
            
    # Compile to CSV
    all_rows = []
    for stem, data in existing.items():
        for exp in data.get("experiments", []):
            exp["source_filename"] = stem
            exp["paper_doi"] = data.get("paper_doi", stem)
            exp["paper_title"] = data.get("paper_title")
            exp["extraction_timestamp"] = datetime.now().isoformat()
            exp["model_used"] = model_name
            all_rows.append(exp)

    if all_rows:
        df = pd.DataFrame(all_rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        # Use a gemini specific dataset name if testing
        csv_path = output_csv.with_name(f"{output_csv.stem}_gemini.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"\nSaved aggregated dataset to {csv_path} ({len(df)} rows)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "Readings" / "ML_for_Mxenes_corpus")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "corpus_dataset.csv")
    parser.add_argument("--json-dir", type=Path, default=PROJECT_ROOT / "mxene-pipeline" / "data" / "processed_json")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_extraction(
        corpus_dir=args.corpus_dir,
        output_csv=args.output,
        json_dir=args.json_dir,
        model_name=args.model,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
