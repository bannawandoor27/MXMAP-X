#!/usr/bin/env python3
"""
Automate MXene experimental data extraction using the Gemini CLI.
This script orchestrates the `gemini` CLI tool over the entire corpus of PDFs,
handling JSON extraction, parsing, retry logic, and compilation.

Requirements:
    gemini CLI installed and configured
    pip install pandas tqdm

Usage:
    python scripts/extract_corpus_gemini_cli.py [--limit 5]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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
log = logging.getLogger("extract_gemini_cli")

JSON_SCHEMA = """
{
  "$defs": {
    "MXeneExperiment": {
      "type": "object",
      "properties": {
        "mxene_type": { "type": ["string", "null"] },
        "electrolyte": { "type": ["string", "null"] },
        "areal_capacitance_mf_cm2": { "type": ["number", "null"] },
        "volumetric_capacitance_f_cm3": { "type": ["number", "null"] },
        "specific_capacitance_f_g": { "type": ["number", "null"] },
        "ssa_m2_g": { "type": ["number", "null"] },
        "annealing_temperature_c": { "type": ["number", "null"] },
        "scan_rate_mv_s": { "type": ["number", "null"] },
        "current_density_a_g": { "type": ["number", "null"] },
        "cycling_stability": { "type": ["number", "null"] },
        "num_cycles": { "type": ["integer", "null"] },
        "synthesis_method": { "type": ["string", "null"] }
      }
    }
  },
  "type": "object",
  "required": ["experiments"],
  "properties": {
    "experiments": {
      "type": "array",
      "items": { "$ref": "#/$defs/MXeneExperiment" }
    },
    "paper_title": { "type": ["string", "null"] },
    "paper_doi": { "type": ["string", "null"] },
    "extraction_confidence": { "type": ["string", "null"] }
  }
}
"""

def extract_from_pdf_cli(pdf_path: Path, model_name: str, cleaner) -> dict | None:
    """Uses the `gemini` CLI tool to read and extract data from the PDF."""
    
    # 1. Extract the text ourselves to bypass Gemini CLI's sandboxed file read 
    # and to prevent the CLI from freezing when it tries to read a binary PDF as text.
    try:
        processed = cleaner.process(pdf_path)
        # join all chunks back together into a single text since we have high context limits
        full_text = "\n\n".join(processed["chunks"])
    except Exception as e:
        log.error(f"  Failed local PDF extraction: {e}")
        return None

    # 2. Tell the model what structure we want
    prompt = f"""You are a materials science data extraction expert.
You have been provided with the full text of a scientific paper via standard input.

Extract ALL experimental data points about the electrochemical performance of the MXene devices inside it.

CRITICAL INSTRUCTIONS:
1. Because you are reading the entire document, the materials used (like MXene type and electrolyte) are often mentioned in the "Materials and Methods" section, while the capacitance and cycling performance are mentioned in the "Results" section. You MUST connect these pieces of information together for each extracted experiment. Do NOT leave mxene_type or electrolyte as null if they are mentioned ANYWHERE in the paper.
2. If an explicit value is missing from the entire document, return null (not 0).
3. Be precise with units. Convert everything to pure numbers.
4. Extract multiple experiments if the paper tests various conditions (e.g., varying annealing temp or trying different electrolytes).
5. Output ONLY a raw, perfectly valid JSON object that strictly conforms to this schema:
{JSON_SCHEMA}

Do NOT wrap it in ```json blocks. Just start with {{ and end with }}. Do NOT respond with anything else.
"""

    cmd = [
        "gemini",
        "-m", model_name,
        "--yolo",            # auto-approve any tool uses
        "-o", "text",        # purely text output
        "-p", prompt
    ]

    try:
        log.info(f"  Calling gemini CLI for {pdf_path.name}...")
        result = subprocess.run(
            cmd,
            input=full_text,
            capture_output=True,
            text=True,
            timeout=300 # 5 minute timeout per agent run
        )
        
        output = result.stdout.strip()
        stderr = result.stderr.strip()
        
        if result.returncode != 0:
            log.error(f"  Gemini CLI failed with exit code {result.returncode}")
            log.error(f"  Stderr: {stderr}")
            return None

        if not output:
             # sometimes gemini cli prints normal text to stderr, let's check it
             output = stderr

        # Clean output to find the JSON block. Gemini CLI sometimes outputs extra UI logs or markdown
        # First try to find a json codeblock
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", output, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            # Fall back to finding the first { and last }
            start = output.find('{')
            end = output.rfind('}')
            if start != -1 and end != -1:
                json_str = output[start:end+1]
            else:
                log.error(f"  Could not find JSON in output:\n{output[:500]}...")
                return None
        
        try:
            data = json.loads(json_str)
            data["paper_doi"] = pdf_path.stem
            return data
        except json.JSONDecodeError as e:
            log.error(f"  JSON Parsing Error: {e}\nRaw JSON String:\n{json_str[:500]}...")
            return None

    except subprocess.TimeoutExpired:
        log.error("  Gemini CLI timed out after 5 minutes.")
        return None
    except Exception as e:
        log.error(f"  Unexpected error calling CLI: {e}")
        return None


def run_extraction(
    corpus_dir: Path,
    output_csv: Path,
    json_dir: Path,
    model_name: str,
    limit: int | None = None,
) -> None:
    # 0. Import the local cleaner
    sys.path.insert(0, str(PIPELINE_DIR))
    from src.cleaner import PDFCleaner
    cleaner = PDFCleaner()
    
    # Check if CLI is installed
    if subprocess.run(["which", "gemini"], capture_output=True).returncode != 0:
        log.error("The `gemini` CLI is not installed or not in your PATH.")
        sys.exit(1)

    json_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    if not pdfs:
        log.error(f"No PDFs found in {corpus_dir}")
        return

    # Load existing to resume
    existing = {}
    for jf in json_dir.glob("*.geminicli.json"):
        try:
            data = json.loads(jf.read_text())
            existing[data.get("paper_doi") or jf.name.replace(".geminicli.json", "")] = data
        except:
            pass
            
    success = 0
    
    for pdf_path in tqdm(pdfs, desc="Extracting via Gemini CLI"):
        if limit and success >= limit:
            break
            
        stem = pdf_path.stem
        if stem in existing:
            log.info(f"Skipping {pdf_path.name} (already processed)")
            continue

        log.info(f"\n[PROCESS] {pdf_path.name}")
        
        # Exponential backoff for rate limits / agent failures
        max_retries = 3
        base_delay = 10
        result = None
        
        for attempt in range(max_retries):
            result = extract_from_pdf_cli(pdf_path, model_name=model_name, cleaner=cleaner)
            if result is not None:
                break
            
            # If we returned None, it failed. Sleep before retrying just in case of rate limits
            delay = base_delay * (2 ** attempt)
            log.warning(f"  [Failure] Retrying {attempt+1}/{max_retries} after {delay}s...")
            time.sleep(delay)

        if result and result.get("experiments"):
            out_path = json_dir / f"{stem}.geminicli.json"
            out_path.write_text(json.dumps(result, indent=2))
            existing[stem] = result
            success += 1
            log.info(f"  ✓ Saved {len(result['experiments'])} experiment(s)")
            
            # Optional cool down to respect user's quotas
            time.sleep(5) 
        else:
            log.warning(f"  ✗ Failed to extract from {pdf_path.name} after all retries")
            
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
        # Use a distinctive dataset name
        csv_path = output_csv.with_name(f"{output_csv.stem}_geminicli.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"\nSaved aggregated dataset to {csv_path} ({len(df)} rows)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=PROJECT_ROOT / "Readings" / "ML_for_Mxenes_corpus")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "corpus_dataset.csv")
    parser.add_argument("--json-dir", type=Path, default=PROJECT_ROOT / "mxene-pipeline" / "data" / "processed_json_gemini_3_1")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-preview")
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
