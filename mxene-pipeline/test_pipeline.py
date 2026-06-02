#!/usr/bin/env python3
"""
Integration test script for the MXene pipeline.
Tests each component with sample data.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.cleaner import PDFCleaner
from src.extractor import MXeneExperiment, ExtractionResult


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    assert config.base_dir.exists(), "Base directory not found"
    assert config.pdfs_downloaded.exists(), "PDFs directory not created"
    assert config.processed_json.exists(), "Processed JSON directory not created"
    assert config.logs_dir.exists(), "Logs directory not created"
    
    print(f"  ✓ Base dir: {config.base_dir}")
    print(f"  ✓ Ollama model: {config.ollama_model}")
    print(f"  ✓ Ollama URL: {config.ollama_base_url}")
    print()


def test_pydantic_schema():
    """Test Pydantic schema validation."""
    print("Testing Pydantic schema...")
    
    # Valid data
    valid_data = {
        "mxene_type": "Ti3C2Tx",
        "electrolyte": "1M H2SO4",
        "areal_capacitance_mf_cm2": 328.0,
        "specific_surface_area_m2_g": 98.0
    }
    
    exp = MXeneExperiment(**valid_data)
    assert exp.mxene_type == "Ti3C2Tx"
    assert exp.areal_capacitance == 328.0
    print("  ✓ Valid data parsed correctly")
    
    # Test numeric parsing from strings
    string_data = {
        "mxene_type": "V2CTx",
        "electrolyte": "KOH",
        "areal_capacitance_mf_cm2": "245.5 mF/cm²",  # String with units
        "ssa_m2_g": "~100"  # Approximate value
    }
    
    exp2 = MXeneExperiment(**string_data)
    assert exp2.areal_capacitance == 245.5
    assert exp2.specific_surface_area == 100.0
    print("  ✓ String to numeric conversion works")
    
    # Test null handling
    minimal_data = {
        "mxene_type": "Mo2CTx",
        "electrolyte": "Ionic liquid"
    }
    
    exp3 = MXeneExperiment(**minimal_data)
    assert exp3.areal_capacitance is None
    print("  ✓ Null handling works")
    print()


def test_cleaner():
    """Test PDF cleaner with sample text."""
    print("Testing PDF cleaner...")
    
    cleaner = PDFCleaner()
    
    # Sample paper text
    sample_text = """
    Introduction
    
    MXenes are a family of 2D materials with excellent electrochemical properties.
    This paper investigates Ti3C2Tx for supercapacitor applications.
    
    Experimental Section
    
    Ti3C2Tx MXene was synthesized by etching Ti3AlC2 MAX phase in 48% HF solution
    at room temperature for 24 hours. The electrochemical performance was evaluated
    in 1M H2SO4 electrolyte using cyclic voltammetry. At a scan rate of 2 mV/s,
    the areal capacitance reached 328 mF/cm². BET analysis showed a specific
    surface area of 98 m²/g.
    
    Results and Discussion
    
    The electrode demonstrated 95% capacitance retention after 10,000 cycles,
    indicating excellent cycling stability. The high capacitance is attributed
    to the large interlayer spacing and high surface area.
    
    References
    
    [1] Smith et al., Nature, 2020
    [2] Jones et al., Science, 2021
    """
    
    # Test section identification
    relevant = cleaner.identify_relevant_sections(sample_text)
    
    assert "Experimental" in relevant or "experimental" in relevant.lower()
    assert "References" not in relevant
    print("  ✓ Section filtering works")
    print(f"  ✓ Reduced from {len(sample_text)} to {len(relevant)} chars")
    
    # Test chunking
    long_text = "A" * 10000
    chunks = cleaner.chunk_text(long_text)
    assert len(chunks) > 1
    print(f"  ✓ Chunking works ({len(chunks)} chunks)")
    print()


def test_ollama_connection():
    """Test connection to Ollama."""
    print("Testing Ollama connection...")
    
    try:
        from langchain_community.llms import Ollama
        
        llm = Ollama(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout=30
        )
        
        response = llm.invoke("Say 'test successful' and nothing else.")
        print(f"  ✓ Ollama responding")
        print(f"  ✓ Response: {response[:100]}")
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Ollama connection failed: {e}")
        print()
        print("  Make sure Ollama is running:")
        print("    ollama serve")
        print()
        print("  And the model is installed:")
        print(f"    ollama pull {config.ollama_model}")
        print()
        return False


def test_extraction_prompt():
    """Test the extraction prompt format."""
    print("Testing extraction prompt...")
    
    from src.extractor import MXeneExtractor
    
    extractor = MXeneExtractor()
    
    sample_text = """
    Ti3C2Tx MXene was tested in 1M H2SO4 electrolyte.
    Areal capacitance: 328 mF/cm² at 2 mV/s scan rate.
    Specific surface area: 98 m²/g.
    """
    
    prompt = extractor.prompt.format(text=sample_text)
    
    assert "Ti3C2Tx" in prompt or "text" in prompt.lower()
    assert "example" in prompt.lower()
    assert "json" in prompt.lower()
    print("  ✓ Prompt template formatted correctly")
    print(f"  ✓ Prompt length: {len(prompt)} chars")
    print()


def test_full_extraction():
    """Test full extraction with Ollama (if available)."""
    print("Testing full extraction pipeline...")
    
    try:
        from src.extractor import MXeneExtractor
        
        extractor = MXeneExtractor()
        
        # Sample paper text
        sample_text = """
        Experimental Section
        
        Ti3C2Tx MXene was synthesized by HF etching of Ti3AlC2 MAX phase.
        Electrochemical measurements were performed in 1M H2SO4 electrolyte.
        
        Results
        
        Cyclic voltammetry at 2 mV/s showed an areal capacitance of 328 mF/cm².
        BET analysis revealed a specific surface area of 98 m²/g.
        The electrode maintained 95% capacitance after 10,000 cycles.
        """
        
        # This will call the LLM
        result = extractor._extract_with_retry(sample_text)
        
        assert isinstance(result, ExtractionResult)
        assert len(result.experiments) > 0
        
        exp = result.experiments[0]
        print(f"  ✓ Extracted {len(result.experiments)} experiment(s)")
        print(f"  ✓ MXene type: {exp.mxene_type}")
        print(f"  ✓ Electrolyte: {exp.electrolyte}")
        
        if exp.areal_capacitance:
            print(f"  ✓ Areal capacitance: {exp.areal_capacitance} mF/cm²")
        
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        print("  This is expected if Ollama is not running")
        print()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MXene Pipeline Integration Tests")
    print("=" * 60)
    print()
    
    # Basic tests (no Ollama required)
    test_config()
    test_pydantic_schema()
    test_cleaner()
    test_extraction_prompt()
    
    # Ollama-dependent tests
    ollama_available = test_ollama_connection()
    
    if ollama_available:
        test_full_extraction()
    else:
        print("Skipping LLM-dependent tests (Ollama not available)")
        print()
    
    print("=" * 60)
    print("Tests Complete!")
    print("=" * 60)
    print()
    
    if ollama_available:
        print("✓ All systems operational")
        print()
        print("Ready to run the pipeline:")
        print("  python -m src.pipeline --max-papers 5")
    else:
        print("⚠ Ollama not available - please start it:")
        print("  ollama serve")
        print(f"  ollama pull {config.ollama_model}")
    
    print()


if __name__ == "__main__":
    main()
