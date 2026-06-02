"""LLM-based data extraction with Pydantic validation."""

import json
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from .cleaner import PDFCleaner
from .config import config


logger = logging.getLogger(__name__)


class MXeneExperiment(BaseModel):
    """Structured schema for MXene experimental data."""
    
    mxene_type: Optional[str] = Field(
        default=None,
        description="Type of MXene material (e.g., Ti3C2Tx, V2CTx, Mo2CTx)"
    )
    
    electrolyte: Optional[str] = Field(
        default=None,
        description="Electrolyte composition (e.g., H2SO4, KOH, ionic liquid)"
    )
    
    areal_capacitance: Optional[float] = Field(
        default=None,
        alias="areal_capacitance_mf_cm2",
        description="Areal capacitance in mF/cm² at specific scan rate"
    )
    
    volumetric_capacitance: Optional[float] = Field(
        default=None,
        alias="volumetric_capacitance_f_cm3",
        description="Volumetric capacitance in F/cm³"
    )
    
    specific_capacitance: Optional[float] = Field(
        default=None,
        alias="specific_capacitance_f_g",
        description="Specific capacitance in F/g"
    )
    
    specific_surface_area: Optional[float] = Field(
        default=None,
        alias="ssa_m2_g",
        description="Specific surface area in m²/g from BET analysis"
    )
    
    annealing_temp: Optional[float] = Field(
        default=None,
        alias="annealing_temperature_c",
        description="Annealing temperature in Celsius"
    )
    
    scan_rate: Optional[float] = Field(
        default=None,
        alias="scan_rate_mv_s",
        description="Scan rate in mV/s for CV measurements"
    )
    
    current_density: Optional[float] = Field(
        default=None,
        alias="current_density_a_g",
        description="Current density in A/g for GCD measurements"
    )
    
    cycling_stability: Optional[float] = Field(
        default=None,
        description="Capacitance retention after cycling (percentage)"
    )
    
    num_cycles: Optional[int] = Field(
        default=None,
        description="Number of charge-discharge cycles tested"
    )
    
    synthesis_method: Optional[str] = Field(
        default=None,
        description="Synthesis method (e.g., HF etching, MILD, CVD)"
    )
    
    @field_validator('areal_capacitance', 'volumetric_capacitance', 
               'specific_capacitance', 'specific_surface_area',
               'annealing_temp', 'scan_rate', 'current_density',
               'cycling_stability', mode='before')
    @classmethod
    def parse_numeric(cls, v):
        """Parse numeric values from strings."""
        if v is None or v == "":
            return None
        
        if isinstance(v, (int, float)):
            return float(v)
        
        # Try to extract number from string
        if isinstance(v, str):
            import re
            # Remove common units and extract number
            v = v.replace(',', '').replace('~', '').replace('≈', '')
            match = re.search(r'[-+]?\d*\.?\d+', v)
            if match:
                return float(match.group())
        
        return None
    
    class Config:
        populate_by_name = True


class ExtractionResult(BaseModel):
    """Container for multiple experiments from a single paper."""
    
    experiments: List[MXeneExperiment] = Field(
        description="List of experimental data points extracted from the paper"
    )
    
    paper_title: Optional[str] = None
    paper_doi: Optional[str] = None
    extraction_confidence: Optional[str] = Field(
        default="medium",
        description="Confidence level: high, medium, or low"
    )


class MXeneExtractor:
    """Extracts structured data from papers using local LLM."""
    
    # One-shot example for grounding the model
    EXAMPLE_TEXT = """
    Experimental Section:
    Ti3C2Tx MXene was synthesized by etching Ti3AlC2 MAX phase in 48% HF solution.
    The electrochemical performance was evaluated in 1M H2SO4 electrolyte using
    cyclic voltammetry. At a scan rate of 2 mV/s, the areal capacitance reached
    328 mF/cm². BET analysis showed a specific surface area of 98 m²/g.
    The electrode demonstrated 95% capacitance retention after 10,000 cycles.
    """
    
    EXAMPLE_JSON = {
        "experiments": [{
            "mxene_type": "Ti3C2Tx",
            "electrolyte": "1M H2SO4",
            "areal_capacitance_mf_cm2": 328.0,
            "specific_surface_area_m2_g": 98.0,
            "scan_rate_mv_s": 2.0,
            "cycling_stability": 95.0,
            "num_cycles": 10000,
            "synthesis_method": "HF etching"
        }]
    }
    
    def __init__(self):
        # Initialize Ollama LLM
        self.llm = Ollama(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            temperature=config.ollama_temperature,
            timeout=config.ollama_timeout
        )
        
        # Setup Pydantic parser
        self.parser = PydanticOutputParser(pydantic_object=ExtractionResult)
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=self._build_prompt_template(),
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "example_text": self.EXAMPLE_TEXT,
                "example_json": json.dumps(self.EXAMPLE_JSON, indent=2)
            }
        )
        
        self.cleaner = PDFCleaner()
        
        # Setup logging
        log_file = config.logs_dir / "extractor.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _build_prompt_template(self) -> str:
        """Construct the extraction prompt with one-shot example."""
        return """You are a materials science data extraction expert specializing in MXene supercapacitor research.

Your task is to accurately extract structured experimental data from a full scientific paper.
Because you are reading the entire document at once, the materials used (like MXene type and electrolyte) are often mentioned in the "Materials and Methods" section, while the actual capacitance and cycling performance are mentioned later in "Results and Discussion". 
You MUST connect these pieces of information together for each extracted experiment. Do NOT leave mxene_type or electrolyte as null if they were mentioned anywhere in the paper for that experiment.

EXAMPLE INPUT:
{example_text}

EXAMPLE OUTPUT:
{example_json}

INSTRUCTIONS:
1. Extract ALL experimental data points about MXene electrochemical performance.
2. Ensure every experiment is as complete as possible by looking back at the synthesis/methods sections to fill in mxene_type and electrolyte.
3. If an explicit value is completely missing from the ENTIRE document, use null (not 0 or empty string).
4. Be precise with units - capacitance can be areal (mF/cm²), volumetric (F/cm³), or specific (F/g).
5. Extract multiple experiments if the paper tests various conditions (e.g., varying annealing temp or trying different electrolytes).
6. Convert all numeric values to numbers (not strings).

{format_instructions}

TEXT TO ANALYZE:
{text}

OUTPUT (valid JSON only):"""
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(min=2, max=10)
    )
    def _extract_with_retry(self, text: str) -> ExtractionResult:
        """Extract data with retry on parsing errors."""
        try:
            # Generate LLM response
            prompt_value = self.prompt.format(text=text)
            response = self.llm.invoke(prompt_value)
            
            # Parse with Pydantic
            result = self.parser.parse(response)
            return result
        
        except Exception as e:
            logger.warning(f"Extraction attempt failed: {e}")
            
            # Try correction prompt on second attempt
            correction_prompt = f"""
            The previous extraction had an error: {str(e)}
            
            Please provide a valid JSON response following this exact format:
            {self.parser.get_format_instructions()}
            
            TEXT:
            {text[:2000]}
            
            OUTPUT (valid JSON only):
            """
            
            response = self.llm.invoke(correction_prompt)
            result = self.parser.parse(response)
            return result
    
    def extract_from_text(self, text: str) -> Optional[ExtractionResult]:
        """
        Extract structured data from a raw text string.
        
        Args:
            text: Cleaned text to analyze
        
        Returns:
            ExtractionResult or None if extraction fails
        """
        try:
            result = self._extract_with_retry(text)
            return result
        except Exception as e:
            logger.error(f"Failed to extract from text: {e}")
            return None

    def extract_from_pdf(self, pdf_path: Path) -> Optional[ExtractionResult]:
        """
        Complete extraction pipeline for a single PDF.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            ExtractionResult or None if extraction fails
        """
        try:
            logger.info(f"Extracting from {pdf_path.name}")
            
            # Clean and prepare text
            processed = self.cleaner.process(pdf_path)
            
            # Use full text if small enough, otherwise first chunk
            if processed['cleaned_length'] <= config.max_context_length:
                text_to_analyze = processed['full_text']
            else:
                logger.warning(
                    f"Text too long ({processed['cleaned_length']} chars), "
                    f"using first chunk"
                )
                text_to_analyze = processed['chunks'][0]
            
            # Extract with LLM
            result = self._extract_with_retry(text_to_analyze)
            
            # Add metadata
            result.paper_doi = pdf_path.stem
            
            logger.info(
                f"Extracted {len(result.experiments)} experiments "
                f"from {pdf_path.name}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to extract from {pdf_path.name}: {e}")
            return None
    
    def save_result(self, result: ExtractionResult, output_path: Path):
        """Save extraction result to JSON."""
        with open(output_path, 'w') as f:
            json.dump(result.dict(by_alias=True), f, indent=2)
        
        logger.info(f"Saved result to {output_path}")


def main():
    """CLI entry point for testing extractor."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.extractor <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    extractor = MXeneExtractor()
    result = extractor.extract_from_pdf(pdf_path)
    
    if result:
        print(f"\n✓ Extracted {len(result.experiments)} experiments")
        print(json.dumps(result.dict(by_alias=True), indent=2))
    else:
        print("\n✗ Extraction failed")


if __name__ == "__main__":
    main()
