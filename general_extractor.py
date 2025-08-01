#!/usr/bin/env python3
"""
General Text-to-JSON Extraction System
Automatically handles PDF and text inputs with unified processing pipeline
"""

import json
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Union, Dict, Any
from app import TextToJSONExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_input_type(file_path: str) -> str:
    """
    Detect if input is PDF or text based on file extension
    
    Args:
        file_path: Path to the input file
    
    Returns:
        str: 'pdf' or 'text'
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.pdf':
        return 'pdf'
    elif extension in ['.txt', '.md', '.json']:
        return 'text'
    else:
        # Try to read as text first, if fails assume PDF
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(100)  # Read first 100 chars
            return 'text'
        except UnicodeDecodeError:
            return 'pdf'

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file using multiple libraries
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        str: Extracted text from PDF
    """
    try:
        # Try PyMuPDF first (most reliable)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text.strip()
        except ImportError:
            pass
        
        # Try pdfplumber (better for complex layouts)
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            pass
        
        # Try PyPDF2 (basic text extraction)
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except ImportError:
            pass
        
        raise ImportError("No PDF library found. Install one of: PyMuPDF, pdfplumber, or PyPDF2")
        
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def read_text_file(file_path: str) -> str:
    """
    Read text from a text file
    
    Args:
        file_path: Path to the text file
    
    Returns:
        str: Content of the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise Exception(f"Error reading text file: {str(e)}")

async def extract_from_file(input_file_path: str, schema_file_path: str, 
                           output_file_path: str = None, 
                           confidence_threshold: float = 0.7,
                           api_key: str = None) -> dict:
    """
    Extract structured data from file (PDF or text) using schema file
    
    Args:
        input_file_path: Path to the input file (PDF or text)
        schema_file_path: Path to the .json file containing the schema
        output_file_path: Path to save the extraction result (optional)
        confidence_threshold: Minimum confidence score (0.0-1.0)
        api_key: Claude API key (if not provided, will look for it in config)
    
    Returns:
        dict: Extraction result with metadata
    """
    
    # Validate input files
    if not Path(input_file_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    if not Path(schema_file_path).exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file_path}")
    
    # Detect input type
    input_type = detect_input_type(input_file_path)
    logger.info(f"Detected input type: {input_type}")
    
    # Load API key
    if not api_key:
        try:
            from config import settings
            api_key = settings.validate_api_key()
        except ImportError:
            raise ValueError("No API key provided and config.py not found")
        except ValueError as e:
            raise ValueError(f"API key configuration error: {str(e)}")
    
    if not api_key or api_key == "":
        raise ValueError("No API key provided. Please set CLAUDE_API_KEY in .env file or pass --api-key")
    
    # Extract text based on input type
    if input_type == 'pdf':
        logger.info(f"Extracting text from PDF: {input_file_path}")
        text = extract_text_from_pdf(input_file_path)
        logger.info(f"PDF text extraction completed. Length: {len(text)} characters")
    else:
        logger.info(f"Reading text file: {input_file_path}")
        text = read_text_file(input_file_path)
        logger.info(f"Text file reading completed. Length: {len(text)} characters")
    
    # Read schema
    logger.info(f"Reading schema file: {schema_file_path}")
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    
    logger.info(f"Schema type: {schema.get('type', 'unknown')}")
    
    # Initialize extractor
    logger.info("Initializing extraction system...")
    extractor = TextToJSONExtractor(api_key)
    
    # Run extraction
    logger.info("Starting extraction process...")
    start_time = datetime.now()
    
    result = await extractor.extract(text, schema, confidence_threshold)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Extraction completed in {processing_time:.2f} seconds")
    
    # Add file metadata
    result['metadata']['input_files'] = {
        'input_file': input_file_path,
        'input_type': input_type,
        'schema_file': schema_file_path,
        'text_length': len(text),
        'schema_complexity': result['metadata']['processing_stats']['complexity_score']
    }
    
    # Save result
    if output_file_path:
        logger.info(f"Saving result to: {output_file_path}")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Input type: {input_type}")
    logger.info(f"Overall confidence: {result['metadata']['overall_confidence']:.2f}")
    logger.info(f"Processing time: {result['metadata']['total_processing_time']:.2f}s")
    logger.info(f"Strategy used: {result['metadata']['processing_stats']['strategy_used']}")
    logger.info(f"API calls: {result['metadata']['processing_stats']['api_calls']}")
    logger.info(f"Tokens processed: {result['metadata']['processing_stats']['total_tokens']}")
    
    if result['metadata']['human_review_required']:
        logger.info(f"Human review required for: {result['metadata']['human_review_required']}")
    
    if result['metadata']['validation_errors']:
        logger.warning(f"Validation errors: {len(result['metadata']['validation_errors'])}")
        for error in result['metadata']['validation_errors'][:3]:  # Show first 3
            logger.warning(f"  - {error}")
    
    return result

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='General File-to-JSON Extraction (PDF/Text)')
    parser.add_argument('--input-file', required=True, 
                       help='Path to input file (.pdf, .txt, .md, .json)')
    parser.add_argument('--schema-file', required=True, 
                       help='Path to JSON schema file (.json)')
    parser.add_argument('--output-file', 
                       help='Path to save output JSON (default: extraction_result.json)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, 
                       help='Confidence threshold (0.0-1.0, default: 0.7)')
    parser.add_argument('--api-key', 
                       help='Claude API key (optional, will use config.py if not provided)')
    
    args = parser.parse_args()
    
    # Set default output file
    if not args.output_file:
        args.output_file = "extraction_result.json"
    
    try:
        result = asyncio.run(extract_from_file(
            input_file_path=args.input_file,
            schema_file_path=args.schema_file,
            output_file_path=args.output_file,
            confidence_threshold=args.confidence_threshold,
            api_key=args.api_key
        ))
        
        logger.info(f"✅ Extraction completed successfully!")
        logger.info(f"📁 Result saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 


