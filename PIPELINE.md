# Pipeline Documentation: PDF Resume to Structured JSON

## Overview

This document explains the complete pipeline for converting unstructured PDF resumes into structured JSON data using Claude AI. The system follows a multi-stage process that handles PDF text extraction, schema analysis, AI-powered extraction, and validation.

## Pipeline Architecture

```
PDF Input → Text Extraction → Schema Analysis → AI Extraction → Validation → JSON Output
     ↓              ↓              ↓              ↓              ↓              ↓
  PDF File    Raw Text     Complexity    Structured    Confidence    Final JSON
              Extraction    Analysis      Data         Metrics       with Metadata
```

## Stage 1: PDF Input & Text Extraction

### Components:
- **`pdf_extractor.py`** - Main entry point for PDF processing
- **PDF Libraries** - PyMuPDF, PyPDF2, pdfplumber (fallback chain)

### Process:
1. **File Validation**: Check if PDF file exists and is accessible
2. **Library Detection**: Try multiple PDF libraries in order of reliability:
   - PyMuPDF (fitz) - Most reliable for complex layouts
   - pdfplumber - Good for tabular data
   - PyPDF2 - Basic text extraction
3. **Text Extraction**: Extract raw text from all PDF pages
4. **Text Cleaning**: Remove extra whitespace and normalize formatting

### Code Flow:
```python
def extract_text_from_pdf(pdf_path: str) -> str:
    # Try PyMuPDF first (most reliable)
    # Fallback to pdfplumber
    # Fallback to PyPDF2
    # Return cleaned text
```

## Stage 2: Schema Analysis & Strategy Selection

### Components:
- **`app.py`** - Contains `SchemaAnalyzer` class
- **Complexity Calculation** - Determines processing strategy

### Process:
1. **Schema Parsing**: Load and validate JSON schema
2. **Complexity Analysis**: Calculate schema complexity score based on:
   - Nesting depth
   - Number of objects and fields
   - Enum fields and required fields
   - Field type distribution
3. **Strategy Selection**: Choose extraction strategy based on complexity:
   - **Simple** (≤30): Single pass extraction
   - **Medium** (≤70): Multi-pass with validation
   - **Complex** (≤90): Hierarchical processing
   - **Ultra-complex** (>90): Decomposed parallel processing

### Code Flow:
```python
class SchemaAnalyzer:
    def analyze_schema(self, schema: Dict) -> Dict:
        # Calculate depth, objects, fields, enums
        # Determine complexity score
        # Select processing strategy
```

## Stage 3: AI-Powered Extraction

### Components:
- **`app.py`** - Contains `TextToJSONExtractor` and `ClaudeAPIClient`
- **Claude API** - Anthropic's Claude 3.5 Sonnet model

### Process:
1. **API Initialization**: Set up Claude API client with authentication
2. **Prompt Engineering**: Build extraction prompts based on schema
3. **Strategy Execution**: Run selected extraction strategy:

#### Strategy 1: Single Pass (Simple Schemas)
```python
async def _single_pass_extraction(self, text, schema, threshold):
    # Build simple extraction prompt
    # Single API call to Claude
    # Basic validation
    # Return results
```

#### Strategy 2: Multi-Pass Validation (Medium Schemas)
```python
async def _multi_pass_extraction(self, text, schema, threshold):
    # Initial extraction
    # Validation against schema
    # Refinement pass for low-confidence fields
    # Final validation
    # Return results with confidence metrics
```

#### Strategy 3: Hierarchical Processing (Complex Schemas)
```python
async def _hierarchical_extraction(self, text, schema, threshold):
    # Decompose schema into components
    # Extract each component separately
    # Merge results hierarchically
    # Validate final structure
    # Return comprehensive results
```

#### Strategy 4: Decomposed Parallel (Ultra-Complex Schemas)
```python
async def _decomposed_extraction(self, text, schema, threshold):
    # Break schema into independent components
    # Parallel extraction of components
    # Merge and validate results
    # Handle dependencies between components
    # Return final structured data
```

## Stage 4: Validation & Confidence Scoring

### Components:
- **`app.py`** - Contains validation logic and confidence metrics
- **JSON Schema Validation** - Ensures extracted data matches schema

### Process:
1. **Schema Validation**: Check if extracted data conforms to JSON schema
2. **Confidence Calculation**: Calculate confidence scores for each field:
   - **Direct Match**: High confidence for exact matches
   - **Inferred Match**: Medium confidence for logical inferences
   - **Low Confidence**: Fields requiring human review
3. **Error Detection**: Identify validation errors and missing required fields
4. **Human Review Flags**: Mark fields that may need manual verification

### Code Flow:
```python
def _validate_against_schema(self, data, schema):
    # JSON schema validation
    # Required field checking
    # Type validation
    # Return validation errors

def _calculate_extraction_confidence(self, data, schema):
    # Calculate confidence per field
    # Determine overall confidence
    # Flag low-confidence fields
```

## Stage 5: Metadata Generation & Output

### Components:
- **`pdf_extractor.py`** - Handles output formatting
- **Processing Statistics** - Tracks performance metrics

### Process:
1. **Metadata Assembly**: Compile comprehensive metadata:
   - Processing statistics (time, API calls, tokens)
   - Confidence metrics for each field
   - Validation results and errors
   - Input file information
2. **Output Structuring**: Create final JSON with:
   - `extracted_data`: Structured resume information
   - `metadata`: Processing details and confidence scores
3. **File Saving**: Save results to specified output file

### Output Structure:
```json
{
  "extracted_data": {
    "personal_info": { "name": "John Doe", "email": "..." },
    "experience": [ { "company": "...", "position": "..." } ],
    "education": [ { "institution": "...", "degree": "..." } ],
    "skills": { "technical_skills": ["Python", "JavaScript"] }
  },
  "metadata": {
    "overall_confidence": 0.85,
    "processing_stats": {
      "total_tokens": 1500,
      "api_calls": 3,
      "processing_time": 5.2,
      "strategy_used": "multi_pass_validation"
    },
    "confidence_metrics": [
      {
        "field_path": "personal_info.name",
        "confidence_score": 0.95,
        "extraction_method": "direct_match"
      }
    ],
    "validation_errors": [],
    "human_review_required": []
  }
}
```

## File Dependencies & Flow

### Core Files:
```
app.py (Core Engine)
├── TextToJSONExtractor
├── SchemaAnalyzer
├── ClaudeAPIClient
└── Validation Logic

pdf_extractor.py (PDF Interface)
├── extract_text_from_pdf()
├── extract_from_pdf()
└── Command-line interface

config.py (Configuration)
├── API key management
├── Processing settings
└── Environment variables
```

### Data Files:
```
Input:
├── Manish-Sharma-Resume-2025.pdf (Your resume)
└── resume_schema.json (Extraction schema)

Output:
├── resume_extraction_result.json (Structured data)
└── Processing metadata
```

## Error Handling & Resilience

### Error Types:
1. **File Errors**: Missing PDF or schema files
2. **PDF Extraction Errors**: Corrupted or password-protected PDFs
3. **API Errors**: Network issues or invalid API keys
4. **Schema Errors**: Invalid JSON schema format
5. **Validation Errors**: Extracted data doesn't match schema

### Recovery Mechanisms:
- **PDF Library Fallback**: Multiple PDF extraction libraries
- **API Retry Logic**: Automatic retries for transient failures
- **Graceful Degradation**: Continue processing with partial results
- **Detailed Error Reporting**: Comprehensive error messages and suggestions

## Performance Optimization

### Token Management:
- **Chunking**: Split large texts into manageable chunks
- **Token Counting**: Accurate token usage tracking
- **Overlap Handling**: Maintain context between chunks

### Strategy Optimization:
- **Complexity-Based Selection**: Choose optimal strategy for schema complexity
- **Parallel Processing**: For ultra-complex schemas
- **Caching**: (Optional) Cache results for repeated extractions

### Memory Management:
- **Streaming**: Process large PDFs without loading entire file
- **Garbage Collection**: Clean up temporary objects
- **Resource Limits**: Prevent memory overflow

## Monitoring & Debugging

### Logging Levels:
- **INFO**: General processing steps
- **DEBUG**: Detailed API calls and responses
- **WARNING**: Low confidence fields and validation errors
- **ERROR**: Processing failures and exceptions

### Metrics Tracked:
- Processing time per stage
- API call count and token usage
- Confidence scores distribution
- Validation error rates
- Strategy effectiveness

## Usage Examples

### Command Line:
```bash
python pdf_extractor.py \
  --pdf-file Manish-Sharma-Resume-2025.pdf \
  --schema-file resume_schema.json \
  --output-file my_resume_result.json \
  --confidence-threshold 0.7
```

### Programmatic:
```python
import asyncio
from pdf_extractor import extract_from_pdf

async def main():
    result = await extract_from_pdf(
        pdf_file_path="Manish-Sharma-Resume-2025.pdf",
        schema_file_path="resume_schema.json",
        output_file_path="result.json"
    )
    print(f"Confidence: {result['metadata']['overall_confidence']:.2f}")

asyncio.run(main())
```

## Pipeline Summary

The complete pipeline transforms your unstructured PDF resume into structured JSON through these key stages:

1. **PDF → Text**: Extract raw text using multiple PDF libraries
2. **Schema Analysis**: Determine optimal processing strategy
3. **AI Extraction**: Use Claude to extract structured data
4. **Validation**: Ensure data quality and schema compliance
5. **Output**: Generate comprehensive JSON with metadata

This architecture ensures robust, accurate, and scalable resume processing with detailed confidence scoring and error handling. 