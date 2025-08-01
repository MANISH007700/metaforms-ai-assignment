# MetaForms AI Assignment: Universal File-to-JSON Extraction System

## Overview

This assignment implements a universal file processing system that automatically extracts structured JSON data from any file type (PDF, text, markdown) using Claude AI. The system intelligently detects file types and applies optimal processing strategies based on schema complexity.

## System Architecture

```
Input Files → Type Detection → Text Extraction → Schema Analysis → AI Extraction → Validation → JSON Output
     ↓              ↓              ↓              ↓              ↓              ↓              ↓
  Any File    Auto-Detect    Raw Text     Complexity    Structured    Confidence    Final JSON
  (.pdf/.txt/  File Type    Extraction   Analysis      Data         Metrics       with Metadata
   .md/.json)
```

## Key Components

### 1. **general_extractor.py** - Main Entry Point
- **Universal Interface**: Handles PDF, TXT, MD, JSON files
- **Auto-Detection**: Determines file type automatically
- **Smart Processing**: Routes to appropriate extraction method

### 2. **app.py** - Core Engine
- **TextToJSONExtractor**: Main extraction logic
- **SchemaAnalyzer**: Calculates complexity and selects strategy
- **ClaudeAPIClient**: AI-powered data extraction

### 3. **config.py** - Configuration
- **Environment Variables**: Loads API keys from `.env`
- **Settings Management**: Processing parameters

## Complexity Calculation & Strategy Selection

### Complexity Score Formula:
```
Complexity = (Depth × 10) + (Objects × 1) + (Fields × 0.5) + (Enums × 0.3) + (Required × 2)
```

### Strategy Selection:
- **0-30**: Single Pass (Simple schemas)
- **31-70**: Multi-Pass Validation (Medium complexity)
- **71-90**: Hierarchical Processing (Complex nested data)
- **90+**: Decomposed Parallel (Ultra-complex schemas)

### Example Complexity Calculation:
```json
{
  "personal_info": {
    "name": "string",
    "contact": {
      "email": "string",
      "phone": "string"
    }
  },
  "experience": [
    {
      "company": "string",
      "position": "string"
    }
  ]
}
```
**Complexity Score**: 42 (Depth: 3, Objects: 4, Fields: 6, Required: 2)
**Strategy**: Multi-Pass Validation

## How to Run

### 1. Setup Environment
```bash
# Install dependencies
pip install aiohttp tiktoken PyMuPDF pdfplumber PyPDF2 python-dotenv

# Create .env file
echo "CLAUDE_API_KEY=your-actual-api-key" > .env
```

### 2. Run Extraction
```bash
# PDF Resume
python general_extractor.py \
  --input-file resume.pdf \
  --schema-file resume_schema.json \
  --output-file result.json

# Text File
python general_extractor.py \
  --input-file document.txt \
  --schema-file schema.json \
  --output-file result.json
```

### 3. Programmatic Usage
```python
import asyncio
from general_extractor import extract_from_file

async def main():
    result = await extract_from_file(
        input_file_path="any_file.pdf",
        schema_file_path="schema.json",
        output_file_path="result.json"
    )
    print(f"Input type: {result['metadata']['input_files']['input_type']}")
    print(f"Confidence: {result['metadata']['overall_confidence']:.2f}")

asyncio.run(main())
```

## Pipeline Flow

### Stage 1: Input Detection
- **File Extension Check**: `.pdf` → PDF, `.txt/.md/.json` → Text
- **Content Analysis**: Try text read, fallback to PDF
- **Library Selection**: PyMuPDF → pdfplumber → PyPDF2

### Stage 2: Text Extraction
- **PDF**: Multi-library extraction with fallback
- **Text**: Direct UTF-8 file reading
- **Cleaning**: Normalize whitespace and formatting

### Stage 3: Schema Analysis
- **Complexity Calculation**: Depth, objects, fields, enums, required
- **Strategy Selection**: Based on complexity score
- **Processing Plan**: Single pass, multi-pass, hierarchical, or parallel

### Stage 4: AI Extraction
- **Claude API**: Anthropic's Claude 3.5 Sonnet
- **Prompt Engineering**: Schema-based extraction prompts
- **Strategy Execution**: Apply selected processing strategy

### Stage 5: Validation & Output
- **Schema Validation**: Ensure extracted data matches schema
- **Confidence Scoring**: Calculate confidence per field
- **Metadata Generation**: Processing stats, errors, input type info

## Output Format

```json
{
  "extracted_data": {
    "personal_info": { "name": "John Doe", "email": "john@email.com" },
    "experience": [ { "company": "Tech Corp", "position": "Developer" } ]
  },
  "metadata": {
    "overall_confidence": 0.85,
    "processing_stats": {
      "strategy_used": "multi_pass_validation",
      "complexity_score": 42,
      "processing_time": 5.2
    },
    "input_files": {
      "input_type": "pdf",
      "text_length": 15000
    }
  }
}
```

## Key Features

- **🎯 Universal Input**: Handles PDF, TXT, MD, JSON automatically
- **🔍 Smart Detection**: Auto-detects file type and processing method
- **📊 Complexity-Based**: Chooses optimal strategy based on schema complexity
- **✅ Validation**: Comprehensive schema validation and confidence scoring
- **🛡️ Error Handling**: Robust error recovery and detailed reporting
- **📈 Performance**: Token management, chunking, and memory optimization

## File Structure

```
metaforms-ai-assignment/
├── general_extractor.py    # 🚀 MAIN ENTRY POINT
├── app.py                  # Core extraction engine
├── config.py               # Configuration with .env
├── req.txt                 # Dependencies
├── inputs/                 # Input files directory
├── outputs/                # Output files directory
└── README.md               # Detailed documentation
```

## Thought Process: Complexity Calculation

The complexity calculation considers multiple factors that affect processing difficulty:

1. **Depth (×10)**: Nested objects require more processing steps
2. **Objects (×1)**: Each object increases extraction complexity
3. **Fields (×0.5)**: More fields mean more data to extract
4. **Enums (×0.3)**: Enum validation adds complexity
5. **Required (×2)**: Required fields are critical and need higher confidence

This weighted approach ensures that complex nested schemas with many required fields get the most sophisticated processing strategies, while simple flat schemas use fast single-pass extraction.

The system automatically adapts its processing approach based on this calculated complexity, ensuring optimal performance and accuracy for any schema structure. 