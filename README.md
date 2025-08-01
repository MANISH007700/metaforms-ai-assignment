# Universal File-to-JSON Extraction System

This system automatically extracts structured data from any file type (PDF, text, markdown, etc.) using JSON schemas and stores the results. It's built on top of the advanced Claude API-based extraction system with intelligent file type detection.

## Features

- 🎯 **Universal input**: Automatically handles PDF, TXT, MD, JSON files
- 🔍 **Smart detection**: Auto-detects file type and processes accordingly
- 🎯 **Adaptive processing**: Automatically chooses the best extraction strategy based on schema complexity
- 📊 **Comprehensive metadata**: Provides confidence scores, processing stats, and validation results
- 🔍 **Human review flags**: Identifies fields that may need manual review
- 💾 **Flexible output**: Saves results to JSON files with full metadata

## Quick Start

### 1. Setup

Make sure you have the required dependencies:

```bash
pip install aiohttp tiktoken PyMuPDF pdfplumber PyPDF2 python-dotenv
```

### 2. Configure API Key

**Option A: Manual Setup**
Create a `.env` file in the project directory and add your Claude API key:

```bash
# Create .env file
echo "CLAUDE_API_KEY=your-actual-claude-api-key-here" > .env
```

Or manually create `.env` file with:
```
CLAUDE_API_KEY=your-actual-claude-api-key-here
```


### 3. Use the Universal Extractor

The `general_extractor.py` is your main entry point for all file types:

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

# Markdown File
python general_extractor.py \
  --input-file notes.md \
  --schema-file schema.json \
  --output-file result.json
```

## File Structure

```
metaforms-ai-assignment/
├── general_extractor.py           # 🚀 MAIN ENTRY POINT
├── app.py                         # Core extraction engine
├── config.py                      # Configuration with .env
├── req.txt                        # Dependencies
├── inputs/                        # Input files directory
├── outputs/                       # Output files directory
├── PIPELINE.md                    # Detailed pipeline doc
├── pipeline_diagram.md            # Visual flow diagrams
└── README.md                      # This file
```

## Supported File Types

### 📄 PDF Files (.pdf)
- **Automatic text extraction** using multiple libraries
- **Fallback system**: PyMuPDF → pdfplumber → PyPDF2
- **Complex layout handling** for resumes, documents, reports

### 📝 Text Files (.txt, .md, .json)
- **Direct file reading** for immediate processing
- **UTF-8 encoding** support
- **Markdown parsing** for structured documents

### 🔄 Auto-Detection Logic
```
File Extension Check:
├── .pdf → PDF (extract text)
├── .txt, .md, .json → Text (direct read)
└── Other → Try text read, fallback to PDF
```

## Usage Examples

### Command Line Usage

```bash
# Basic usage with any file type
python general_extractor.py --input-file input.pdf --schema-file schema.json

# With custom output and confidence threshold
python general_extractor.py \
  --input-file input.txt \
  --schema-file schema.json \
  --output-file my_result.json \
  --confidence-threshold 0.8

# With API key override
python general_extractor.py \
  --input-file input.md \
  --schema-file schema.json \
  --api-key "your-api-key-here"
```

## Output Format

The system generates a comprehensive JSON result with:

```json
{
  "extracted_data": {
    // The structured data extracted from the file
  },
  "metadata": {
    "processing_stats": {
      "total_tokens": 1234,
      "api_calls": 2,
      "processing_time": 5.67,
      "complexity_score": 45,
      "strategy_used": "multi_pass_validation",
      "success_rate": 0.85
    },
    "confidence_metrics": [
      {
        "field_path": "customer.name",
        "confidence_score": 0.95,
        "extraction_method": "direct_match",
        "validation_passed": true,
        "human_review_required": false
      }
    ],
    "validation_errors": [],
    "human_review_required": [],
    "overall_confidence": 0.85,
    "total_processing_time": 5.67,
    "timestamp": "2024-03-15T10:30:00",
    "input_files": {
      "input_file": "resume.pdf",
      "input_type": "pdf",  // or "text"
      "schema_file": "schema.json",
      "text_length": 2048,
      "schema_complexity": 45
    }
  }
}
```

## Processing Strategies

The system automatically chooses the best processing strategy:

1. **Single Pass** (Complexity ≤ 30): Simple, fast extraction
2. **Multi-Pass Validation** (Complexity ≤ 70): Extraction with validation and refinement
3. **Hierarchical Processing** (Complexity ≤ 90): Complex nested schema handling
4. **Decomposed Parallel** (Complexity > 90): Ultra-complex schema with parallel processing

## Error Handling

The system provides comprehensive error handling:

- **File not found**: Clear error messages for missing input files
- **Invalid schema**: JSON schema validation
- **API errors**: Network and Claude API error handling
- **Validation errors**: Schema compliance checking
- **Human review flags**: Fields that may need manual review
- **File type errors**: Automatic fallback for unsupported formats

## Performance Tips

1. **File size**: For very large files (>50KB), consider chunking
2. **Schema complexity**: Complex schemas take longer but provide better structure
3. **Confidence threshold**: Higher thresholds (0.8+) may require more API calls
4. **Caching**: Results are not cached by default - implement caching for repeated extractions

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Claude API key is set in `.env` file
2. **File Encoding**: Ensure text files are UTF-8 encoded
3. **Schema Validation**: Check that your JSON schema is valid
4. **Memory Issues**: For very large files, consider processing in chunks

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is part of the MetaForms text extraction system.

## Support

For issues or questions:
1. Check the error messages in the console output
2. Verify your input files and schema format
3. Ensure your API key is valid and has sufficient credits
