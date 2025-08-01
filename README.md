# File-Based Text-to-JSON Extraction System

This system extracts structured data from text files using JSON schemas and stores the results. It's built on top of the advanced Claude API-based extraction system.

## Features

- 📁 **File-based input**: Takes text from `.txt` files and schemas from `.json` files
- 🎯 **Adaptive processing**: Automatically chooses the best extraction strategy based on schema complexity
- 📊 **Comprehensive metadata**: Provides confidence scores, processing stats, and validation results
- 🔍 **Human review flags**: Identifies fields that may need manual review
- 💾 **Flexible output**: Saves results to JSON files with full metadata

## Quick Start

### 1. Setup

Make sure you have the required dependencies:

```bash
pip install aiohttp tiktoken
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

### 3. Run with Sample Files

Test the system with the provided sample files:

```bash
python test_extraction.py
```

This will run two test cases:
- Customer service interaction extraction
- Email conversation extraction

### 4. Use Your Own Files

```bash
python file_extractor.py \
  --text-file your_text.txt \
  --schema-file your_schema.json \
  --output-file result.json \
  --confidence-threshold 0.7
```

## File Structure

```
metaforms/
├── app.py                          # Main extraction system
├── file_extractor.py              # File-based extraction interface
├── test_extraction.py             # Test script with sample files
├── config.py                      # Configuration (API key)
├── sample_text.txt                # Sample customer service text
├── customer_service_schema.json   # Sample customer service schema
├── email_conversation.txt         # Sample email conversation
├── email_schema.json              # Sample email schema
└── README.md                      # This file
```

## Sample Files

### 1. Customer Service Sample

**Text File**: `sample_text.txt`
- Contains a customer service interaction report
- Includes customer info, interaction details, technical specs, and outcomes

**Schema File**: `customer_service_schema.json`
- Defines structure for customer data, interaction details, billing info, and support tickets
- Includes validation rules and required fields

### 2. Email Conversation Sample

**Text File**: `email_conversation.txt`
- Contains a multi-party email conversation about a project
- Includes technical requirements, timeline, budget, and team assignments

**Schema File**: `email_schema.json`
- Defines structure for conversation participants, messages, project details, and agreements
- Handles complex nested data with arrays and objects

## Usage Examples

### Command Line Usage

```bash
# Basic usage
python file_extractor.py --text-file input.txt --schema-file schema.json

# With custom output and confidence threshold
python file_extractor.py \
  --text-file input.txt \
  --schema-file schema.json \
  --output-file my_result.json \
  --confidence-threshold 0.8

# With API key override
python file_extractor.py \
  --text-file input.txt \
  --schema-file schema.json \
  --api-key "your-api-key-here"
```

### Programmatic Usage

```python
import asyncio
from file_extractor import extract_from_files

async def main():
    result = await extract_from_files(
        text_file_path="input.txt",
        schema_file_path="schema.json",
        output_file_path="result.json",
        confidence_threshold=0.7
    )
    
    print(f"Confidence: {result['metadata']['overall_confidence']:.2f}")
    print(f"Processing time: {result['metadata']['total_processing_time']:.2f}s")

asyncio.run(main())
```

## Output Format

The system generates a comprehensive JSON result with:

```json
{
  "extracted_data": {
    // The structured data extracted from the text
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
      "text_file": "input.txt",
      "schema_file": "schema.json",
      "text_length": 2048,
      "schema_complexity": 45
    }
  }
}
```

## Schema Design Tips

### 1. Simple Schema Example

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "number"},
    "email": {"type": "string"},
    "active": {"type": "boolean"}
  },
  "required": ["name", "email"]
}
```

### 2. Complex Schema Example

```json
{
  "type": "object",
  "properties": {
    "customer": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "contact": {
          "type": "object",
          "properties": {
            "email": {"type": "string"},
            "phone": {"type": "string"}
          }
        }
      }
    },
    "orders": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "items": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "product": {"type": "string"},
                "quantity": {"type": "number"},
                "price": {"type": "number"}
              }
            }
          }
        }
      }
    }
  }
}
```

### 3. Schema Features

- **Enums**: `"enum": ["option1", "option2", "option3"]`
- **Required fields**: `"required": ["field1", "field2"]`
- **Nested objects**: Use `"type": "object"` with `properties`
- **Arrays**: Use `"type": "array"` with `items`
- **Validation**: The system validates against your schema

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

## Performance Tips

1. **Text length**: For very long texts (>50KB), consider chunking
2. **Schema complexity**: Complex schemas take longer but provide better structure
3. **Confidence threshold**: Higher thresholds (0.8+) may require more API calls
4. **Caching**: Results are not cached by default - implement caching for repeated extractions

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Claude API key is set in `config.py`
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