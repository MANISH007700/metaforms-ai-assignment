# General Pipeline Flow Diagram

## Complete System Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Files   │    │  Schema File    │    │   API Key       │
│                 │    │                 │    │                 │
│ • PDF (.pdf)    │    │ resume_schema.  │    │ config.py or    │
│ • Text (.txt)   │    │ json            │    │ --api-key flag  │
│ • Markdown (.md)│    │                 │    │                 │
│ • JSON (.json)  │    │                 │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   general_extractor.py  │
                    │   (Unified Entry Point) │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Input Type Detection │
                    │                        │
                    │ • File extension check │
                    │ • Content analysis     │
                    │ • Auto-detect PDF/text │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Text Extraction      │
                    │                        │
                    │ PDF Input:             │
                    │ ├─ PyMuPDF (Primary)   │
                    │ ├─ pdfplumber (Backup) │
                    │ └─ PyPDF2 (Fallback)   │
                    │                        │
                    │ Text Input:            │
                    │ └─ Direct file read    │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Schema Analysis      │
                    │                        │
                    │ • Calculate complexity │
                    │ • Determine strategy   │
                    │ • Select approach      │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   AI Extraction        │
                    │   (Claude API)         │
                    │                        │
                    │ Strategy Selection:    │
                    │ • Simple (≤30)        │
                    │ • Medium (≤70)        │
                    │ • Complex (≤90)       │
                    │ • Ultra (>90)         │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Validation &         │
                    │   Confidence Scoring   │
                    │                        │
                    │ • Schema validation    │
                    │ • Confidence metrics   │
                    │ • Error detection      │
                    │ • Review flags         │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Output Generation    │
                    │                        │
                    │ • Structured JSON      │
                    │ • Processing metadata  │
                    │ • Confidence scores    │
                    │ • Input type info      │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Final Output         │
                    │                        │
                    │ extraction_result.json │
                    │ with input type info   │
                    └─────────────────────────┘
```

## Input Type Detection Logic

```
File Extension Check:
├── .pdf → PDF (extract text)
├── .txt, .md, .json → Text (direct read)
└── Other → Try text read, fallback to PDF
```

## Detailed Component Flow

### 1. Input Processing (Unified)
```
Input File → Type Detection → Text Extraction → Cleaned Text
```

### 2. Schema Processing
```
JSON Schema → Schema Validation → Complexity Analysis → Strategy Selection
```

### 3. AI Processing
```
Text + Schema → Prompt Engineering → Claude API → Structured Data
```

### 4. Validation & Output
```
Structured Data → Schema Validation → Confidence Scoring → Final JSON
```

## Strategy Selection Logic

```
Complexity Score → Strategy Choice
    0-30     → Single Pass
   31-70     → Multi-Pass Validation
   71-90     → Hierarchical Processing
   90+       → Decomposed Parallel
```

## Error Handling Flow

```
Error Detected → Error Type Classification → Recovery Attempt → Fallback → Final Result
```


### Command Line (Unified):
```bash
# PDF Input
python general_extractor.py \
  --input-file Manish-Sharma-Resume-2025.pdf \
  --schema-file resume_schema.json \
  --output-file pdf_result.json

# Text Input
python general_extractor.py \
  --input-file sample_text.txt \
  --schema-file customer_service_schema.json \
  --output-file text_result.json
```

## Data Flow Summary

1. **Input**: Any file (PDF/text) + JSON schema + API key
2. **Processing**: Auto-detect type → Extract text → Schema analysis → AI extraction → Validation
3. **Output**: Structured JSON with metadata, confidence scores, and input type info

This unified pipeline ensures robust, accurate, and scalable processing for any input type with comprehensive error handling and detailed reporting. 