#!/usr/bin/env python3
"""
Advanced Text-to-JSON Extraction System
Converts unstructured text to structured JSON following complex schemas
"""

import json
import re
import logging
import asyncio
import aiohttp
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import tiktoken
from datetime import datetime
import statistics
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConfidenceMetrics:
    """Confidence metrics for extracted fields"""
    field_path: str
    confidence_score: float
    extraction_method: str
    validation_passed: bool
    human_review_required: bool
    error_details: Optional[str] = None

@dataclass
class ProcessingStats:
    """Processing statistics and metrics"""
    total_tokens: int
    api_calls: int
    processing_time: float
    complexity_score: int
    strategy_used: str
    success_rate: float

class SchemaAnalyzer:
    """Analyzes JSON schema complexity and structure"""
    
    def __init__(self):
        self.complexity_weights = {
            'depth': 10,
            'objects': 1,
            'fields': 0.5,
            'enums': 0.3,
            'required_fields': 2
        }
    
    def analyze_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive schema analysis"""
        analysis = {
            'max_depth': self._calculate_depth(schema),
            'total_objects': self._count_objects(schema),
            'total_fields': self._count_fields(schema),
            'enum_fields': self._count_enums(schema),
            'required_fields': self._count_required(schema),
            'field_types': self._analyze_field_types(schema),
            'dependencies': self._find_dependencies(schema)
        }
        
        analysis['complexity_score'] = self._calculate_complexity_score(analysis)
        analysis['processing_strategy'] = self._determine_strategy(analysis['complexity_score'])
        
        return analysis
    
    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        if not isinstance(obj, dict):
            return current_depth
        
        max_depth = current_depth
        for value in obj.values():
            if isinstance(value, dict):
                depth = self._calculate_depth(value, current_depth + 1)
                max_depth = max(max_depth, depth)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                depth = self._calculate_depth(value[0], current_depth + 1)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_objects(self, obj: Any) -> int:
        """Count total nested objects"""
        if not isinstance(obj, dict):
            return 0
        
        count = 1  # Current object
        for value in obj.values():
            if isinstance(value, dict):
                count += self._count_objects(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        count += self._count_objects(item)
        
        return count
    
    def _count_fields(self, obj: Any) -> int:
        """Count total fields"""
        if not isinstance(obj, dict):
            return 0
        
        count = len(obj)
        for value in obj.values():
            if isinstance(value, dict):
                count += self._count_fields(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        count += self._count_fields(item)
        
        return count
    
    def _count_enums(self, obj: Any) -> int:
        """Count enum fields"""
        if not isinstance(obj, dict):
            return 0
        
        count = 0
        for key, value in obj.items():
            if key == 'enum' and isinstance(value, list):
                count += len(value)
            elif isinstance(value, dict):
                count += self._count_enums(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        count += self._count_enums(item)
        
        return count
    
    def _count_required(self, obj: Any) -> int:
        """Count required fields"""
        if not isinstance(obj, dict):
            return 0
        
        count = 0
        if 'required' in obj and isinstance(obj['required'], list):
            count += len(obj['required'])
        
        for value in obj.values():
            if isinstance(value, dict):
                count += self._count_required(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        count += self._count_required(item)
        
        return count
    def _analyze_field_types(self, obj: Any) -> Dict[str, int]:
        """Analyze field type distribution - simplified version"""
        types = {}
        
        def count_types(schema_obj):
            if isinstance(schema_obj, dict):
                if 'type' in schema_obj:
                    field_type = schema_obj['type']
                    if isinstance(field_type, str):
                        types[field_type] = types.get(field_type, 0) + 1
                
                for value in schema_obj.values():
                    if isinstance(value, (dict, list)):
                        count_types(value)
            elif isinstance(schema_obj, list):
                for item in schema_obj:
                    count_types(item)
        
        try:
            count_types(obj)
        except:
            types['unknown'] = 1
        
        return types

    # def _analyze_field_types(self, obj: Any) -> Dict[str, int]:
    #     """Analyze field type distribution"""
    #     types = {}
        
    #     def count_types(schema_obj):
    #         if isinstance(schema_obj, dict):
    #             if 'type' in schema_obj:
    #                 field_type = schema_obj['type']
    #                 types[field_type] = types.get(field_type, 0) + 1
                
    #             for value in schema_obj.values():
    #                 if isinstance(value, (dict, list)):
    #                     count_types(value)
    #         elif isinstance(schema_obj, list):
    #             for item in schema_obj:
    #                 count_types(item)
        
    #     count_types(obj)
    #     return types
    
    def _find_dependencies(self, obj: Any) -> List[str]:
        """Find field dependencies and references"""
        dependencies = []
        
        def find_refs(schema_obj, path=""):
            if isinstance(schema_obj, dict):
                if '$ref' in schema_obj:
                    dependencies.append(f"{path} -> {schema_obj['$ref']}")
                
                for key, value in schema_obj.items():
                    new_path = f"{path}.{key}" if path else key
                    find_refs(value, new_path)
            elif isinstance(schema_obj, list):
                for i, item in enumerate(schema_obj):
                    find_refs(item, f"{path}[{i}]")
        
        find_refs(obj)
        return dependencies
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate overall complexity score (0-100)"""
        score = 0
        score += min(analysis['max_depth'] * self.complexity_weights['depth'], 50)
        score += min(analysis['total_objects'] * self.complexity_weights['objects'], 20)
        score += min(analysis['total_fields'] * self.complexity_weights['fields'], 15)
        score += min(analysis['enum_fields'] * self.complexity_weights['enums'], 10)
        score += min(analysis['required_fields'] * self.complexity_weights['required_fields'], 5)
        
        return min(int(score), 100)
    
    def _determine_strategy(self, complexity_score: int) -> str:
        """Determine processing strategy based on complexity"""
        if complexity_score <= 30:
            return "single_pass"
        elif complexity_score <= 70:
            return "multi_pass_validation"
        elif complexity_score <= 90:
            return "hierarchical_processing"
        else:
            return "decomposed_parallel"

class TextProcessor:
    """Handles text preprocessing and chunking"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def chunk_text(self, text: str, max_tokens: int = 15000, overlap: int = 500) -> List[str]:
        """Intelligently chunk text while preserving context"""
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            
            # Try to end at sentence boundary
            chunk_text = self.encoding.decode(tokens[start:end])
            
            if end < len(tokens):
                # Find last sentence boundary
                sentences = re.split(r'[.!?]+', chunk_text)
                if len(sentences) > 1:
                    # Keep all but the last incomplete sentence
                    chunk_text = '.'.join(sentences[:-1]) + '.'
                    # Update end position
                    end = start + len(self.encoding.encode(chunk_text))
            
            chunks.append(chunk_text)
            start = max(end - overlap, start + 1)
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

class ClaudeAPIClient:
    """Handles Claude API interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20241022"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def extract_structured_data(self, text: str, schema: Dict[str, Any], 
                                    extraction_prompt: str = None) -> Tuple[Dict[str, Any], float]:
        """Extract structured data using Claude API"""
        
        if not extraction_prompt:
            extraction_prompt = self._build_extraction_prompt(schema)
        
        messages = [
            {
                "role": "user",
                "content": f"{extraction_prompt}\n\nInput Text:\n{text}\n\nPlease extract the data according to the schema and return only valid JSON."
            }
        ]
        
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 4000,
            "messages": messages
        }
        
        try:
            async with self.session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result['content'][0]['text']
                    
                    # Extract JSON from response
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        extracted_data = json.loads(json_match.group())
                        confidence = self._calculate_extraction_confidence(extracted_data, schema)
                        return extracted_data, confidence
                    else:
                        logger.error("No JSON found in Claude response")
                        return {}, 0.0
                else:
                    error_text = await response.text()
                    logger.error(f"Claude API error: {response.status} - {error_text}")
                    return {}, 0.0
                    
        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            return {}, 0.0
    
    def _build_extraction_prompt(self, schema: Dict[str, Any]) -> str:
        """Build extraction prompt based on schema"""
        return f"""You are an expert data extraction system. Extract information from the provided text according to this JSON schema:

{json.dumps(schema, indent=2)}

Rules:
1. Return only valid JSON that strictly follows the schema
2. Use null for missing values
3. Ensure all required fields are present
4. Follow enum constraints exactly
5. Maintain proper data types (strings, numbers, booleans, arrays)
6. Extract information accurately from the input text
7. If information is ambiguous or missing, use your best judgment based on context

The extracted JSON should be complete and valid."""
    
    def _calculate_extraction_confidence(self, extracted_data: Dict[str, Any], 
                                       schema: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted data"""
        total_fields = 0
        filled_fields = 0
        
        def count_fields(data, schema_part):
            nonlocal total_fields, filled_fields
            
            if isinstance(schema_part, dict) and 'properties' in schema_part:
                for field_name, field_schema in schema_part['properties'].items():
                    total_fields += 1
                    if field_name in data and data[field_name] is not None:
                        filled_fields += 1
                        if isinstance(data[field_name], dict) and 'properties' in field_schema:
                            count_fields(data[field_name], field_schema)
        
        count_fields(extracted_data, schema)
        
        return filled_fields / total_fields if total_fields > 0 else 0.0

class TextToJSONExtractor:
    """Main extraction system orchestrator"""
    
    def __init__(self, claude_api_key: str):
        self.claude_client = None
        self.claude_api_key = claude_api_key
        self.schema_analyzer = SchemaAnalyzer()
        self.text_processor = TextProcessor()
        self.extraction_history = []
    
    async def extract(self, text: str, schema: Dict[str, Any], 
                     confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Main extraction method"""
        start_time = datetime.now()
        
        # Phase 1: Input Analysis & Preprocessing
        logger.info("Phase 1: Analyzing input and preprocessing...")
        processed_text = self.text_processor.preprocess_text(text)
        input_tokens = self.text_processor.count_tokens(processed_text)
        
        # Phase 2: Schema Complexity Analysis
        logger.info("Phase 2: Analyzing schema complexity...")
        schema_analysis = self.schema_analyzer.analyze_schema(schema)
        logger.info(f"Schema complexity score: {schema_analysis['complexity_score']}")
        logger.info(f"Processing strategy: {schema_analysis['processing_strategy']}")
        
        # Phase 3: Adaptive Processing Strategy
        logger.info("Phase 3: Executing adaptive processing strategy...")
        extracted_data, confidence_metrics, processing_stats = await self._execute_extraction_strategy(
            processed_text, schema, schema_analysis, confidence_threshold
        )
        
        # Phase 4: Final Assembly and Validation
        logger.info("Phase 4: Final validation and assembly...")
        final_result = await self._finalize_extraction(
            extracted_data, schema, confidence_metrics, processing_stats, start_time
        )
        
        return final_result
    
    async def _execute_extraction_strategy(self, text: str, schema: Dict[str, Any], 
                                         schema_analysis: Dict[str, Any], 
                                         confidence_threshold: float) -> Tuple[Dict[str, Any], List[ConfidenceMetrics], ProcessingStats]:
        """Execute the determined processing strategy"""
        strategy = schema_analysis['processing_strategy']
        
        async with ClaudeAPIClient(self.claude_api_key) as client:
            self.claude_client = client
            
            if strategy == "single_pass":
                return await self._single_pass_extraction(text, schema, confidence_threshold)
            elif strategy == "multi_pass_validation":
                return await self._multi_pass_extraction(text, schema, confidence_threshold)
            elif strategy == "hierarchical_processing":
                return await self._hierarchical_extraction(text, schema, confidence_threshold)
            else:  # decomposed_parallel
                return await self._decomposed_extraction(text, schema, confidence_threshold)
    
    async def _single_pass_extraction(self, text: str, schema: Dict[str, Any], 
                                    confidence_threshold: float) -> Tuple[Dict[str, Any], List[ConfidenceMetrics], ProcessingStats]:
        """Simple single-pass extraction for low complexity schemas"""
        start_time = datetime.now()
        
        extracted_data, confidence = await self.claude_client.extract_structured_data(text, schema)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        confidence_metrics = [ConfidenceMetrics(
            field_path="root",
            confidence_score=confidence,
            extraction_method="single_pass",
            validation_passed=confidence >= confidence_threshold,
            human_review_required=confidence < confidence_threshold
        )]
        
        stats = ProcessingStats(
            total_tokens=self.text_processor.count_tokens(text),
            api_calls=1,
            processing_time=processing_time,
            complexity_score=30,
            strategy_used="single_pass",
            success_rate=confidence
        )
        
        return extracted_data, confidence_metrics, stats
    
    async def _multi_pass_extraction(self, text: str, schema: Dict[str, Any], 
                                   confidence_threshold: float) -> Tuple[Dict[str, Any], List[ConfidenceMetrics], ProcessingStats]:
        """Multi-pass extraction with validation"""
        start_time = datetime.now()
        api_calls = 0
        
        # First pass - initial extraction
        extracted_data, initial_confidence = await self.claude_client.extract_structured_data(text, schema)
        api_calls += 1
        
        # Second pass - validation and refinement
        if initial_confidence < confidence_threshold:
            validation_prompt = f"""Please review and improve this extracted data:

Original Text: {text[:2000]}...

Current Extraction: {json.dumps(extracted_data, indent=2)}

Schema: {json.dumps(schema, indent=2)}

Please provide an improved extraction that better matches the schema and source text."""
            
            refined_data, refined_confidence = await self.claude_client.extract_structured_data(
                validation_prompt, schema
            )
            api_calls += 1
            
            if refined_confidence > initial_confidence:
                extracted_data = refined_data
                initial_confidence = refined_confidence
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        confidence_metrics = [ConfidenceMetrics(
            field_path="root",
            confidence_score=initial_confidence,
            extraction_method="multi_pass_validation",
            validation_passed=initial_confidence >= confidence_threshold,
            human_review_required=initial_confidence < confidence_threshold
        )]
        
        stats = ProcessingStats(
            total_tokens=self.text_processor.count_tokens(text),
            api_calls=api_calls,
            processing_time=processing_time,
            complexity_score=50,
            strategy_used="multi_pass_validation",
            success_rate=initial_confidence
        )
        
        return extracted_data, confidence_metrics, stats
    
    async def _hierarchical_extraction(self, text: str, schema: Dict[str, Any], 
                                     confidence_threshold: float) -> Tuple[Dict[str, Any], List[ConfidenceMetrics], ProcessingStats]:
        """Hierarchical extraction for complex nested schemas"""
        start_time = datetime.now()
        api_calls = 0
        confidence_metrics = []
        
        # Break down schema into hierarchical levels
        schema_levels = self._decompose_schema_hierarchically(schema)
        
        final_data = {}
        total_confidence = []
        
        for level, level_schema in enumerate(schema_levels):
            level_data, level_confidence = await self.claude_client.extract_structured_data(
                text, level_schema
            )
            api_calls += 1
            
            # Merge level data into final result
            final_data = self._merge_hierarchical_data(final_data, level_data)
            total_confidence.append(level_confidence)
            
            confidence_metrics.append(ConfidenceMetrics(
                field_path=f"level_{level}",
                confidence_score=level_confidence,
                extraction_method="hierarchical_processing",
                validation_passed=level_confidence >= confidence_threshold,
                human_review_required=level_confidence < confidence_threshold
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        avg_confidence = statistics.mean(total_confidence) if total_confidence else 0.0
        
        stats = ProcessingStats(
            total_tokens=self.text_processor.count_tokens(text),
            api_calls=api_calls,
            processing_time=processing_time,
            complexity_score=80,
            strategy_used="hierarchical_processing",
            success_rate=avg_confidence
        )
        
        return final_data, confidence_metrics, stats
    
    async def _decomposed_extraction(self, text: str, schema: Dict[str, Any], 
                                   confidence_threshold: float) -> Tuple[Dict[str, Any], List[ConfidenceMetrics], ProcessingStats]:
        """Decomposed parallel extraction for ultra-complex schemas"""
        start_time = datetime.now()
        
        # For ultra-complex schemas, chunk both text and schema
        text_chunks = self.text_processor.chunk_text(text, max_tokens=10000)
        schema_components = self._decompose_schema_components(schema)
        
        extraction_tasks = []
        for chunk in text_chunks:
            for component_name, component_schema in schema_components.items():
                extraction_tasks.append(
                    self._extract_component(chunk, component_schema, component_name)
                )
        
        # Execute extractions in parallel (simulated with sequential for this demo)
        results = []
        async with ClaudeAPIClient(self.claude_api_key) as client:
            self.claude_client = client
            for task in extraction_tasks:
                result = await task
                results.append(result)
        
        # Merge all results
        final_data = self._merge_decomposed_results(results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate overall confidence
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0
        
        confidence_metrics = [ConfidenceMetrics(
            field_path="decomposed_root",
            confidence_score=avg_confidence,
            extraction_method="decomposed_parallel",
            validation_passed=avg_confidence >= confidence_threshold,
            human_review_required=avg_confidence < confidence_threshold
        )]
        
        stats = ProcessingStats(
            total_tokens=sum(self.text_processor.count_tokens(chunk) for chunk in text_chunks),
            api_calls=len(extraction_tasks),
            processing_time=processing_time,
            complexity_score=95,
            strategy_used="decomposed_parallel",
            success_rate=avg_confidence
        )
        
        return final_data, confidence_metrics, stats
    
    def _decompose_schema_hierarchically(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose complex schema into hierarchical levels"""
        levels = []
        
        # Extract top-level fields first
        if 'properties' in schema:
            top_level = {'type': 'object', 'properties': {}}
            nested_schemas = {}
            
            for field_name, field_schema in schema['properties'].items():
                if isinstance(field_schema, dict) and field_schema.get('type') == 'object':
                    # This is a nested object - save for next level
                    nested_schemas[field_name] = field_schema
                    # Add placeholder in top level
                    top_level['properties'][field_name] = {'type': 'object'}
                else:
                    # Simple field - add to top level
                    top_level['properties'][field_name] = field_schema
            
            levels.append(top_level)
            
            # Add nested levels
            for nested_name, nested_schema in nested_schemas.items():
                levels.append({
                    'type': 'object',
                    'properties': {nested_name: nested_schema}
                })
        
        return levels if levels else [schema]
    
    def _merge_hierarchical_data(self, base_data: Dict[str, Any], 
                               new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge hierarchical extraction results"""
        result = base_data.copy()
        
        for key, value in new_data.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_hierarchical_data(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _decompose_schema_components(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Decompose schema into manageable components"""
        components = {}
        
        if 'properties' in schema:
            for field_name, field_schema in schema['properties'].items():
                components[field_name] = {
                    'type': 'object',
                    'properties': {field_name: field_schema}
                }
        
        return components
    
    async def _extract_component(self, text: str, schema: Dict[str, Any], 
                               component_name: str) -> Dict[str, Any]:
        """Extract a single schema component"""
        extracted_data, confidence = await self.claude_client.extract_structured_data(text, schema)
        
        return {
            'component': component_name,
            'data': extracted_data,
            'confidence': confidence
        }
    
    def _merge_decomposed_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results from decomposed extraction"""
        final_data = {}
        
        # Group results by component and merge with highest confidence
        component_results = {}
        for result in results:
            component = result['component']
            if component not in component_results or result['confidence'] > component_results[component]['confidence']:
                component_results[component] = result
        
        # Merge all component data
        for component_result in component_results.values():
            final_data.update(component_result['data'])
        
        return final_data
    
    async def _finalize_extraction(self, extracted_data: Dict[str, Any], 
                                 schema: Dict[str, Any], 
                                 confidence_metrics: List[ConfidenceMetrics],
                                 processing_stats: ProcessingStats,
                                 start_time: datetime) -> Dict[str, Any]:
        """Finalize extraction with validation and reporting"""
        
        # Validate against schema
        validation_errors = self._validate_against_schema(extracted_data, schema)
        
        # Calculate overall metrics
        total_processing_time = (datetime.now() - start_time).total_seconds()
        avg_confidence = statistics.mean([m.confidence_score for m in confidence_metrics])
        
        # Determine fields requiring human review
        human_review_fields = [
            m.field_path for m in confidence_metrics 
            if m.human_review_required or not m.validation_passed
        ]
        
        return {
            'extracted_data': extracted_data,
            'metadata': {
                'processing_stats': asdict(processing_stats),
                'confidence_metrics': [asdict(m) for m in confidence_metrics],
                'validation_errors': validation_errors,
                'human_review_required': human_review_fields,
                'overall_confidence': avg_confidence,
                'total_processing_time': total_processing_time,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _validate_against_schema(self, data: Dict[str, Any], 
                               schema: Dict[str, Any]) -> List[str]:
        """Basic schema validation"""
        errors = []
        
        # Check required fields
        if 'required' in schema and isinstance(schema['required'], list):
            for required_field in schema['required']:
                if required_field not in data:
                    errors.append(f"Missing required field: {required_field}")
        
        # Check field types (basic validation)
        if 'properties' in schema:
            for field_name, field_schema in schema['properties'].items():
                if field_name in data and data[field_name] is not None:
                    expected_type = field_schema.get('type')
                    actual_value = data[field_name]
                    
                    if expected_type == 'string' and not isinstance(actual_value, str):
                        errors.append(f"Field {field_name} should be string, got {type(actual_value)}")
                    elif expected_type == 'number' and not isinstance(actual_value, (int, float)):
                        errors.append(f"Field {field_name} should be number, got {type(actual_value)}")
                    elif expected_type == 'boolean' and not isinstance(actual_value, bool):
                        errors.append(f"Field {field_name} should be boolean, got {type(actual_value)}")
                    elif expected_type == 'array' and not isinstance(actual_value, list):
                        errors.append(f"Field {field_name} should be array, got {type(actual_value)}")
        
        return errors

# Test Runner and Examples
class TestRunner:
    """Test runner for the extraction system"""
    
    def __init__(self, extractor: TextToJSONExtractor):
        self.extractor = extractor
    
    async def run_tests(self):
        """Run comprehensive tests"""
        logger.info("Starting comprehensive test suite...")
        
        # Test 1: Simple schema
        await self._test_simple_schema()
        
        # Test 2: Medium complexity schema
        await self._test_medium_schema()
        
        # Test 3: Complex nested schema
        await self._test_complex_schema()
        
        # Test 4: Large document processing
        await self._test_large_document()
        
        logger.info("All tests completed!")
    
    async def _test_simple_schema(self):
        """Test simple schema extraction"""
        logger.info("Test 1: Simple schema extraction")
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "active": {"type": "boolean"}
            },
            "required": ["name", "email"]
        }
        
        text = """
        John Smith is a 32-year-old software engineer. His email address is john.smith@example.com.
        He is currently active in the system and has been working with us for 3 years.
        """
        
        result = await self.extractor.extract(text, schema)
        logger.info(f"Simple test result: {json.dumps(result, indent=2)}")
    
    async def _test_medium_schema(self):
        """Test medium complexity schema"""
        logger.info("Test 2: Medium complexity schema extraction")
        
        schema = {
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
                                "phone": {"type": "string"},
                                "address": {
                                    "type": "object",
                                    "properties": {
                                        "street": {"type": "string"},
                                        "city": {"type": "string"},
                                        "zipcode": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "order": {
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
                        },
                        "total": {"type": "number"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "processing", "shipped", "delivered"]
                        }
                    }
                }
            },
            "required": ["customer", "order"]
        }
        
        text = """
        Order #ORD-2024-001 for customer Sarah Johnson
        Contact: sarah.johnson@email.com, (555) 123-4567
        Shipping Address: 123 Main Street, Springfield, IL 62701
        
        Items:
        - Widget A: 2 units @ $15.99 each
        - Widget B: 1 unit @ $25.50
        - Premium Service: 1 unit @ $10.00
        
        Order Total: $67.48
        Status: Processing
        """
        
        result = await self.extractor.extract(text, schema)
        logger.info(f"Medium test result: {json.dumps(result['extracted_data'], indent=2)}")
    
    async def _test_complex_schema(self):
        """Test complex nested schema"""
        logger.info("Test 3: Complex nested schema extraction")
        
        schema = {
            "type": "object",
            "properties": {
                "project": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "stakeholders": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "role": {"type": "string"},
                                    "department": {"type": "string"},
                                    "responsibilities": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        },
                        "requirements": {
                            "type": "object",
                            "properties": {
                                "functional": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "title": {"type": "string"},
                                            "description": {"type": "string"},
                                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                                            "status": {"type": "string", "enum": ["approved", "pending", "rejected"]},
                                            "dependencies": {"type": "array", "items": {"type": "string"}}
                                        }
                                    }
                                },
                                "non_functional": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "category": {"type": "string"},
                                            "requirement": {"type": "string"},
                                            "metric": {"type": "string"},
                                            "target_value": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        },
                        "timeline": {
                            "type": "object",
                            "properties": {
                                "start_date": {"type": "string"},
                                "end_date": {"type": "string"},
                                "milestones": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "date": {"type": "string"},
                                            "deliverables": {"type": "array", "items": {"type": "string"}}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["project"]
        }
        
        text = """
        PROJECT: Customer Portal Redesign
        
        DESCRIPTION: Complete overhaul of the customer-facing portal to improve user experience and add new self-service capabilities.
        
        STAKEHOLDERS:
        - Alice Chen, Product Manager, Product Team - Requirements gathering, stakeholder communication
        - Bob Rodriguez, Lead Developer, Engineering Team - Technical architecture, development oversight  
        - Carol Williams, UX Designer, Design Team - User experience design, usability testing
        - David Kumar, QA Manager, Quality Assurance - Testing strategy, quality validation
        
        FUNCTIONAL REQUIREMENTS:
        
        REQ-001: User Authentication System
        - High priority, approved status
        - Implement secure login with multi-factor authentication
        - Dependencies: Security framework setup
        
        REQ-002: Dashboard Customization
        - Medium priority, pending status  
        - Allow users to customize their dashboard layout
        - Dependencies: REQ-001
        
        REQ-003: Document Management
        - High priority, approved status
        - Enable upload, download, and organization of documents
        - Dependencies: REQ-001, Storage integration
        
        NON-FUNCTIONAL REQUIREMENTS:
        - Performance: Page load time should be under 2 seconds (target: <2s)
        - Security: All data must be encrypted in transit and at rest (target: 256-bit encryption)
        - Availability: System uptime should be 99.9% (target: 99.9%)
        
        TIMELINE:
        Project Start: January 15, 2024
        Project End: June 30, 2024
        
        MILESTONES:
        - Requirements Finalization (February 15, 2024)
          Deliverables: Requirements document, wireframes
        - Design Completion (March 30, 2024)  
          Deliverables: UI mockups, design system
        - Development Phase 1 (May 15, 2024)
          Deliverables: Core functionality, authentication system
        - Testing & Launch (June 30, 2024)
          Deliverables: Test reports, production deployment
        """
        
        result = await self.extractor.extract(text, schema)
        logger.info(f"Complex test - Confidence: {result['metadata']['overall_confidence']:.2f}")
        logger.info(f"Human review required: {result['metadata']['human_review_required']}")
    
    async def _test_large_document(self):
        """Test large document processing"""
        logger.info("Test 4: Large document processing")
        
        # Generate a large document
        large_text = """
        COMPREHENSIVE PROJECT REQUIREMENTS DOCUMENT
        
        """ + """
        SECTION 1: EXECUTIVE SUMMARY
        This document outlines the requirements for the new enterprise resource planning system.
        The system will integrate multiple business functions including finance, HR, operations, and customer management.
        
        """ * 50  # Repeat to make it large
        
        schema = {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content_summary": {"type": "string"}
                        }
                    }
                },
                "key_topics": {"type": "array", "items": {"type": "string"}},
                "document_length": {"type": "string"}
            }
        }
        
        result = await self.extractor.extract(large_text, schema)
        logger.info(f"Large document processing completed in {result['metadata']['processing_stats']['processing_time']:.2f}s")
        logger.info(f"Tokens processed: {result['metadata']['processing_stats']['total_tokens']}")
        logger.info(f"API calls: {result['metadata']['processing_stats']['api_calls']}")

# Main execution and API setup
async def main():
    """Main execution function"""
    
    # Configuration
    try:
        from config import settings
        CLAUDE_API_KEY = settings.validate_api_key()
    except ValueError as e:
        logger.error(f"API key configuration error: {str(e)}")
        return
    
    # Initialize the extraction system
    extractor = TextToJSONExtractor(CLAUDE_API_KEY)
    
    # Run tests
    test_runner = TestRunner(extractor)
    await test_runner.run_tests()
    
    # Interactive example
    logger.info("\n" + "="*50)
    logger.info("INTERACTIVE EXAMPLE")
    logger.info("="*50)
    
    # Example: Email chain extraction
    email_chain = """
    From: alice@company.com
    To: bob@vendor.com
    Subject: Re: API Integration Requirements
    Date: 2024-01-15
    
    Hi Bob,
    
    Thanks for the updated proposal. After reviewing with our team, we have the following requirements:
    
    1. Authentication: OAuth 2.0 with refresh tokens
    2. Rate limiting: 1000 requests per hour
    3. Data format: JSON with UTF-8 encoding
    4. Response time: Under 200ms for 95% of requests
    5. Availability: 99.9% uptime SLA
    6. Security: TLS 1.3 minimum, API key rotation every 90 days
    
    Timeline:
    - Integration testing: February 1-15, 2024
    - Pilot launch: February 20, 2024
    - Full production: March 1, 2024
    
    Budget approved: $50,000 for implementation
    Monthly recurring: $5,000
    
    Please confirm these requirements and provide updated timeline.
    
    Best regards,
    Alice Chen
    Technical Product Manager
    
    ---
    
    From: bob@vendor.com  
    To: alice@company.com
    Subject: Re: API Integration Requirements
    Date: 2024-01-16
    
    Hi Alice,
    
    Perfect! We can meet all your requirements. Here's our confirmation:
    
    Technical Specs Confirmed:
    ✓ OAuth 2.0 with refresh tokens - supported
    ✓ Rate limiting 1000/hour - can provide up to 2000/hour
    ✓ JSON/UTF-8 - standard format
    ✓ <200ms response time - our average is 50ms
    ✓ 99.9% uptime - we guarantee 99.95%
    ✓ TLS 1.3 and key rotation - security standard
    
    Updated Timeline:
    - Technical setup: January 20-25, 2024
    - Integration testing: February 1-15, 2024 ✓
    - Pilot launch: February 18, 2024 (2 days early!)
    - Full production: March 1, 2024 ✓
    
    Pricing Confirmed:
    - Implementation: $45,000 (5k discount for early commitment)
    - Monthly recurring: $4,500 (volume discount applied)
    
    Contract ready for signature. Shall we schedule a call this week?
    
    Best,
    Bob Martinez
    Solutions Architect
    """
    
    email_schema = {
        "type": "object",
        "properties": {
            "email_chain": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "participants": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                                "title": {"type": "string"},
                                "company": {"type": "string"}
                            }
                        }
                    },
                    "requirements": {
                        "type": "object",
                        "properties": {
                            "technical": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "specification": {"type": "string"},
                                        "status": {"type": "string", "enum": ["requested", "confirmed", "modified", "pending"]}
                                    }
                                }
                            },
                            "business": {
                                "type": "object",
                                "properties": {
                                    "budget": {
                                        "type": "object",
                                        "properties": {
                                            "implementation_cost": {"type": "number"},
                                            "monthly_recurring": {"type": "number"},
                                            "currency": {"type": "string"}
                                        }
                                    },
                                    "timeline": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "milestone": {"type": "string"},
                                                "date": {"type": "string"},
                                                "status": {"type": "string", "enum": ["planned", "confirmed", "modified"]}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "agreements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "agreed_value": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                            }
                        }
                    },
                    "next_steps": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "required": ["email_chain"]
    }
    
    logger.info("Processing email chain extraction...")
    result = await extractor.extract(email_chain, email_schema, confidence_threshold=0.6)
    
    logger.info(f"\nExtraction completed!")
    logger.info(f"Overall confidence: {result['metadata']['overall_confidence']:.2f}")
    logger.info(f"Processing time: {result['metadata']['total_processing_time']:.2f}s")
    logger.info(f"Strategy used: {result['metadata']['processing_stats']['strategy_used']}")
    logger.info(f"Human review required for: {result['metadata']['human_review_required']}")
    
    # Save result to file
    output_file = "extraction_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Full result saved to {output_file}")
    
    # Display key extracted data
    if result['extracted_data'] and 'email_chain' in result['extracted_data']:
        email_data = result['extracted_data']['email_chain']
        
        logger.info("\n" + "="*30)
        logger.info("KEY EXTRACTED DATA")
        logger.info("="*30)
        
        if 'participants' in email_data:
            logger.info(f"Participants: {len(email_data['participants'])}")
            for p in email_data['participants']:
                logger.info(f"  - {p.get('name', 'Unknown')} ({p.get('title', 'Unknown title')})")
        
        if 'requirements' in email_data and 'technical' in email_data['requirements']:
            logger.info(f"Technical Requirements: {len(email_data['requirements']['technical'])}")
            for req in email_data['requirements']['technical'][:3]:  # Show first 3
                logger.info(f"  - {req.get('category', 'Unknown')}: {req.get('specification', 'Unknown')}")
        
        if 'agreements' in email_data:
            logger.info(f"Key Agreements: {len(email_data['agreements'])}")
            for agreement in email_data['agreements'][:3]:  # Show first 3
                logger.info(f"  - {agreement.get('topic', 'Unknown')}: {agreement.get('agreed_value', 'Unknown')}")

# API Server Setup (Flask-based)
def create_api_server(extractor: TextToJSONExtractor):
    """Create a Flask API server for the extraction system"""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        logger.error("Flask not installed. Run: pip install flask flask-cors")
        return None
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    @app.route('/extract', methods=['POST'])
    async def extract_data():
        try:
            data = request.get_json()
            
            if not data or 'text' not in data or 'schema' not in data:
                return jsonify({
                    "error": "Missing required fields: 'text' and 'schema'"
                }), 400
            
            text = data['text']
            schema = data['schema']
            confidence_threshold = data.get('confidence_threshold', 0.7)
            
            # Run extraction
            result = await extractor.extract(text, schema, confidence_threshold)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500
    
    @app.route('/analyze_schema', methods=['POST'])
    def analyze_schema():
        try:
            data = request.get_json()
            
            if not data or 'schema' not in data:
                return jsonify({"error": "Missing 'schema' field"}), 400
            
            schema = data['schema']
            analysis = extractor.schema_analyzer.analyze_schema(schema)
            
            return jsonify(analysis)
            
        except Exception as e:
            logger.error(f"Schema analysis error: {str(e)}")
            return jsonify({
                "error": "Schema analysis failed",
                "details": str(e)
            }), 500
    
    return app

# CLI Interface
def run_cli():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Text-to-JSON Extraction System')
    parser.add_argument('--api-key', required=True, help='Claude API key')
    parser.add_argument('--text-file', help='Input text file')
    parser.add_argument('--schema-file', help='JSON schema file')
    parser.add_argument('--output-file', help='Output JSON file')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Confidence threshold')
    parser.add_argument('--server', action='store_true', help='Run as API server')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    
    args = parser.parse_args()
    
    async def run_extraction():
        extractor = TextToJSONExtractor(args.api_key)
        
        if args.test:
            test_runner = TestRunner(extractor)
            await test_runner.run_tests()
            return
        
        if args.server:
            app = create_api_server(extractor)
            if app:
                logger.info(f"Starting API server on port {args.port}")
                app.run(host='0.0.0.0', port=args.port, debug=True)
            return
        
        if not args.text_file or not args.schema_file:
            logger.error("--text-file and --schema-file required for extraction")
            return
        
        # Load input files
        with open(args.text_file, 'r') as f:
            text = f.read()
        
        with open(args.schema_file, 'r') as f:
            schema = json.load(f)
        
        # Run extraction
        result = await extractor.extract(text, schema, args.confidence_threshold)
        
        # Save result
        output_file = args.output_file or 'extraction_result.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Extraction completed. Result saved to {output_file}")
        logger.info(f"Confidence: {result['metadata']['overall_confidence']:.2f}")
    
    asyncio.run(run_extraction())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # CLI mode
        run_cli()
    else:
        # Direct execution mode
        asyncio.run(main())