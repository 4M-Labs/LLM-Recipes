# Routing Pattern

The Routing pattern uses an LLM to analyze incoming requests and direct them to the appropriate handler or specialist model. This pattern is essential for building systems that can effectively triage and distribute tasks.

## Key Concepts

1. **Request Analysis**: Understand and categorize incoming requests
2. **Handler Selection**: Choose the most appropriate handler for each request
3. **Dynamic Routing**: Adapt routing decisions based on context
4. **Fallback Mechanisms**: Handle edge cases and unknown request types

## When to Use

Use the Routing pattern when:
- You have diverse types of incoming requests
- Different handlers specialize in different tasks
- You need intelligent task distribution
- Request categorization is complex

## Implementation

### Basic Structure

```python
async def route_request(request: str) -> Dict[str, Any]:
    # Step 1: Analyze request
    analysis = await analyze_request(request)
    
    # Step 2: Select handler
    handler = select_handler(analysis)
    
    # Step 3: Process request
    result = await handler.process(request)
    
    return result
```

### Key Components

1. **Request Analyzer**:
   - Extract key information
   - Identify request type
   - Determine priority
   - Assess complexity

2. **Router Logic**:
   - Handler mapping
   - Load balancing
   - Priority queuing
   - Fallback strategies

3. **Handlers**:
   - Specialized processors
   - Error handling
   - Result formatting
   - Performance monitoring

## Best Practices

1. **Analysis Quality**:
   - Thorough request examination
   - Clear categorization criteria
   - Context preservation
   - Confidence scoring

2. **Routing Decisions**:
   - Clear routing rules
   - Handler capability matching
   - Load consideration
   - Error recovery paths

3. **Handler Management**:
   - Clear handler interfaces
   - Consistent error handling
   - Performance monitoring
   - Resource management

## Example Use Cases

1. **Customer Support**:
   ```
   Request → Analysis → Route to:
   - Technical Support
   - Billing Support
   - Product Information
   - General Inquiries
   ```

2. **Content Processing**:
   ```
   Content → Analysis → Route to:
   - Text Processor
   - Image Analyzer
   - Code Reviewer
   - Data Validator
   ```

3. **Task Distribution**:
   ```
   Task → Analysis → Route to:
   - Research Agent
   - Writing Agent
   - Analysis Agent
   - Review Agent
   ```

## Common Pitfalls

1. **Analysis Errors**:
   - Incorrect categorization
   - Missing context
   - Ambiguous requests
   - Over-complexity

2. **Routing Issues**:
   - Suboptimal handler selection
   - Load imbalance
   - Circular routing
   - Dead ends

3. **Handler Problems**:
   - Capability mismatches
   - Resource exhaustion
   - Error propagation
   - Inconsistent interfaces

## Optimization Tips

1. **Analysis Efficiency**:
   - Use efficient models
   - Cache similar requests
   - Implement fast paths
   - Prioritize important features

2. **Routing Performance**:
   - Smart caching
   - Load balancing
   - Priority handling
   - Quick fallbacks

3. **Handler Optimization**:
   - Resource pooling
   - Result caching
   - Parallel processing
   - Graceful degradation

## Implementation Example

```python
from typing import Dict, Any, List
import openai
from enum import Enum

class RequestType(Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    PRODUCT = "product"
    GENERAL = "general"

class RoutingResult:
    def __init__(
        self,
        request_type: RequestType,
        confidence: float,
        metadata: Dict[str, Any]
    ):
        self.request_type = request_type
        self.confidence = confidence
        self.metadata = metadata

async def analyze_request(request: str) -> RoutingResult:
    """Analyze the request and determine its type."""
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",  # Using smaller model for efficiency
        messages=[
            {
                "role": "system",
                "content": """Analyze the request and categorize it.
                Return a JSON object with:
                {
                    "type": "technical|billing|product|general",
                    "confidence": float between 0 and 1,
                    "keywords": list of relevant terms,
                    "priority": "high|medium|low"
                }"""
            },
            {
                "role": "user",
                "content": request
            }
        ],
        temperature=0.3,
    )
    
    result = response.choices[0].message.content
    parsed = json.loads(result)
    
    return RoutingResult(
        request_type=RequestType(parsed["type"]),
        confidence=parsed["confidence"],
        metadata={
            "keywords": parsed["keywords"],
            "priority": parsed["priority"]
        }
    )

class RequestHandler:
    def __init__(self, handler_type: RequestType):
        self.type = handler_type
    
    async def process(self, request: str) -> Dict[str, Any]:
        """Process the request using appropriate model and prompts."""
        # Implementation specific to handler type
        pass

class Router:
    def __init__(self):
        self.handlers: Dict[RequestType, List[RequestHandler]] = {
            rt: [] for rt in RequestType
        }
    
    def register_handler(self, handler: RequestHandler):
        """Register a new handler for a request type."""
        self.handlers[handler.type].append(handler)
    
    def select_handler(self, analysis: RoutingResult) -> RequestHandler:
        """Select the most appropriate handler based on analysis."""
        available_handlers = self.handlers[analysis.request_type]
        
        if not available_handlers:
            raise ValueError(f"No handlers available for {analysis.request_type}")
        
        # Simple round-robin for now
        return available_handlers[0]
    
    async def route_request(self, request: str) -> Dict[str, Any]:
        """Analyze request and route to appropriate handler."""
        try:
            # Analyze request
            analysis = await analyze_request(request)
            
            # Select handler
            handler = self.select_handler(analysis)
            
            # Process request
            result = await handler.process(request)
            
            return {
                "result": result,
                "metadata": {
                    "type": analysis.request_type.value,
                    "confidence": analysis.confidence,
                    "handler": handler.__class__.__name__
                }
            }
            
        except Exception as e:
            # Handle routing errors
            return {
                "error": str(e),
                "metadata": {
                    "original_request": request
                }
            }
```

## Monitoring and Debugging

1. **Key Metrics**:
   - Routing accuracy
   - Handler performance
   - Error rates
   - Response times

2. **Logging**:
   - Request details
   - Routing decisions
   - Handler selection
   - Error conditions

3. **Debugging Tools**:
   - Request tracing
   - Handler monitoring
   - Load visualization
   - Error analysis

## References

- [Implementation Examples](../../agents/routing/)
- [Setup Guide](../setup/routing-setup.md)
- [Troubleshooting Guide](../troubleshooting.md) 