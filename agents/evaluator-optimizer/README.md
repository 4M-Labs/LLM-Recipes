# Evaluator-Optimizer Agent Pattern

This pattern implements a feedback loop where generated solutions are evaluated against specific criteria and refined until they meet quality standards. It's particularly useful for tasks that require high-quality output meeting multiple criteria.

## Overview

The Evaluator-Optimizer pattern consists of three main components:

1. **Solution Generator**: Generates initial and improved solutions based on prompts and feedback
2. **Evaluator**: Assesses solutions against specified criteria
3. **Optimizer**: Refines solutions based on evaluation feedback

The pattern implements an iterative improvement cycle:
```
Generate → Evaluate → Improve → Evaluate → ... (until criteria met or max attempts reached)
```

## Features

- Customizable evaluation criteria with weights
- Detailed feedback for each criterion
- Weighted scoring system
- Automatic improvement based on feedback
- Maximum attempt limit to prevent infinite loops
- Comprehensive metadata about the optimization process

## Usage

### Basic Example

```javascript
import { generateWithEvaluation } from './evaluator-optimizer.js';

const prompt = "Write a function that calculates the Fibonacci sequence up to n terms.";
const criteriaDetails = [
  {
    name: "Correctness",
    description: "The solution should correctly implement the Fibonacci sequence logic",
    weight: 10
  },
  {
    name: "Efficiency",
    description: "The implementation should be efficient and avoid unnecessary calculations",
    weight: 8
  },
  {
    name: "Code Style",
    description: "The code should be well-formatted, readable, and follow best practices",
    weight: 7
  },
  {
    name: "Documentation",
    description: "The solution should include clear comments and documentation",
    weight: 6
  }
];

const result = await generateWithEvaluation(prompt, criteriaDetails);
console.log(result.finalSolution);
```

### Output Structure

The `generateWithEvaluation` function returns an object with:

```javascript
{
  originalPrompt: string,
  finalSolution: string,
  evaluationResult: {
    meetsAllCriteria: boolean,
    criteriaResults: {
      [criterionName]: {
        passed: boolean,
        score: number,
        feedback: string
      }
    },
    overallScore: number,
    feedback: string
  },
  attempts: number,
  success: boolean,
  allSolutions: Array<{
    solution: string,
    evaluation: EvaluationResult
  }>,
  metadata: {
    totalTokensUsed: number,
    processingTimeMs: number,
    finalAttempt: number,
    averageScore: number
  }
}
```

## Configuration

The agent can be configured through constants:

- `MAX_ATTEMPTS`: Maximum number of improvement iterations (default: 5)
- `MINIMUM_ACCEPTABLE_SCORE`: Minimum score to consider a solution acceptable (default: 7.5)

## Best Practices

1. **Criteria Design**:
   - Make criteria specific and measurable
   - Use appropriate weights based on importance
   - Include both technical and qualitative criteria

2. **Prompt Engineering**:
   - Be specific about requirements
   - Include examples if possible
   - Specify format requirements

3. **Performance Optimization**:
   - Use smaller models for initial attempts
   - Scale up model size for refinement
   - Set appropriate MAX_ATTEMPTS based on task complexity

## Dependencies

- OpenAI API
- termcolor (for console output formatting)

## Environment Variables

Required environment variables:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Error Handling

The agent includes error handling for:
- API failures
- Invalid responses
- Timeout conditions

Errors are logged with appropriate context for debugging.

## Limitations

- Quality depends on initial prompt clarity
- May require multiple iterations for complex tasks
- Token usage increases with each iteration
- Subject to OpenAI API rate limits and costs

## Contributing

Feel free to contribute by:
1. Improving evaluation criteria
2. Adding new optimization strategies
3. Enhancing error handling
4. Optimizing token usage
5. Adding new features

## License

MIT License - feel free to use and modify for your needs. 