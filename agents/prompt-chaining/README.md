# Prompt Chaining Agent Pattern

The Prompt Chaining pattern involves breaking down complex tasks into a sequence of simpler prompts, where each prompt builds upon the results of previous prompts. This creates a chain of reasoning that can solve problems more effectively than a single prompt.

## When to Use This Pattern

Use the Prompt Chaining pattern when:

- A task is too complex to solve with a single prompt
- The problem can be naturally decomposed into sequential steps
- Each step produces output that serves as input to the next step
- You need to maintain control over the exact sequence of operations
- Different prompts or models may be optimal for different stages of processing

## Implementation

A Prompt Chaining implementation typically involves:

1. Defining a clear sequence of steps
2. Crafting specialized prompts for each step
3. Managing the flow of information between steps
4. Handling errors that might occur at any step

## Example Implementation

See [prompt-chaining.ts](./prompt-chaining.ts) for a TypeScript implementation and [prompt_chaining.py](./prompt_chaining.py) for a Python implementation.

```typescript
// Basic structure of a prompt chaining implementation
async function solveWithPromptChain(input: string) {
  // Step 1: Analyze the problem
  const analysis = await executePrompt("Analyze this problem: " + input);
  
  // Step 2: Generate a solution plan based on analysis
  const plan = await executePrompt("Create a plan to solve this problem: " + analysis);
  
  // Step 3: Execute the plan to produce a solution
  const solution = await executePrompt("Implement this plan: " + plan);
  
  return solution;
}
```

## Considerations

- **Error Propagation**: Errors in early steps can affect all subsequent steps
- **Context Limits**: Be mindful of token limits when passing context between steps
- **Latency**: Multiple sequential calls increase overall latency
- **Observability**: Add logging between steps to aid debugging
- **Cost**: Multiple calls to LLMs increase the total cost

## Variations

1. **Dynamic Chaining**: Determining next steps based on results from previous steps
2. **Branching Chains**: Creating decision points that lead to different prompt paths
3. **Recursive Chains**: Applying the same prompt multiple times with refined inputs
4. **Hybrid Chains**: Combining LLM prompts with traditional algorithms

## Related Patterns

- **Routing**: Can be used to determine which chain to execute
- **Evaluation/Feedback Loops**: Adding validation steps between processing steps
- **Tool Usage**: Individual steps may involve tool usage for enhanced capabilities