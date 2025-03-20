# Automatic Reasoning and Tool-use (ART) Prompting

Automatic Reasoning and Tool-use (ART) is an advanced prompting technique that enables language models to decompose complex tasks into simpler subtasks, reason about how to solve them, and use tools when necessary. ART combines structured reasoning with tool use in a systematic framework.

## How It Works

ART prompting follows a structured approach:

1. **Task Decomposition**: Breaking down complex problems into manageable subtasks
2. **Reasoning**: Thinking through how to approach each subtask
3. **Tool Selection**: Identifying when and which tools to use
4. **Tool Use**: Executing tools with appropriate parameters
5. **Result Integration**: Combining tool outputs with reasoning to solve the original task

This technique helps models tackle complex problems by providing a systematic framework for reasoning and leveraging external capabilities.

## Example Implementation

```javascript
import { generateText } from 'ai';

// Define available tools
const tools = {
  search: async (query) => {
    // Simulated search function
    return `Search results for "${query}": [simulated results]`;
  },
  calculator: async (expression) => {
    // Safely evaluate mathematical expressions
    try {
      return `Result: ${eval(expression)}`;
    } catch (error) {
      return `Error: ${error.message}`;
    }
  },
  weather: async (location) => {
    // Simulated weather API
    return `Weather in ${location}: Partly cloudy, 72°F`;
  }
};

async function artPrompting(query) {
  // Step 1: Task decomposition
  const decompositionResponse = await generateText({
    model: yourModel,
    system: `You are an expert at breaking down complex problems into simpler subtasks.`,
    prompt: `Task: ${query}
    
    Break down this task into a sequence of 2-5 subtasks that would help solve it.
    For each subtask, specify:
    1. A clear description of what needs to be done
    2. Whether this subtask might require using a tool
    
    Available tools: search, calculator, weather
    
    Format your response as:
    Subtask 1: [description]
    Requires tool: [yes/no]
    
    Subtask 2: [description]
    Requires tool: [yes/no]
    
    ...and so on.`,
  });
  
  // Parse the subtasks
  const subtasks = decompositionResponse
    .split('\n\n')
    .filter(block => block.trim().startsWith('Subtask'))
    .map(block => {
      const lines = block.split('\n');
      const description = lines[0].replace(/^Subtask \d+:\s*/, '').trim();
      const requiresTool = lines[1].includes('yes');
      return { description, requiresTool };
    });
  
  // Step 2: Process each subtask
  const results = [];
  
  for (let i = 0; i < subtasks.length; i++) {
    const subtask = subtasks[i];
    
    if (subtask.requiresTool) {
      // Step 2a: Tool selection for subtasks that need tools
      const toolSelectionResponse = await generateText({
        model: yourModel,
        system: `You are an expert at selecting the right tool for a task.`,
        prompt: `Subtask: ${subtask.description}
        
        Available tools:
        - search: Search for information on the web
        - calculator: Perform mathematical calculations
        - weather: Get weather information for a location
        
        Select the most appropriate tool for this subtask and specify the parameters.
        
        Format your response as:
        Tool: [tool name]
        Parameters: [parameters for the tool]
        Reasoning: [brief explanation of why this tool is appropriate]`,
      });
      
      // Parse the tool selection
      const toolMatch = toolSelectionResponse.match(/Tool:\s*(.*)/);
      const paramsMatch = toolSelectionResponse.match(/Parameters:\s*(.*)/);
      
      if (toolMatch && paramsMatch) {
        const toolName = toolMatch[1].trim();
        const toolParams = paramsMatch[1].trim();
        
        // Step 2b: Tool use
        if (tools[toolName]) {
          const toolResult = await tools[toolName](toolParams);
          results.push({
            subtask: subtask.description,
            tool: toolName,
            parameters: toolParams,
            result: toolResult
          });
        } else {
          results.push({
            subtask: subtask.description,
            error: `Tool "${toolName}" not found`
          });
        }
      } else {
        results.push({
          subtask: subtask.description,
          error: "Failed to parse tool selection"
        });
      }
    } else {
      // Step 2c: Reasoning for subtasks that don't need tools
      const reasoningResponse = await generateText({
        model: yourModel,
        system: `You are an expert at reasoning through problems.`,
        prompt: `Subtask: ${subtask.description}
        
        Think through this subtask step by step without using external tools.
        
        Format your response as:
        Reasoning: [your step-by-step reasoning]
        Conclusion: [your conclusion for this subtask]`,
      });
      
      // Parse the reasoning
      const conclusionMatch = reasoningResponse.match(/Conclusion:\s*(.*)/s);
      const conclusion = conclusionMatch ? conclusionMatch[1].trim() : "No conclusion provided";
      
      results.push({
        subtask: subtask.description,
        reasoning: reasoningResponse,
        conclusion
      });
    }
  }
  
  // Step 3: Integration of results
  const integrationPrompt = `Original task: ${query}
  
  Subtask results:
  ${results.map((result, index) => {
    if (result.error) {
      return `Subtask ${index + 1}: ${result.subtask}\nError: ${result.error}`;
    } else if (result.tool) {
      return `Subtask ${index + 1}: ${result.subtask}\nTool used: ${result.tool}\nParameters: ${result.parameters}\nResult: ${result.result}`;
    } else {
      return `Subtask ${index + 1}: ${result.subtask}\nConclusion: ${result.conclusion}`;
    }
  }).join('\n\n')}
  
  Based on these results, provide a comprehensive answer to the original task.`;
  
  const finalResponse = await generateText({
    model: yourModel,
    system: `You are an expert at synthesizing information to solve complex problems.`,
    prompt: integrationPrompt,
  });
  
  return {
    originalQuery: query,
    subtasks,
    results,
    finalAnswer: finalResponse
  };
}
```

## Using the Vercel AI SDK

```javascript
import { streamText } from 'ai';

// Define available tools
const tools = {
  search: async (query) => {
    // Implement search functionality
    return `Search results for "${query}": [results here]`;
  },
  calculator: async (expression) => {
    // Safely evaluate mathematical expressions
    try {
      return `Result: ${eval(expression)}`;
    } catch (error) {
      return `Error: ${error.message}`;
    }
  },
  database: async (query) => {
    // Simulated database query
    return `Database results for "${query}": [data here]`;
  }
};

async function artPromptingWithVercel(query) {
  // Step 1: Task decomposition with streaming
  const decompositionStream = await streamText({
    model: yourModel,
    system: `You are an expert at breaking down complex problems into simpler subtasks.`,
    prompt: `Task: ${query}
    
    Break down this task into a sequence of 2-5 subtasks that would help solve it.
    For each subtask, specify:
    1. A clear description of what needs to be done
    2. Whether this subtask might require using a tool
    
    Available tools: search, calculator, database
    
    Format your response as:
    Subtask 1: [description]
    Requires tool: [yes/no]
    
    Subtask 2: [description]
    Requires tool: [yes/no]
    
    ...and so on.`,
  });
  
  let decompositionText = '';
  for await (const chunk of decompositionStream) {
    decompositionText += chunk;
  }
  
  // Parse the subtasks
  const subtasks = decompositionText
    .split('\n\n')
    .filter(block => block.trim().startsWith('Subtask'))
    .map(block => {
      const lines = block.split('\n');
      const description = lines[0].replace(/^Subtask \d+:\s*/, '').trim();
      const requiresTool = lines[1].includes('yes');
      return { description, requiresTool };
    });
  
  // Step 2: Process each subtask
  const results = [];
  
  for (let i = 0; i < subtasks.length; i++) {
    const subtask = subtasks[i];
    
    if (subtask.requiresTool) {
      // Tool selection and use
      const toolSelectionStream = await streamText({
        model: yourModel,
        system: `You are an expert at selecting and using tools.`,
        prompt: `Subtask: ${subtask.description}
        
        Available tools:
        - search: Search for information on the web
        - calculator: Perform mathematical calculations
        - database: Query a database for information
        
        1. Select the most appropriate tool for this subtask
        2. Specify the exact parameters to use with the tool
        
        Format your response as:
        Tool: [tool name]
        Parameters: [parameters for the tool]
        Reasoning: [brief explanation of why this tool is appropriate]`,
      });
      
      let toolSelectionText = '';
      for await (const chunk of toolSelectionStream) {
        toolSelectionText += chunk;
      }
      
      // Parse the tool selection
      const toolMatch = toolSelectionText.match(/Tool:\s*(.*)/);
      const paramsMatch = toolSelectionText.match(/Parameters:\s*(.*)/);
      
      if (toolMatch && paramsMatch) {
        const toolName = toolMatch[1].trim();
        const toolParams = paramsMatch[1].trim();
        
        // Execute the tool
        if (tools[toolName]) {
          const toolResult = await tools[toolName](toolParams);
          results.push({
            subtask: subtask.description,
            tool: toolName,
            parameters: toolParams,
            result: toolResult
          });
        } else {
          results.push({
            subtask: subtask.description,
            error: `Tool "${toolName}" not found`
          });
        }
      }
    } else {
      // Reasoning for subtasks that don't need tools
      const reasoningStream = await streamText({
        model: yourModel,
        system: `You are an expert at reasoning through problems.`,
        prompt: `Subtask: ${subtask.description}
        
        Think through this subtask step by step without using external tools.
        
        Format your response as:
        Reasoning: [your step-by-step reasoning]
        Conclusion: [your conclusion for this subtask]`,
      });
      
      let reasoningText = '';
      for await (const chunk of reasoningStream) {
        reasoningText += chunk;
      }
      
      // Parse the reasoning
      const conclusionMatch = reasoningText.match(/Conclusion:\s*(.*)/s);
      const conclusion = conclusionMatch ? conclusionMatch[1].trim() : "No conclusion provided";
      
      results.push({
        subtask: subtask.description,
        reasoning: reasoningText,
        conclusion
      });
    }
  }
  
  // Step 3: Integration of results
  const integrationPrompt = `Original task: ${query}
  
  Subtask results:
  ${results.map((result, index) => {
    if (result.error) {
      return `Subtask ${index + 1}: ${result.subtask}\nError: ${result.error}`;
    } else if (result.tool) {
      return `Subtask ${index + 1}: ${result.subtask}\nTool used: ${result.tool}\nParameters: ${result.parameters}\nResult: ${result.result}`;
    } else {
      return `Subtask ${index + 1}: ${result.subtask}\nConclusion: ${result.conclusion}`;
    }
  }).join('\n\n')}
  
  Based on these results, provide a comprehensive answer to the original task.`;
  
  const finalResponseStream = await streamText({
    model: yourModel,
    system: `You are an expert at synthesizing information to solve complex problems.`,
    prompt: integrationPrompt,
  });
  
  let finalResponse = '';
  for await (const chunk of finalResponseStream) {
    finalResponse += chunk;
  }
  
  return {
    originalQuery: query,
    subtasks,
    results,
    finalAnswer: finalResponse
  };
}
```

## Advanced ART Patterns

### Recursive Task Decomposition

```javascript
async function recursiveArtPrompting(query, maxDepth = 2) {
  // Initial task decomposition
  const result = await decomposeAndSolve(query, 0, maxDepth);
  return result;
}

async function decomposeAndSolve(task, currentDepth, maxDepth) {
  // Base case: maximum recursion depth reached
  if (currentDepth >= maxDepth) {
    return await solveDirectly(task);
  }
  
  // Decompose the task
  const decompositionResponse = await generateText({
    model: yourModel,
    system: `You are an expert at breaking down complex problems into simpler subtasks.`,
    prompt: `Task: ${task}
    
    Determine if this task should be broken down further or solved directly.
    If it should be broken down, provide 2-3 subtasks.
    
    Format your response as:
    Decision: [break down/solve directly]
    
    If breaking down:
    Subtask 1: [description]
    Subtask 2: [description]
    Subtask 3: [description] (optional)`,
  });
  
  // Parse the decision
  const decisionMatch = decompositionResponse.match(/Decision:\s*(.*)/);
  const decision = decisionMatch ? decisionMatch[1].trim().toLowerCase() : "solve directly";
  
  if (decision.includes("break down")) {
    // Parse the subtasks
    const subtaskMatches = decompositionResponse.matchAll(/Subtask \d+:\s*(.*)/g);
    const subtasks = Array.from(subtaskMatches, match => match[1].trim());
    
    // Recursively solve each subtask
    const subtaskResults = await Promise.all(
      subtasks.map(subtask => decomposeAndSolve(subtask, currentDepth + 1, maxDepth))
    );
    
    // Integrate the results
    return await integrateResults(task, subtasks, subtaskResults);
  } else {
    // Solve the task directly
    return await solveDirectly(task);
  }
}

async function solveDirectly(task) {
  // Determine if a tool is needed
  const toolSelectionResponse = await generateText({
    model: yourModel,
    system: `You are an expert at determining when tools are needed to solve problems.`,
    prompt: `Task: ${task}
    
    Determine if this task requires using a tool.
    Available tools: search, calculator, database
    
    Format your response as:
    Requires tool: [yes/no]
    
    If yes:
    Tool: [tool name]
    Parameters: [parameters]`,
  });
  
  // Parse the tool requirement
  const requiresToolMatch = toolSelectionResponse.match(/Requires tool:\s*(.*)/);
  const requiresTool = requiresToolMatch ? requiresToolMatch[1].trim().toLowerCase() === "yes" : false;
  
  if (requiresTool) {
    // Parse the tool selection
    const toolMatch = toolSelectionResponse.match(/Tool:\s*(.*)/);
    const paramsMatch = toolSelectionResponse.match(/Parameters:\s*(.*)/);
    
    if (toolMatch && paramsMatch) {
      const toolName = toolMatch[1].trim();
      const toolParams = paramsMatch[1].trim();
      
      // Execute the tool
      if (tools[toolName]) {
        const toolResult = await tools[toolName](toolParams);
        
        // Generate a solution using the tool result
        const solutionResponse = await generateText({
          model: yourModel,
          system: `You are an expert at solving problems using tool results.`,
          prompt: `Task: ${task}
          
          Tool used: ${toolName}
          Parameters: ${toolParams}
          Result: ${toolResult}
          
          Provide a solution to the task based on this tool result.`,
        });
        
        return {
          task,
          toolUsed: toolName,
          toolParams,
          toolResult,
          solution: solutionResponse
        };
      }
    }
  }
  
  // Solve without tools
  const solutionResponse = await generateText({
    model: yourModel,
    system: `You are an expert at solving problems through reasoning.`,
    prompt: `Task: ${task}
    
    Solve this task through step-by-step reasoning without using external tools.
    
    Format your response as:
    Reasoning: [your step-by-step reasoning]
    Solution: [your solution to the task]`,
  });
  
  return {
    task,
    solution: solutionResponse
  };
}

async function integrateResults(originalTask, subtasks, subtaskResults) {
  const integrationPrompt = `Original task: ${originalTask}
  
  Subtask results:
  ${subtasks.map((subtask, index) => {
    const result = subtaskResults[index];
    return `Subtask: ${subtask}\nSolution: ${result.solution}`;
  }).join('\n\n')}
  
  Integrate these subtask results to provide a comprehensive solution to the original task.`;
  
  const integratedSolution = await generateText({
    model: yourModel,
    system: `You are an expert at synthesizing information to solve complex problems.`,
    prompt: integrationPrompt,
  });
  
  return {
    task: originalTask,
    subtasks: subtasks.map((subtask, index) => ({
      description: subtask,
      result: subtaskResults[index]
    })),
    solution: integratedSolution
  };
}
```

### Tool Verification in ART

```javascript
async function artWithToolVerification(query) {
  // Standard ART process with added verification steps
  const decomposition = await decomposeTask(query);
  const results = [];
  
  for (const subtask of decomposition.subtasks) {
    if (subtask.requiresTool) {
      // Tool selection
      const toolSelection = await selectTool(subtask.description);
      
      // Tool verification - check if the selected tool is appropriate
      const verificationResponse = await generateText({
        model: yourModel,
        system: `You are an expert at verifying tool selections.`,
        prompt: `Subtask: ${subtask.description}
        
        Selected tool: ${toolSelection.tool}
        Parameters: ${toolSelection.parameters}
        Reasoning: ${toolSelection.reasoning}
        
        Verify if this tool selection is appropriate for the subtask.
        Consider:
        1. Is this the most appropriate tool for the task?
        2. Are the parameters correctly formatted?
        3. Are there any potential issues or edge cases?
        
        Format your response as:
        Appropriate: [yes/no]
        Issues: [list any issues identified]
        Suggestions: [suggestions for improvement if any]`,
      });
      
      // Parse the verification
      const appropriateMatch = verificationResponse.match(/Appropriate:\s*(.*)/);
      const isAppropriate = appropriateMatch ? 
        appropriateMatch[1].trim().toLowerCase() === "yes" : false;
      
      if (isAppropriate) {
        // Execute the tool
        const toolResult = await executeTool(toolSelection.tool, toolSelection.parameters);
        
        // Verify the tool result
        const resultVerificationResponse = await generateText({
          model: yourModel,
          system: `You are an expert at verifying tool results.`,
          prompt: `Subtask: ${subtask.description}
          
          Tool: ${toolSelection.tool}
          Parameters: ${toolSelection.parameters}
          Result: ${toolResult}
          
          Verify if this result is useful and reliable for the subtask.
          
          Format your response as:
          Useful: [yes/no]
          Reliability: [high/medium/low]
          Issues: [list any issues with the result]
          Next steps: [what to do with this result]`,
        });
        
        results.push({
          subtask: subtask.description,
          tool: toolSelection.tool,
          parameters: toolSelection.parameters,
          result: toolResult,
          verification: resultVerificationResponse
        });
      } else {
        // Try an alternative tool or approach
        const alternativeResponse = await generateText({
          model: yourModel,
          system: `You are an expert at finding alternative approaches when a tool is not appropriate.`,
          prompt: `Subtask: ${subtask.description}
          
          Original tool selection: ${toolSelection.tool}
          Issues: ${verificationResponse}
          
          Suggest an alternative approach to solve this subtask.
          
          Format your response as:
          Alternative: [description of alternative approach]
          Requires tool: [yes/no]
          
          If requires tool:
          Tool: [alternative tool]
          Parameters: [parameters]`,
        });
        
        // Parse and execute the alternative
        // (Implementation details omitted for brevity)
        
        results.push({
          subtask: subtask.description,
          originalTool: toolSelection.tool,
          verification: verificationResponse,
          alternative: alternativeResponse
        });
      }
    } else {
      // Standard reasoning for subtasks that don't need tools
      const reasoning = await reasonThroughSubtask(subtask.description);
      results.push({
        subtask: subtask.description,
        reasoning
      });
    }
  }
  
  // Integration with verification
  const finalSolution = await integrateAndVerify(query, results);
  
  return {
    query,
    results,
    solution: finalSolution
  };
}
```

## Best Practices

1. **Clear task decomposition** - Break complex tasks into well-defined subtasks
2. **Explicit tool definitions** - Clearly define available tools and their capabilities
3. **Structured formats** - Use consistent formats for tool selection, parameters, and results
4. **Verification steps** - Verify tool selections and results when accuracy is critical
5. **Error handling** - Implement robust error handling for tool execution
6. **Result integration** - Provide clear instructions for integrating subtask results
7. **Recursive decomposition** - For very complex tasks, consider recursive decomposition
8. **Tool parameter validation** - Validate tool parameters before execution

## When to Use

ART prompting is particularly effective for:
- Complex tasks requiring multiple steps and different types of reasoning
- Problems that benefit from external tools or information sources
- Tasks involving both factual knowledge and computational steps
- Situations where systematic decomposition improves problem-solving
- Applications requiring transparent reasoning and tool use
- Tasks where verification of intermediate steps is important 