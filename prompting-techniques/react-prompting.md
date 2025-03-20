# ReAct Prompting (Reasoning + Acting)

ReAct (Reasoning + Acting) is a prompting technique that interleaves reasoning traces and task-specific actions in a synergistic way. This approach enables LLMs to solve complex tasks by alternating between generating reasoning steps and taking actions, such as using tools or retrieving information.

## How It Works

In ReAct prompting, the model follows a cycle of:
1. **Thought**: Reasoning about the current state and what to do next
2. **Action**: Deciding on an action to take (e.g., using a tool)
3. **Observation**: Receiving feedback or information from the action
4. **Repeat**: Continuing this cycle until the task is complete

This technique helps models break down complex problems, use tools effectively, and recover from errors.

## Basic Example

```javascript
import { generateText } from 'ai';

// Simulated tool functions
const tools = {
  search: async (query) => {
    // In a real implementation, this would call a search API
    return `Search results for "${query}": [simulated results]`;
  },
  calculator: async (expression) => {
    // In a real implementation, this would evaluate the expression safely
    try {
      // Use a safer evaluation method in production
      return `Result of "${expression}": ${eval(expression)}`;
    } catch (error) {
      return `Error calculating: ${error.message}`;
    }
  }
};

async function reactAgent(query, maxSteps = 5) {
  let context = `Query: ${query}\n\n`;
  
  for (let step = 0; step < maxSteps; step++) {
    // Generate the next thought, action, and observation
    const response = await generateText({
      model: "llama-3-70b-instruct",
      system: `You are a problem-solving agent that can think and act to answer questions.
      
      You have access to the following tools:
      - search: Search for information on the web
      - calculator: Perform mathematical calculations
      
      For each step, you should:
      1. Think about what you know and what you need to find out
      2. Decide on an action to take
      3. Observe the result of your action
      
      Format your response as:
      Thought: [your reasoning about what to do next]
      Action: [tool_name][tool_input]
      
      After your final step, provide your answer to the original query.`,
      prompt: context,
    });
    
    // Add the response to the context
    context += response + '\n';
    
    // Parse the action from the response
    const actionMatch = response.match(/Action: (\w+)\[(.*?)\]/);
    if (actionMatch) {
      const [_, toolName, toolInput] = actionMatch;
      
      // Execute the tool if it exists
      if (tools[toolName]) {
        const observation = await tools[toolName](toolInput);
        context += `Observation: ${observation}\n\n`;
      } else {
        context += `Observation: Error: Tool "${toolName}" not found.\n\n`;
      }
    }
    
    // Check if the agent has provided a final answer
    if (response.includes('Final Answer:') || !actionMatch) {
      break;
    }
  }
  
  // Extract the final answer
  const finalAnswerMatch = context.match(/Final Answer: ([\s\S]+?)(?:\n\n|$)/);
  return finalAnswerMatch ? finalAnswerMatch[1].trim() : context;
}

// Example usage
reactAgent("What is the square root of 144 plus the population of France?").then(answer => {
  console.log("Final answer:", answer);
});
```

## Using LangChain for ReAct

```javascript
import { ChatOpenAI } from "@langchain/openai";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { Calculator } from "@langchain/community/tools/calculator";
import { WebBrowser } from "@langchain/community/tools/webbrowser";

async function createReActAgent() {
  // Initialize the language model
  const model = new ChatOpenAI({
    model: "gpt-4",
    temperature: 0,
  });

  // Define the tools the agent can use
  const tools = [
    new Calculator(),
    new WebBrowser(),
  ];

  // Create the agent with ReAct framework
  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: "structured-chat-zero-shot-react-description",
    verbose: true,
  });

  return executor;
}

// Example usage
async function runReActExample() {
  const agent = await createReActAgent();
  const result = await agent.invoke({
    input: "What is the current temperature in New York City in Celsius, and what is that temperature squared?",
  });
  
  console.log("Agent response:", result.output);
}

runReActExample();
```

## Advanced ReAct Patterns

### Tool Selection with ReAct

```javascript
import { generateText } from 'ai';

async function reactWithToolSelection(query, availableTools) {
  // First, have the model decide which tools might be needed
  const toolSelectionResponse = await generateText({
    model: "llama-3-70b-instruct",
    system: "You are a helpful assistant that analyzes problems and determines which tools might be needed to solve them.",
    prompt: `For the following query, list the tools that might be useful to solve it. Choose from: ${Object.keys(availableTools).join(', ')}.
    
    Query: ${query}
    
    Format your response as:
    Thought: [your analysis of the problem]
    Needed Tools: [comma-separated list of tool names]`,
  });
  
  // Parse the needed tools
  const toolsMatch = toolSelectionResponse.match(/Needed Tools: (.*)/);
  const neededToolNames = toolsMatch ? 
    toolsMatch[1].split(',').map(t => t.trim()) : 
    Object.keys(availableTools);
  
  // Filter the available tools to only include the needed ones
  const selectedTools = {};
  for (const toolName of neededToolNames) {
    if (availableTools[toolName]) {
      selectedTools[toolName] = availableTools[toolName];
    }
  }
  
  // Now proceed with ReAct using only the selected tools
  return reactWithTools(query, selectedTools);
}
```

### Self-Reflection in ReAct

```javascript
import { generateText } from 'ai';

async function reactWithReflection(query, tools, maxSteps = 5) {
  let context = `Query: ${query}\n\n`;
  let step = 1;
  
  while (step <= maxSteps) {
    // Generate the next thought, action, and observation
    const response = await generateText({
      model: "llama-3-70b-instruct",
      system: `You are a problem-solving agent that can think, act, and reflect to answer questions.`,
      prompt: `${context}
      
      Step ${step}:
      1. Think about what you know and what you need to find out
      2. Decide on an action to take from these tools: ${Object.keys(tools).join(', ')}
      3. After receiving an observation, reflect on whether you're making progress
      
      Format your response as:
      Thought: [your reasoning about what to do next]
      Action: [tool_name][tool_input]`,
    });
    
    context += response + '\n';
    
    // Parse and execute the action
    const actionMatch = response.match(/Action: (\w+)\[(.*?)\]/);
    if (actionMatch) {
      const [_, toolName, toolInput] = actionMatch;
      
      if (tools[toolName]) {
        const observation = await tools[toolName](toolInput);
        context += `Observation: ${observation}\n\n`;
        
        // Add a reflection step after each observation
        const reflection = await generateText({
          model: "llama-3-70b-instruct",
          system: "You are a reflective agent that evaluates progress toward solving a problem.",
          prompt: `${context}
          
          Reflect on your progress so far:
          1. Are you getting closer to answering the original query?
          2. Is there a more efficient approach you could take?
          3. Do you need to correct any misunderstandings or errors?
          
          Format your response as:
          Reflection: [your assessment of progress and any course corrections needed]`,
        });
        
        context += reflection + '\n\n';
      } else {
        context += `Observation: Error: Tool "${toolName}" not found.\n\n`;
      }
    }
    
    // Check if the agent has provided a final answer
    if (response.includes('Final Answer:')) {
      break;
    }
    
    step++;
  }
  
  // Extract the final answer
  const finalAnswerMatch = context.match(/Final Answer: ([\s\S]+?)(?:\n\n|$)/);
  return finalAnswerMatch ? finalAnswerMatch[1].trim() : "Could not determine a final answer within the step limit.";
}
```

### Multi-Agent ReAct

```javascript
import { generateText } from 'ai';

async function multiAgentReAct(query, specialistAgents, tools, maxSteps = 10) {
  let context = `Main Query: ${query}\n\n`;
  let step = 1;
  
  // Initialize the coordinator agent
  const coordinator = {
    name: "Coordinator",
    description: "Manages the overall problem-solving process and delegates to specialist agents",
    delegateTo: async (agentName, subQuery) => {
      if (!specialistAgents[agentName]) {
        return `Error: Agent "${agentName}" not found.`;
      }
      
      // Call the specialist agent with the sub-query
      const specialistResponse = await reactAgent(
        subQuery, 
        tools,
        3, // Limit specialist steps
        `You are ${agentName}, ${specialistAgents[agentName].description}`
      );
      
      return `${agentName}'s Response: ${specialistResponse}`;
    }
  };
  
  while (step <= maxSteps) {
    // Generate the coordinator's next thought and action
    const response = await generateText({
      model: "llama-3-70b-instruct",
      system: `You are the Coordinator agent that manages a team of specialist agents to solve complex problems.
      
      Available specialist agents:
      ${Object.entries(specialistAgents).map(([name, agent]) => `- ${name}: ${agent.description}`).join('\n')}
      
      You can:
      1. Use tools directly: Action: [tool_name][tool_input]
      2. Delegate to specialists: Delegate: [agent_name][sub_query]
      3. Provide a final answer: Final Answer: [answer]
      
      Think carefully about which approach is best for each step.`,
      prompt: context,
    });
    
    context += `Step ${step} (Coordinator):\n${response}\n\n`;
    
    // Check for delegation
    const delegateMatch = response.match(/Delegate: (\w+)\[(.*?)\]/);
    if (delegateMatch) {
      const [_, agentName, subQuery] = delegateMatch;
      const observation = await coordinator.delegateTo(agentName, subQuery);
      context += `Observation: ${observation}\n\n`;
    } else {
      // Check for direct tool use
      const actionMatch = response.match(/Action: (\w+)\[(.*?)\]/);
      if (actionMatch) {
        const [_, toolName, toolInput] = actionMatch;
        
        if (tools[toolName]) {
          const observation = await tools[toolName](toolInput);
          context += `Observation: ${observation}\n\n`;
        } else {
          context += `Observation: Error: Tool "${toolName}" not found.\n\n`;
        }
      }
    }
    
    // Check if the coordinator has provided a final answer
    if (response.includes('Final Answer:')) {
      break;
    }
    
    step++;
  }
  
  // Extract the final answer
  const finalAnswerMatch = context.match(/Final Answer: ([\s\S]+?)(?:\n\n|$)/);
  return finalAnswerMatch ? finalAnswerMatch[1].trim() : "Could not determine a final answer within the step limit.";
}

// Example specialist agents
const specialists = {
  "Researcher": {
    description: "Specializes in finding and summarizing information"
  },
  "Mathematician": {
    description: "Specializes in solving mathematical problems with precision"
  },
  "Planner": {
    description: "Specializes in breaking down complex tasks into manageable steps"
  }
};

// Example usage
multiAgentReAct(
  "I need to plan a trip to Paris, calculate a budget in euros, and find the top 3 attractions",
  specialists,
  tools
).then(answer => console.log(answer));
```

## Best Practices

1. **Clearly define tool capabilities** and limitations in the system prompt
2. **Use structured formats** for thoughts, actions, and observations
3. **Implement error handling** for tool execution failures
4. **Limit the maximum number of steps** to prevent infinite loops
5. **Provide clear examples** of the reasoning-action-observation cycle
6. **Include self-reflection** to help the model correct its course
7. **Use temperature settings strategically** (lower for precise tool use, higher for creative reasoning)
8. **Design tools with clear interfaces** and predictable outputs
9. **Implement validation** for tool inputs to prevent errors
10. **Maintain context efficiently** to avoid token limits
11. **Consider specialized agents** for different aspects of complex tasks
12. **Implement fallback mechanisms** when tools fail or are unavailable

## When to Use

ReAct prompting is particularly effective for:
- Complex multi-step tasks requiring external information
- Problem-solving that benefits from using specialized tools
- Tasks requiring dynamic planning and adaptation
- Information retrieval and synthesis from multiple sources
- Scenarios where step-by-step reasoning improves accuracy
- Applications requiring transparency in the decision-making process
- Tasks that combine both factual knowledge and computational abilities

## Limitations

- Increases token usage significantly due to verbose reasoning traces
- May struggle with tool API changes or unexpected tool outputs
- Can get stuck in reasoning loops without proper guardrails
- Performance depends on the quality of available tools
- May generate plausible-sounding but incorrect reasoning
- Tool execution adds latency to the overall response time
- Complex implementations can be difficult to debug and maintain

## Research Insights

Recent research has shown that:

- ReAct significantly outperforms standard prompting on tasks requiring tool use and multi-step reasoning
- The explicit reasoning step helps models better decide when and how to use tools
- Models with ReAct prompting show improved performance on tasks requiring factual knowledge and computation
- Self-reflection mechanisms can help models recover from errors and improve overall performance
- The quality of the initial system prompt strongly influences the effectiveness of ReAct
- Combining ReAct with retrieval-augmented generation can further improve performance on knowledge-intensive tasks
- Multi-agent ReAct frameworks can effectively distribute complex tasks among specialized agents

## Real-World Applications

- **Personal assistants**: Completing complex tasks like travel planning or research
- **Data analysis**: Querying databases, performing calculations, and generating insights
- **Customer support**: Troubleshooting technical issues using knowledge bases and diagnostic tools
- **Educational tutoring**: Breaking down complex problems and guiding students through solutions
- **Content creation**: Researching topics and generating structured content
- **Code generation**: Writing, testing, and debugging code with access to documentation
- **Decision support systems**: Analyzing data and providing recommendations with transparent reasoning 