/**
 * Parallelization Agent Pattern Implementation
 * 
 * This file demonstrates the Parallelization pattern using the OpenAI API.
 * It showcases how to distribute tasks across multiple LLM calls simultaneously,
 * aggregating results to handle complex or large-scale operations efficiently.
 */

import { OpenAI } from 'openai';
import { termcolor } from 'termcolor';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Process a single subtask using the appropriate model
 * 
 * @param {Object} subtask The subtask to process
 * @returns {Promise<Object>} The result of the subtask processing
 */
async function processSubtask(subtask) {
  console.log(termcolor.blue(`Processing subtask ${subtask.id}: ${subtask.type}`));
  
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `You are processing a subtask of type: ${subtask.type}. Focus on this specific aspect only.`
      },
      {
        role: "user",
        content: subtask.content
      }
    ],
    temperature: 0.7,
  });
  
  return {
    taskId: subtask.id,
    result: response.choices[0].message.content,
    metadata: {
      type: subtask.type,
      tokens: response.usage?.total_tokens || 0
    }
  };
}

/**
 * Process multiple subtasks in parallel while respecting dependencies
 * 
 * @param {Array<Object>} subtasks List of subtasks to process
 * @returns {Promise<Array<Object>>} List of results from all subtasks
 */
async function processTasksInParallel(subtasks) {
  const processedTasks = new Set();
  const results = [];
  
  while (processedTasks.size < subtasks.length) {
    // Find tasks whose dependencies are satisfied
    const readyTasks = subtasks.filter(task =>
      !processedTasks.has(task.id) &&
      task.dependencies.every(dep => processedTasks.has(dep))
    );
    
    if (readyTasks.length === 0) {
      throw new Error("Circular dependency detected or invalid dependency configuration");
    }
    
    // Process ready tasks in parallel
    console.log(termcolor.yellow(`Processing ${readyTasks.length} tasks in parallel`));
    const currentResults = await Promise.all(
      readyTasks.map(task => processSubtask(task))
    );
    
    results.push(...currentResults);
    readyTasks.forEach(task => processedTasks.add(task.id));
  }
  
  return results;
}

/**
 * Aggregate results from multiple subtasks into a coherent output
 * 
 * @param {Array<Object>} results List of results from subtasks
 * @returns {Promise<string>} Aggregated and summarized result
 */
async function aggregateResults(results) {
  console.log(termcolor.green("Aggregating results..."));
  
  // Prepare results for aggregation
  const resultsText = results.map(r =>
    `Task ${r.taskId} (${r.metadata.type}):\n${r.result}`
  ).join("\n\n");
  
  const response = await openai.chat.completions.create({
    model: "gpt-4o", // Using more capable model for aggregation
    messages: [
      {
        role: "system",
        content: `You are an aggregator that combines multiple task results into a coherent output.
        Synthesize the information, resolve any conflicts, and provide a clear summary.`
      },
      {
        role: "user",
        content: `Aggregate and summarize these task results:\n\n${resultsText}`
      }
    ],
    temperature: 0.5,
  });
  
  return response.choices[0].message.content;
}

/**
 * Main function that implements the Parallelization pattern
 * 
 * @param {string} task The main task to process
 * @param {Array<Object>} subtaskDefinitions List of subtask definitions with their dependencies
 * @returns {Promise<Object>} Aggregated results from all subtasks
 */
export async function parallelProcess(task, subtaskDefinitions) {
  console.log(termcolor.yellow(`Starting parallel processing for task: ${task}`));
  
  // Process subtasks in parallel
  const results = await processTasksInParallel(subtaskDefinitions);
  
  // Aggregate results
  const summary = await aggregateResults(results);
  
  // Calculate metadata
  const totalTokens = results.reduce((sum, r) => sum + (r.metadata.tokens || 0), 0);
  
  return {
    results,
    summary,
    metadata: {
      totalTasks: results.length,
      totalTokens,
      processingTimeMs: Date.now() - startTime
    }
  };
}

// Example usage
if (require.main === module) {
  (async () => {
    const task = "Analyze a research paper and provide a comprehensive review";
    const subtasks = [
      {
        id: "methodology",
        type: "analysis",
        content: "Analyze the methodology section of the paper",
        dependencies: []
      },
      {
        id: "results",
        type: "analysis",
        content: "Analyze the results section of the paper",
        dependencies: []
      },
      {
        id: "discussion",
        type: "analysis",
        content: "Analyze the discussion section of the paper",
        dependencies: ["methodology", "results"]
      },
      {
        id: "conclusion",
        type: "synthesis",
        content: "Synthesize the findings and provide recommendations",
        dependencies: ["discussion"]
      }
    ];
    
    try {
      const result = await parallelProcess(task, subtasks);
      
      console.log("\nFinal Results:");
      console.log("=============");
      console.log("\nIndividual Task Results:");
      result.results.forEach(r => {
        console.log(`\nTask ${r.taskId} (${r.metadata.type}):`);
        console.log(r.result);
      });
      
      console.log("\nAggregated Summary:");
      console.log("==================");
      console.log(result.summary);
      
      console.log("\nMetadata:");
      console.log("========");
      console.log(`Total tasks: ${result.metadata.totalTasks}`);
      console.log(`Total tokens: ${result.metadata.totalTokens}`);
      console.log(`Processing time: ${result.metadata.processingTimeMs}ms`);
      
    } catch (error) {
      console.error(termcolor.red(`Error during parallel processing: ${error.message}`));
    }
  })();
} 