/**
 * Orchestrator-Workers Agent Pattern Implementation
 * 
 * This file demonstrates the Orchestrator-Workers pattern using the OpenAI API.
 * It showcases how to implement a central orchestrator that directs multiple worker LLMs
 * to perform subtasks, synthesizing their outputs for complex, coordinated operations.
 */

import { OpenAI } from 'openai';
import { termcolor } from 'termcolor';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Create a plan for achieving the goal using available workers
 * 
 * @param {string} goal The goal to achieve
 * @returns {Promise<Object>} A structured plan with tasks and dependencies
 */
async function createExecutionPlan(goal) {
  console.log(termcolor.blue("Creating execution plan..."));
  
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You are a task orchestrator. Break down the given goal into subtasks
        that can be assigned to worker LLMs. Define dependencies between tasks and specify
        success criteria.
        
        Available worker types:
        - researcher: Gathers and analyzes information
        - synthesizer: Combines and summarizes information
        - validator: Checks outputs against requirements
        - specialist: Performs domain-specific tasks
        
        Return your plan in JSON format with:
        {
            "tasks": [
                {
                    "id": string,
                    "type": string (worker type),
                    "input": string (task description),
                    "context": object (relevant context),
                    "requirements": string[] (specific requirements)
                }
            ],
            "dependencies": {
                "taskId": string[] (ids of required tasks)
            },
            "success_criteria": object (criteria for success)
        }`
      },
      {
        role: "user",
        content: `Create an execution plan for this goal: ${goal}`
      }
    ],
    temperature: 0.7,
  });
  
  return JSON.parse(response.choices[0].message.content);
}

/**
 * Assign a task to an appropriate worker LLM
 * 
 * @param {Object} task The task to be performed
 * @returns {Promise<Object>} The result from the worker
 */
async function assignWorker(task) {
  console.log(termcolor.cyan(`Assigning task ${task.id} to ${task.type} worker`));
  
  // Select appropriate model and configuration based on task type
  let model, systemPrompt;
  
  switch (task.type) {
    case 'researcher':
      model = "gpt-4o";
      systemPrompt = `You are a research worker. Gather and analyze information thoroughly.
      Focus on accuracy and completeness. Provide citations where possible.`;
      break;
    case 'synthesizer':
      model = "gpt-4o";
      systemPrompt = `You are a synthesis worker. Combine information from multiple sources
      into coherent, well-structured outputs. Focus on clarity and logical flow.`;
      break;
    case 'validator':
      model = "gpt-4o-mini";
      systemPrompt = `You are a validation worker. Check outputs against specified requirements.
      Be thorough and explicit about any issues found.`;
      break;
    default:  // specialist
      model = "gpt-4o";
      systemPrompt = `You are a specialist worker. Apply domain expertise to solve specific problems.
      Focus on accuracy and practical applicability.`;
  }
  
  const response = await openai.chat.completions.create({
    model,
    messages: [
      {
        role: "system",
        content: systemPrompt
      },
      {
        role: "user",
        content: `Task: ${task.input}
        
        Requirements:
        ${task.requirements.join(', ')}
        
        Context:
        ${JSON.stringify(task.context)}`
      }
    ],
    temperature: 0.5,
  });
  
  return {
    taskId: task.id,
    output: response.choices[0].message.content,
    metadata: {
      type: task.type,
      tokens: response.usage?.total_tokens || 0
    },
    status: 'success'
  };
}

/**
 * Synthesize results from multiple workers into a final output
 * 
 * @param {Array<Object>} results List of results from workers
 * @param {string} goal The original goal
 * @returns {Promise<string>} Synthesized final output
 */
async function synthesizeResults(results, goal) {
  console.log(termcolor.green("Synthesizing final results..."));
  
  const resultsText = results.map(r =>
    `Task ${r.taskId} (${r.metadata.type}):\n${r.output}`
  ).join("\n\n");
  
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: `You are a final synthesizer. Combine all worker outputs into a
        coherent final result that achieves the original goal. Ensure all key points
        are included and properly connected.`
      },
      {
        role: "user",
        content: `Original Goal: ${goal}

        Worker Results:
        ${resultsText}
        
        Please synthesize these results into a final output that achieves the goal.`
      }
    ],
    temperature: 0.5,
  });
  
  return response.choices[0].message.content;
}

/**
 * Main function that implements the Orchestrator-Workers pattern
 * 
 * @param {string} goal The goal to achieve
 * @returns {Promise<Object>} Final results and execution metadata
 */
export async function orchestrateTask(goal) {
  console.log(termcolor.yellow(`Starting orchestration for goal: ${goal}`));
  
  // Create execution plan
  const plan = await createExecutionPlan(goal);
  
  // Track completed tasks and their results
  const completedTasks = new Map();
  const allResults = [];
  
  // Execute tasks in dependency order
  while (completedTasks.size < plan.tasks.length) {
    // Find tasks whose dependencies are met
    const availableTasks = plan.tasks.filter(task =>
      !completedTasks.has(task.id) &&
      (plan.dependencies[task.id] || []).every(dep => completedTasks.has(dep))
    );
    
    if (availableTasks.length === 0) {
      throw new Error("Circular dependency detected or invalid task configuration");
    }
    
    // Process available tasks
    for (const task of availableTasks) {
      // Add results from dependencies to task context
      task.context.dependencyResults = {};
      (plan.dependencies[task.id] || []).forEach(dep => {
        task.context.dependencyResults[dep] = completedTasks.get(dep).output;
      });
      
      // Assign task to worker
      const result = await assignWorker(task);
      completedTasks.set(task.id, result);
      allResults.push(result);
    }
  }
  
  // Synthesize final results
  const finalOutput = await synthesizeResults(allResults, goal);
  
  // Calculate metadata
  const totalTokens = allResults.reduce((sum, r) => sum + (r.metadata.tokens || 0), 0);
  
  return {
    goal,
    finalOutput,
    workerResults: allResults,
    executionPlan: plan,
    metadata: {
      totalTasks: allResults.length,
      totalTokens,
      success: allResults.every(r => r.status === 'success')
    }
  };
}

// Example usage
if (require.main === module) {
  (async () => {
    const goal = `Create a comprehensive market analysis report for a new smartphone app,
    including target audience analysis, competitor research, and pricing strategy.`;
    
    try {
      const result = await orchestrateTask(goal);
      
      console.log("\nFinal Results:");
      console.log("=============");
      console.log(result.finalOutput);
      
      console.log("\nExecution Statistics:");
      console.log("====================");
      console.log(`Total tasks: ${result.metadata.totalTasks}`);
      console.log(`Total tokens: ${result.metadata.totalTokens}`);
      console.log(`Overall success: ${result.metadata.success}`);
      
    } catch (error) {
      console.error(termcolor.red(`Error during orchestration: ${error.message}`));
    }
  })();
} 