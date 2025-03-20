/**
 * Autonomous Agent Pattern Implementation
 * 
 * This file demonstrates the Autonomous Agent pattern using the Vercel AI SDK.
 * It showcases how to create an agent that can operate independently to achieve
 * goals through a continuous cycle of perception, reasoning, and action.
 */

import { OpenAI } from 'openai';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Memory class to store agent state
class Memory {
  constructor() {
    this.state = {
      goal: '',
      observations: [],
      actions: [],
      iterations: 0,
      currentState: 'initialized'
    };
  }
  
  setGoal(goal) {
    this.state.goal = goal;
    this.state.currentState = 'goal_set';
  }
  
  addObservation(observation) {
    this.state.observations.push(observation);
    this.state.currentState = 'observed';
  }
  
  addActionResult(action, result) {
    this.state.actions.push({ action, result });
    this.state.iterations++;
    this.state.currentState = 'action_taken';
  }
  
  getRecentObservations(count = 5) {
    return this.state.observations.slice(-count);
  }
  
  getRecentActions(count = 3) {
    return this.state.actions.slice(-count);
  }
  
  getFullState() {
    return { ...this.state };
  }
  
  getSummaryState() {
    const recentObservations = this.getRecentObservations().map(o => o.content).join('\n');
    const recentActions = this.getRecentActions().map(a => 
      `Action: ${a.action.type} - Result: ${a.result.status}`
    ).join('\n');
    
    return `Goal: ${this.state.goal}
Iterations: ${this.state.iterations}
Current State: ${this.state.currentState}
Recent Observations:
${recentObservations}
Recent Actions:
${recentActions}`;
  }
  
  get iterations() {
    return this.state.iterations;
  }
  
  getFinalResult() {
    const lastAction = this.state.actions[this.state.actions.length - 1];
    return lastAction ? lastAction.result : {
      status: 'failure',
      output: 'No actions were taken',
      goalAchieved: false
    };
  }
}

// Web search environment implementation
class WebSearchEnvironment {
  constructor(initialContext) {
    this.context = initialContext;
    this.searchResults = [];
  }
  
  async observe() {
    // In a real implementation, this might fetch updated information
    // For this example, we'll just return the current context
    return {
      timestamp: Date.now(),
      content: `Current context: ${this.context}\nSearch results: ${this.searchResults.join('\n')}`,
      type: 'environment'
    };
  }
  
  async executeAction(action) {
    console.log(`Executing action: ${action.type}`);
    
    switch (action.type) {
      case 'search':
        return await this.performSearch(action.parameters.query);
      case 'analyze':
        return await this.analyzeInformation(action.parameters.topic);
      case 'summarize':
        return await this.summarizeFindings();
      case 'conclude':
        return this.concludeTask(action.parameters.conclusion);
      default:
        return {
          status: 'failure',
          output: `Unknown action type: ${action.type}`,
          goalAchieved: false
        };
    }
  }
  
  async performSearch(query) {
    console.log(`Performing search for: ${query}`);
    
    try {
      // Simulate a web search using the LLM
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "You are a search engine. Provide realistic search results for the given query. Return 3-5 relevant results with titles and brief descriptions."
          },
          {
            role: "user",
            content: query
          }
        ],
        temperature: 0.7,
      });
      
      const searchResult = response.choices[0].message.content || "";
      this.searchResults.push(`Query: ${query}\nResults: ${searchResult}`);
      
      return {
        status: 'success',
        output: searchResult,
        goalAchieved: false,
        feedback: 'Search completed successfully'
      };
    } catch (error) {
      console.error('Search error:', error);
      return {
        status: 'failure',
        output: `Error performing search: ${error}`,
        goalAchieved: false
      };
    }
  }
  
  async analyzeInformation(topic) {
    console.log(`Analyzing information about: ${topic}`);
    
    if (this.searchResults.length === 0) {
      return {
        status: 'failure',
        output: 'No search results to analyze. Perform a search first.',
        goalAchieved: false
      };
    }
    
    try {
      // Use the LLM to analyze the search results
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "You are an analytical assistant. Analyze the provided search results and extract key insights about the topic."
          },
          {
            role: "user",
            content: `Topic: ${topic}\nSearch Results:\n${this.searchResults.join('\n\n')}\n\nPlease analyze these results and provide key insights.`
          }
        ],
        temperature: 0.5,
      });
      
      const analysis = response.choices[0].message.content || "";
      
      return {
        status: 'success',
        output: analysis,
        goalAchieved: false,
        feedback: 'Analysis completed successfully'
      };
    } catch (error) {
      console.error('Analysis error:', error);
      return {
        status: 'failure',
        output: `Error analyzing information: ${error}`,
        goalAchieved: false
      };
    }
  }
  
  async summarizeFindings() {
    console.log('Summarizing findings');
    
    if (this.searchResults.length === 0) {
      return {
        status: 'failure',
        output: 'No search results to summarize',
        goalAchieved: false
      };
    }
    
    try {
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: "You are a summarization expert. Create a concise summary of the search results and analysis."
          },
          {
            role: "user",
            content: `Search Results and Analysis:\n${this.searchResults.join('\n\n')}\n\nPlease provide a comprehensive summary.`
          }
        ],
        temperature: 0.5,
      });
      
      const summary = response.choices[0].message.content || "";
      
      return {
        status: 'success',
        output: summary,
        goalAchieved: false,
        feedback: 'Summary generated successfully'
      };
    } catch (error) {
      console.error('Summarization error:', error);
      return {
        status: 'failure',
        output: `Error summarizing findings: ${error}`,
        goalAchieved: false
      };
    }
  }
  
  concludeTask(conclusion) {
    console.log('Concluding task');
    
    return {
      status: 'success',
      output: conclusion,
      goalAchieved: true,
      feedback: 'Task completed successfully'
    };
  }
}

export class AutonomousAgent {
  constructor(environment, maxIterations = 10) {
    this.memory = new Memory();
    this.environment = environment;
    this.maxIterations = maxIterations;
  }
  
  async run(goal) {
    console.log(`Starting autonomous agent with goal: ${goal}`);
    this.memory.setGoal(goal);
    
    while (this.memory.iterations < this.maxIterations) {
      // Observe the environment
      const observation = await this.environment.observe();
      this.memory.addObservation(observation);
      
      // Reason about next action
      const action = await this.reason();
      
      // Execute the action
      const result = await this.environment.executeAction(action);
      this.memory.addActionResult(action, result);
      
      // Check if goal is achieved
      if (this.evaluateCompletion(result)) {
        console.log('Goal achieved!');
        return result;
      }
      
      console.log(`Completed iteration ${this.memory.iterations}`);
    }
    
    console.log('Max iterations reached without achieving goal');
    return this.memory.getFinalResult();
  }
  
  async reason() {
    const state = this.memory.getSummaryState();
    
    try {
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are an autonomous agent tasked with achieving a goal.
Based on the current state, determine the next best action to take.
Available actions:
- search: Perform a web search (parameters: query)
- analyze: Analyze information about a topic (parameters: topic)
- summarize: Summarize current findings (no parameters)
- conclude: Conclude the task with final results (parameters: conclusion)

Return your response in JSON format with three fields:
- type: The action type to take
- parameters: Object containing action parameters
- reasoning: Your reasoning for choosing this action`
          },
          {
            role: "user",
            content: `Current State:\n${state}\n\nWhat action should I take next?`
          }
        ],
        temperature: 0.7,
      });
      
      const actionPlan = JSON.parse(response.choices[0].message.content || "{}");
      return actionPlan;
      
    } catch (error) {
      console.error('Reasoning error:', error);
      return {
        type: 'conclude',
        parameters: {
          conclusion: 'Failed to determine next action due to reasoning error'
        },
        reasoning: 'Error occurred during reasoning process'
      };
    }
  }
  
  evaluateCompletion(result) {
    return result.goalAchieved || result.status === 'success' && result.type === 'conclude';
  }
}

// Example usage
if (require.main === module) {
  (async () => {
    const environment = new WebSearchEnvironment('Initial research context');
    const agent = new AutonomousAgent(environment);
    
    const goal = 'Research the latest developments in quantum computing and provide a summary';
    const result = await agent.run(goal);
    
    console.log('\nFinal Result:');
    console.log('=============');
    console.log(result.output);
  })();
} 