/**
 * Prompt Chaining Pattern Implementation
 * 
 * This file demonstrates the Prompt Chaining pattern using the Vercel AI SDK.
 * It showcases how to chain multiple LLM calls to solve a complex problem step by step.
 */
import { openai } from '@ai-sdk/openai';
import { generateObject, generateText } from 'ai';
import { z } from 'zod';

/**
 * Analysis schema for structured output
 */
const analysisSchema = z.object({
  problemType: z.string().describe('The type or category of the math problem'),
  relevantConcepts: z.array(z.string()).min(1).describe('Mathematical concepts relevant to solving this problem'),
  suggestedApproach: z.string().describe('A high-level approach to solving the problem'),
  potentialChallenges: z.array(z.string()).describe('Potential difficulties or edge cases to consider'),
  estimatedComplexity: z.enum(['simple', 'moderate', 'complex']).describe('The estimated complexity of the problem')
});

/**
 * Solution plan schema for structured output
 */
const planSchema = z.object({
  steps: z.array(z.string()).min(1).describe('Step-by-step approach to solve the problem'),
  formulas: z.array(z.string()).describe('Mathematical formulas needed for the solution'),
  variablesToTrack: z.array(z.string()).describe('Variables that need to be tracked throughout the solution'),
  expectedResult: z.string().describe('The expected form of the final result')
});

/**
 * Main function that solves a math problem through multiple sequential prompts
 * 
 * @param {string} problem The math problem to solve
 * @returns {Promise<Object>} Object containing the original problem, analysis, plan, and final solution
 */
export async function solveMathProblem(problem) {
  console.log('Processing math problem:', problem);
  
  // Step 1: Analyze the problem
  console.log('Step 1: Analyzing problem...');
  const analysis = await analyzeProblem(problem);
  console.log('Analysis complete:', analysis);
  
  // Step 2: Create a solution plan based on analysis
  console.log('Step 2: Creating solution plan...');
  const plan = await createSolutionPlan(problem, analysis);
  console.log('Plan created:', plan);
  
  // Step 3: Generate the solution based on problem, analysis, and plan
  console.log('Step 3: Generating solution...');
  const solution = await generateSolution(problem, analysis, plan);
  console.log('Solution generation complete');
  
  // Return the complete processing result
  return {
    originalProblem: problem,
    analysis,
    plan,
    solution
  };
}

/**
 * Step 1: Analyze the math problem to understand its nature
 * 
 * @param {string} problem The math problem text
 * @returns {Promise<Object>} Analysis of the problem
 */
async function analyzeProblem(problem) {
  const model = openai('gpt-4o-mini');
  
  // Get structured analysis data
  const { object: analysis } = await generateObject({
    model,
    schema: analysisSchema,
    prompt: `Analyze the following math problem to identify its type, relevant concepts, suggested approach, potential challenges, and complexity:
    
    "${problem}"
    
    Provide a comprehensive analysis focusing on understanding the core mathematical concepts involved.`,
  });
  
  return analysis;
}

/**
 * Step 2: Create a solution plan based on the analysis
 * 
 * @param {string} problem The original math problem
 * @param {Object} analysis The analysis from step 1
 * @returns {Promise<Object>} A plan for how to solve the problem
 */
async function createSolutionPlan(problem, analysis) {
  const model = openai('gpt-4o-mini');
  
  // Get structured plan data
  const { object: plan } = await generateObject({
    model,
    schema: planSchema,
    prompt: `Based on the following problem analysis, create a detailed plan for solving this math problem:
    
    Problem: "${problem}"
    
    Analysis:
    - Problem Type: ${analysis.problemType}
    - Relevant Concepts: ${analysis.relevantConcepts.join(', ')}
    - Suggested Approach: ${analysis.suggestedApproach}
    - Potential Challenges: ${analysis.potentialChallenges.join(', ')}
    - Estimated Complexity: ${analysis.estimatedComplexity}
    
    Create a comprehensive step-by-step plan for solving this problem, including necessary formulas and variables to track.`,
  });
  
  return plan;
}

/**
 * Step 3: Generate the final solution based on all previous information
 * 
 * @param {string} problem The original math problem
 * @param {Object} analysis The analysis from step 1
 * @param {Object} plan The solution plan from step 2
 * @returns {Promise<string>} The generated solution text
 */
async function generateSolution(problem, analysis, plan) {
  // Use a more capable model for the final solution
  const model = openai('gpt-4o');
  
  // Generate the solution text
  const { text: solution } = await generateText({
    model,
    system: `You are an expert mathematics tutor. Your task is to solve the given math problem 
    following the provided plan and analysis. Show all your work step by step, explaining each step 
    clearly. Include all calculations, formulas, and reasoning. Make sure your solution is correct 
    and addresses all aspects of the problem.`,
    prompt: `Math Problem to Solve:
    "${problem}"
    
    Problem Analysis:
    - Problem Type: ${analysis.problemType}
    - Relevant Concepts: ${analysis.relevantConcepts.join(', ')}
    - Suggested Approach: ${analysis.suggestedApproach}
    - Potential Challenges: ${analysis.potentialChallenges.join(', ')}
    
    Solution Plan:
    - Steps: ${plan.steps.join('\n')}
    - Formulas Needed: ${plan.formulas.join(', ')}
    - Variables to Track: ${plan.variablesToTrack.join(', ')}
    - Expected Result: ${plan.expectedResult}
    
    Please solve this problem step by step, showing all your work and explaining your reasoning.`,
  });
  
  return solution;
}

/**
 * Example usage
 */
// For demonstration purposes only
if (require.main === module) {
  const sampleProblem = `
  A cylindrical water tank has a radius of 5 meters and a height of 12 meters. 
  If water is flowing into the tank at a rate of 3 cubic meters per minute, 
  how long will it take to fill the tank to 80% of its capacity?
  `;
  
  (async () => {
    console.log("STARTING MATH PROBLEM SOLVER");
    console.log("============================");
    
    const result = await solveMathProblem(sampleProblem);
    
    console.log("\nFINAL SOLUTION:");
    console.log("===============");
    console.log(result.solution);
    
    console.log("\nPROCESSING COMPLETE");
  })();
} 