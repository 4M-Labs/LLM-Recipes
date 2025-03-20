/**
 * Evaluator-Optimizer Agent Pattern Implementation
 * 
 * This file demonstrates the Evaluator-Optimizer pattern using the OpenAI API.
 * It showcases how to implement a feedback loop where generated solutions are
 * evaluated against specific criteria and refined until they meet quality standards.
 */

import { OpenAI } from 'openai';
import { termcolor } from 'termcolor';

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Constants
const MAX_ATTEMPTS = 5;
const MINIMUM_ACCEPTABLE_SCORE = 7.5;

/**
 * Generates an initial solution based on the prompt
 * 
 * @param {string} prompt The prompt for generating a solution
 * @returns {Promise<Object>} Dictionary containing the generated content and usage statistics
 */
async function generateSolution(prompt) {
  console.log(termcolor.blue("Generating initial solution..."));
  
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: "You are a solution generator. Your task is to generate high-quality solutions based on the given prompt. Focus on accuracy, completeness, relevance, clarity, and conciseness."
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0.7,
  });
  
  return {
    content: response.choices[0].message.content,
    usage: response.usage?.total_tokens || 0
  };
}

/**
 * Evaluates a solution against specified criteria
 * 
 * @param {string} solution The solution to evaluate
 * @param {Array<Object>} criteriaDetails Detailed criteria for evaluation
 * @param {string} originalPrompt The original prompt for context
 * @returns {Promise<Object>} Dictionary containing the evaluation result and tokens used
 */
async function evaluateSolution(solution, criteriaDetails, originalPrompt) {
  console.log(termcolor.green("Evaluating solution..."));
  
  // Prepare criteria for the prompt
  const criteriaPrompt = criteriaDetails.map(c => 
    `- ${c.name} (Weight: ${c.weight}/10): ${c.description}`
  ).join('\n');
  
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You are an evaluator that assesses solutions against specific criteria. 
        You will be given a solution and the criteria for evaluation. 
        For each criterion, provide a score (0-10), whether it passed (score >= 7.5), and specific feedback.
        Calculate the overall score as a weighted average of individual criteria scores.
        A solution meets all criteria if all individual criteria pass.
        Provide constructive feedback for improvement.`
      },
      {
        role: "user",
        content: `Original Prompt: ${originalPrompt}
        
        Solution to Evaluate:
        ${solution}
        
        Evaluation Criteria:
        ${criteriaPrompt}
        
        Please evaluate this solution against each criterion and provide your assessment in JSON format with the following structure:
        {
          "meetsAllCriteria": boolean,
          "criteriaResults": {
            "criterionName": {
              "passed": boolean,
              "score": number,
              "feedback": "specific feedback"
            },
            ...
          },
          "overallScore": number,
          "feedback": "overall feedback for improvement"
        }`
      }
    ],
    temperature: 0.2,
  });
  
  const evaluationResult = JSON.parse(response.choices[0].message.content);
  
  // Print evaluation summary
  console.log(termcolor.cyan(`Overall Score: ${evaluationResult.overallScore.toFixed(2)}/10`));
  console.log(termcolor.cyan(`Meets All Criteria: ${evaluationResult.meetsAllCriteria ? 'Yes' : 'No'}`));
  
  return {
    result: evaluationResult,
    tokensUsed: response.usage?.total_tokens || 0
  };
}

/**
 * Generates an improved solution based on feedback
 * 
 * @param {string} originalPrompt The original prompt
 * @param {string} previousSolution The previous solution
 * @param {string} feedback Feedback for improvement
 * @returns {Promise<Object>} Dictionary containing the improved solution and usage statistics
 */
async function generateImprovedSolution(originalPrompt, previousSolution, feedback) {
  console.log(termcolor.blue("Generating improved solution based on feedback..."));
  
  const response = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: `You are an optimizer that improves solutions based on feedback. 
        Your task is to generate an improved version of a solution, addressing the specific feedback provided.
        Focus on making targeted improvements while maintaining the overall structure and strengths of the original solution.`
      },
      {
        role: "user",
        content: `Original Prompt: ${originalPrompt}
        
        Previous Solution:
        ${previousSolution}
        
        Feedback for Improvement:
        ${feedback}
        
        Please generate an improved solution that addresses this feedback.`
      }
    ],
    temperature: 0.7,
  });
  
  return {
    content: response.choices[0].message.content,
    usage: response.usage?.total_tokens || 0
  };
}

/**
 * Main function that implements the Evaluator-Optimizer pattern
 * 
 * @param {string} prompt The initial prompt for generating a solution
 * @param {Array<Object>} criteriaDetails Detailed criteria for evaluation
 * @returns {Promise<Object>} The optimization result with the final solution and evaluation details
 */
export async function generateWithEvaluation(prompt, criteriaDetails) {
  console.log(termcolor.yellow(`Starting optimization process for prompt: ${prompt.slice(0, 50)}...`));
  
  const startTime = Date.now();
  let totalTokensUsed = 0;
  const allSolutions = [];
  
  // Initial solution generation
  const solution = await generateSolution(prompt);
  totalTokensUsed += solution.usage;
  
  // Evaluate the initial solution
  const evaluation = await evaluateSolution(solution.content, criteriaDetails, prompt);
  totalTokensUsed += evaluation.tokensUsed;
  
  allSolutions.push({
    solution: solution.content,
    evaluation: evaluation.result
  });
  
  let attempts = 1;
  
  // Optimization loop
  while (!evaluation.result.meetsAllCriteria && attempts < MAX_ATTEMPTS) {
    console.log(termcolor.yellow(`Attempt ${attempts + 1}: Generating improved solution based on feedback`));
    
    // Generate improved solution based on feedback
    const improvedSolution = await generateImprovedSolution(
      prompt,
      allSolutions[allSolutions.length - 1].solution,
      evaluation.result.feedback
    );
    totalTokensUsed += improvedSolution.usage;
    
    // Evaluate the improved solution
    const improvedEvaluation = await evaluateSolution(improvedSolution.content, criteriaDetails, prompt);
    totalTokensUsed += improvedEvaluation.tokensUsed;
    
    allSolutions.push({
      solution: improvedSolution.content,
      evaluation: improvedEvaluation.result
    });
    
    // Update evaluation for next iteration check
    evaluation.result = improvedEvaluation.result;
    attempts++;
  }
  
  const endTime = Date.now();
  const success = evaluation.result.meetsAllCriteria || evaluation.result.overallScore >= MINIMUM_ACCEPTABLE_SCORE;
  
  // Return final optimization result
  return {
    originalPrompt: prompt,
    finalSolution: allSolutions[allSolutions.length - 1].solution,
    evaluationResult: evaluation.result,
    attempts,
    success,
    allSolutions,
    metadata: {
      totalTokensUsed,
      processingTimeMs: endTime - startTime,
      finalAttempt: attempts,
      averageScore: allSolutions.reduce((sum, s) => sum + s.evaluation.overallScore, 0) / allSolutions.length
    }
  };
}

// Example usage
if (require.main === module) {
  (async () => {
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
    
    try {
      const result = await generateWithEvaluation(prompt, criteriaDetails);
      
      console.log('\nFinal Result:');
      console.log('=============');
      console.log('Success:', result.success);
      console.log('Attempts:', result.attempts);
      console.log('Final Score:', result.evaluationResult.overallScore.toFixed(2));
      console.log('\nFinal Solution:');
      console.log(result.finalSolution);
      
    } catch (error) {
      console.error('Error during optimization process:', error);
    }
  })();
} 