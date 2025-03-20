"""
Evaluator-Optimizer Agent Pattern Implementation

This file demonstrates the Evaluator-Optimizer pattern using the OpenAI API.
It showcases how to implement a feedback loop where generated solutions are
evaluated against specific criteria and refined until they meet quality standards.
"""

import os
import json
import time
from typing import Dict, List, TypedDict, Union, Any, Optional, Literal
import openai
from termcolor import colored

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Type definitions
class CriteriaDetails(TypedDict):
    name: str
    description: str
    weight: int  # 1-10 scale

class CriterionResult(TypedDict):
    passed: bool
    score: float  # 0-10 scale
    feedback: str

class EvaluationResult(TypedDict):
    meetsAllCriteria: bool
    criteriaResults: Dict[str, CriterionResult]
    overallScore: float  # 0-10 scale
    feedback: str

class SolutionWithEvaluation(TypedDict):
    solution: str
    evaluation: EvaluationResult

class OptimizationResult(TypedDict):
    originalPrompt: str
    finalSolution: str
    evaluationResult: EvaluationResult
    attempts: int
    success: bool
    allSolutions: List[SolutionWithEvaluation]
    metadata: Dict[str, Any]

# Constants
MAX_ATTEMPTS = 5
MINIMUM_ACCEPTABLE_SCORE = 7.5

def generate_solution(prompt: str) -> Dict[str, Any]:
    """
    Generates an initial solution based on the prompt
    
    Args:
        prompt: The prompt for generating a solution
        
    Returns:
        Dictionary containing the generated content and usage statistics
    """
    print(colored("Generating initial solution...", "blue"))
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a solution generator. Your task is to generate high-quality solutions based on the given prompt. Focus on accuracy, completeness, relevance, clarity, and conciseness."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
    )
    
    return {
        "content": response.choices[0].message.content,
        "usage": response.usage.total_tokens if response.usage else 0
    }

def evaluate_solution(
    solution: str,
    criteria_details: List[CriteriaDetails],
    original_prompt: str
) -> Dict[str, Union[EvaluationResult, int]]:
    """
    Evaluates a solution against specified criteria
    
    Args:
        solution: The solution to evaluate
        criteria_details: Detailed criteria for evaluation
        original_prompt: The original prompt for context
        
    Returns:
        Dictionary containing the evaluation result and tokens used
    """
    print(colored("Evaluating solution...", "green"))
    
    # Prepare criteria for the prompt
    criteria_prompt = "\n".join([
        f"- {c['name']} (Weight: {c['weight']}/10): {c['description']}"
        for c in criteria_details
    ])
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are an evaluator that assesses solutions against specific criteria. 
                You will be given a solution and the criteria for evaluation. 
                For each criterion, provide a score (0-10), whether it passed (score >= 7.5), and specific feedback.
                Calculate the overall score as a weighted average of individual criteria scores.
                A solution meets all criteria if all individual criteria pass.
                Provide constructive feedback for improvement."""
            },
            {
                "role": "user",
                "content": f"""Original Prompt: {original_prompt}
                
                Solution to Evaluate:
                {solution}
                
                Evaluation Criteria:
                {criteria_prompt}
                
                Please evaluate this solution against each criterion and provide your assessment in JSON format with the following structure:
                {{
                  "meetsAllCriteria": boolean,
                  "criteriaResults": {{
                    "criterionName": {{
                      "passed": boolean,
                      "score": number,
                      "feedback": "specific feedback"
                    }},
                    ...
                  }},
                  "overallScore": number,
                  "feedback": "overall feedback for improvement"
                }}"""
            }
        ],
        temperature=0.2,
    )
    
    evaluation_result = json.loads(response.choices[0].message.content)
    
    # Print evaluation summary
    print(colored(f"Overall Score: {evaluation_result['overallScore']:.2f}/10", "cyan"))
    print(colored(f"Meets All Criteria: {'Yes' if evaluation_result['meetsAllCriteria'] else 'No'}", "cyan"))
    
    return {
        "result": evaluation_result,
        "tokensUsed": response.usage.total_tokens if response.usage else 0
    }

def generate_improved_solution(
    original_prompt: str,
    previous_solution: str,
    feedback: str
) -> Dict[str, Any]:
    """
    Generates an improved solution based on feedback
    
    Args:
        original_prompt: The original prompt
        previous_solution: The previous solution
        feedback: Feedback for improvement
        
    Returns:
        Dictionary containing the improved solution and usage statistics
    """
    print(colored("Generating improved solution based on feedback...", "blue"))
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an optimizer that improves solutions based on feedback. 
                Your task is to generate an improved version of a solution, addressing the specific feedback provided.
                Focus on making targeted improvements while maintaining the overall structure and strengths of the original solution."""
            },
            {
                "role": "user",
                "content": f"""Original Prompt: {original_prompt}
                
                Previous Solution:
                {previous_solution}
                
                Feedback for Improvement:
                {feedback}
                
                Please generate an improved solution that addresses this feedback."""
            }
        ],
        temperature=0.7,
    )
    
    return {
        "content": response.choices[0].message.content,
        "usage": response.usage.total_tokens if response.usage else 0
    }

def generate_with_evaluation(
    prompt: str,
    criteria_details: List[CriteriaDetails]
) -> OptimizationResult:
    """
    Main function that implements the Evaluator-Optimizer pattern
    
    Args:
        prompt: The initial prompt for generating a solution
        criteria_details: Detailed criteria for evaluation
        
    Returns:
        The optimization result with the final solution and evaluation details
    """
    print(colored(f"Starting optimization process for prompt: {prompt[:50]}...", "yellow"))
    
    start_time = time.time()
    total_tokens_used = 0
    all_solutions = []
    
    # Initial solution generation
    solution = generate_solution(prompt)
    total_tokens_used += solution["usage"]
    
    # Evaluate the initial solution
    evaluation = evaluate_solution(solution["content"], criteria_details, prompt)
    total_tokens_used += evaluation["tokensUsed"]
    
    all_solutions.append({
        "solution": solution["content"],
        "evaluation": evaluation["result"]
    })
    
    attempts = 1
    
    # Optimization loop
    while not evaluation["result"]["meetsAllCriteria"] and attempts < MAX_ATTEMPTS:
        print(colored(f"Attempt {attempts + 1}: Generating improved solution based on feedback", "yellow"))
        
        # Generate improved solution based on feedback
        improved_solution = generate_improved_solution(
            prompt,
            solution["content"],
            evaluation["result"]["feedback"]
        )
        total_tokens_used += improved_solution["usage"]
        
        # Update current solution
        solution = improved_solution
        
        # Re-evaluate the improved solution
        evaluation = evaluate_solution(solution["content"], criteria_details, prompt)
        total_tokens_used += evaluation["tokensUsed"]
        
        all_solutions.append({
            "solution": solution["content"],
            "evaluation": evaluation["result"]
        })
        
        attempts += 1
    
    execution_time_ms = (time.time() - start_time) * 1000
    
    return {
        "originalPrompt": prompt,
        "finalSolution": solution["content"],
        "evaluationResult": evaluation["result"],
        "attempts": attempts,
        "success": evaluation["result"]["meetsAllCriteria"],
        "allSolutions": all_solutions,
        "metadata": {
            "totalTokensUsed": total_tokens_used,
            "executionTimeMs": execution_time_ms
        }
    }

if __name__ == "__main__":
    # Example usage
    prompt = "Create a comprehensive marketing strategy for a new eco-friendly smartphone case targeting environmentally conscious millennials."
    
    criteria_details = [
        {
            "name": "accuracy",
            "description": "Information is factually correct and strategies are based on sound marketing principles",
            "weight": 9
        },
        {
            "name": "completeness",
            "description": "Covers all essential aspects of a marketing strategy including target audience, positioning, channels, messaging, and metrics",
            "weight": 8
        },
        {
            "name": "relevance",
            "description": "Specifically addresses eco-friendly aspects and millennial preferences",
            "weight": 10
        },
        {
            "name": "clarity",
            "description": "Ideas are expressed clearly and the strategy is easy to understand",
            "weight": 7
        },
        {
            "name": "conciseness",
            "description": "Information is presented efficiently without unnecessary details",
            "weight": 6
        },
        {
            "name": "actionability",
            "description": "Provides concrete, implementable steps rather than vague suggestions",
            "weight": 9
        }
    ]
    
    print(colored("STARTING EVALUATOR-OPTIMIZER PROCESS", "cyan", attrs=["bold"]))
    print(colored("=====================================", "cyan", attrs=["bold"]))
    
    try:
        result = generate_with_evaluation(prompt, criteria_details)
        
        print(colored("\n=== OPTIMIZATION RESULTS ===", "magenta", attrs=["bold"]))
        print(colored(f"Attempts: {result['attempts']}", "white"))
        print(colored(f"Success: {result['success']}", "white"))
        print(colored(f"Overall Score: {result['evaluationResult']['overallScore']:.2f}/10", "white"))
        print(colored(f"Execution Time: {(result['metadata']['executionTimeMs'] / 1000):.2f} seconds", "white"))
        print(colored(f"Tokens Used: {result['metadata']['totalTokensUsed']}", "white"))
        
        print(colored("\n=== FINAL SOLUTION ===", "magenta", attrs=["bold"]))
        print(result["finalSolution"])
        
        print(colored("\n=== EVALUATION FEEDBACK ===", "magenta", attrs=["bold"]))
        print(result["evaluationResult"]["feedback"])
        
        print(colored("\nPROCESSING COMPLETE", "cyan", attrs=["bold"]))
    except Exception as e:
        print(colored(f"Error in optimization process: {str(e)}", "red"))