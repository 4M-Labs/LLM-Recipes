"""
Prompt Chaining Pattern Implementation

This file demonstrates the Prompt Chaining pattern using the OpenAI API.
It showcases how to chain multiple LLM calls to solve a complex problem step by step.
"""

import os
import json
from typing import Dict, List, Literal, TypedDict, Union
import openai
from termcolor import colored

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Type definitions
class ProblemAnalysis(TypedDict):
    problem_type: str
    relevant_concepts: List[str]
    suggested_approach: str
    potential_challenges: List[str]
    estimated_complexity: Literal["simple", "moderate", "complex"]

class SolutionPlan(TypedDict):
    steps: List[str]
    formulas: List[str]
    variables_to_track: List[str]
    expected_result: str

class MathSolution(TypedDict):
    original_problem: str
    analysis: ProblemAnalysis
    plan: SolutionPlan
    solution: str

def analyze_problem(problem: str) -> ProblemAnalysis:
    """
    Step 1: Analyze the math problem to understand its nature
    
    Args:
        problem: The math problem text
        
    Returns:
        Analysis of the problem
    """
    print(colored("Step 1: Analyzing problem...", "blue"))
    
    # Define the analysis schema for structured output
    analysis_schema = {
        "type": "object",
        "properties": {
            "problem_type": {
                "type": "string",
                "description": "The type or category of the math problem"
            },
            "relevant_concepts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Mathematical concepts relevant to solving this problem"
            },
            "suggested_approach": {
                "type": "string",
                "description": "A high-level approach to solving the problem"
            },
            "potential_challenges": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential difficulties or edge cases to consider"
            },
            "estimated_complexity": {
                "type": "string",
                "enum": ["simple", "moderate", "complex"],
                "description": "The estimated complexity of the problem"
            }
        },
        "required": ["problem_type", "relevant_concepts", "suggested_approach", "potential_challenges", "estimated_complexity"]
    }
    
    # Call OpenAI API with structured output
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a mathematics expert. Analyze the given math problem and provide structured output according to the specified format."},
            {"role": "user", "content": f"""Analyze the following math problem to identify its type, relevant concepts, suggested approach, potential challenges, and complexity:
            
            "{problem}"
            
            Provide a comprehensive analysis focusing on understanding the core mathematical concepts involved.
            
            Return your analysis as a JSON object with the following fields:
            - problem_type: The type or category of the math problem
            - relevant_concepts: Array of mathematical concepts relevant to solving this problem
            - suggested_approach: A high-level approach to solving the problem
            - potential_challenges: Array of potential difficulties or edge cases to consider
            - estimated_complexity: The estimated complexity of the problem (one of: "simple", "moderate", "complex")
            """}
        ]
    )
    
    # Parse the response
    analysis = json.loads(response.choices[0].message.content)
    print(colored("Analysis complete:", "green"))
    print(json.dumps(analysis, indent=2))
    
    return analysis

def create_solution_plan(problem: str, analysis: ProblemAnalysis) -> SolutionPlan:
    """
    Step 2: Create a solution plan based on the analysis
    
    Args:
        problem: The original math problem
        analysis: The analysis from step 1
        
    Returns:
        A plan for how to solve the problem
    """
    print(colored("\nStep 2: Creating solution plan...", "blue"))
    
    # Define the plan schema for structured output
    plan_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Step-by-step approach to solve the problem"
            },
            "formulas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Mathematical formulas needed for the solution"
            },
            "variables_to_track": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variables that need to be tracked throughout the solution"
            },
            "expected_result": {
                "type": "string",
                "description": "The expected form of the final result"
            }
        },
        "required": ["steps", "formulas", "variables_to_track", "expected_result"]
    }
    
    # Call OpenAI API with structured output
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a mathematics expert. Create a detailed plan for solving the given math problem based on the provided analysis."},
            {"role": "user", "content": f"""Based on the following problem analysis, create a detailed plan for solving this math problem:
            
            Problem: "{problem}"
            
            Analysis:
            - Problem Type: {analysis['problem_type']}
            - Relevant Concepts: {', '.join(analysis['relevant_concepts'])}
            - Suggested Approach: {analysis['suggested_approach']}
            - Potential Challenges: {', '.join(analysis['potential_challenges'])}
            - Estimated Complexity: {analysis['estimated_complexity']}
            
            Create a comprehensive step-by-step plan for solving this problem, including necessary formulas and variables to track.
            
            Return your plan as a JSON object with the following fields:
            - steps: Array of step-by-step instructions to solve the problem
            - formulas: Array of mathematical formulas needed for the solution
            - variables_to_track: Array of variables that need to be tracked throughout the solution
            - expected_result: The expected form of the final result
            """}
        ]
    )
    
    # Parse the response
    plan = json.loads(response.choices[0].message.content)
    print(colored("Plan created:", "green"))
    print(json.dumps(plan, indent=2))
    
    return plan

def generate_solution(problem: str, analysis: ProblemAnalysis, plan: SolutionPlan) -> str:
    """
    Step 3: Generate the final solution based on all previous information
    
    Args:
        problem: The original math problem
        analysis: The analysis from step 1
        plan: The solution plan from step 2
        
    Returns:
        The generated solution text
    """
    print(colored("\nStep 3: Generating solution...", "blue"))
    
    # Call OpenAI API for the final solution
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """You are an expert mathematics tutor. Your task is to solve the given math problem 
            following the provided plan and analysis. Show all your work step by step, explaining each step 
            clearly. Include all calculations, formulas, and reasoning. Make sure your solution is correct 
            and addresses all aspects of the problem."""},
            {"role": "user", "content": f"""Math Problem to Solve:
            "{problem}"
            
            Problem Analysis:
            - Problem Type: {analysis['problem_type']}
            - Relevant Concepts: {', '.join(analysis['relevant_concepts'])}
            - Suggested Approach: {analysis['suggested_approach']}
            - Potential Challenges: {', '.join(analysis['potential_challenges'])}
            
            Solution Plan:
            - Steps: {chr(10).join(plan['steps'])}
            - Formulas Needed: {', '.join(plan['formulas'])}
            - Variables to Track: {', '.join(plan['variables_to_track'])}
            - Expected Result: {plan['expected_result']}
            
            Please solve this problem step by step, showing all your work and explaining your reasoning."""}
        ]
    )
    
    # Get the solution text
    solution = response.choices[0].message.content
    print(colored("Solution generation complete", "green"))
    
    return solution

def solve_math_problem(problem: str) -> MathSolution:
    """
    Main function that solves a math problem through multiple sequential prompts
    
    Args:
        problem: The math problem to solve
        
    Returns:
        Object containing the original problem, analysis, plan, and final solution
    """
    print(colored(f"Processing math problem: {problem}", "yellow"))
    
    # Step 1: Analyze the problem
    analysis = analyze_problem(problem)
    
    # Step 2: Create a solution plan based on analysis
    plan = create_solution_plan(problem, analysis)
    
    # Step 3: Generate the solution based on problem, analysis, and plan
    solution = generate_solution(problem, analysis, plan)
    
    # Return the complete processing result
    return {
        "original_problem": problem,
        "analysis": analysis,
        "plan": plan,
        "solution": solution
    }

if __name__ == "__main__":
    # Example usage
    sample_problem = """
    A cylindrical water tank has a radius of 5 meters and a height of 12 meters. 
    If water is flowing into the tank at a rate of 3 cubic meters per minute, 
    how long will it take to fill the tank to 80% of its capacity?
    """
    
    print(colored("STARTING MATH PROBLEM SOLVER", "cyan", attrs=["bold"]))
    print(colored("============================", "cyan", attrs=["bold"]))
    
    result = solve_math_problem(sample_problem)
    
    print(colored("\nFINAL SOLUTION:", "magenta", attrs=["bold"]))
    print(colored("===============", "magenta", attrs=["bold"]))
    print(result["solution"])
    
    print(colored("\nPROCESSING COMPLETE", "cyan", attrs=["bold"]))