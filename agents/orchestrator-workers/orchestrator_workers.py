"""
Orchestrator-Workers Agent Pattern Implementation

This file demonstrates the Orchestrator-Workers pattern using the OpenAI API.
It showcases how to implement a central orchestrator that directs multiple worker LLMs
to perform subtasks, synthesizing their outputs for complex, coordinated operations.
"""

import os
from typing import List, Dict, Any, TypedDict, Optional, Literal
import openai
from termcolor import colored

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class WorkerTask(TypedDict):
    id: str
    type: str
    input: str
    context: Dict[str, Any]
    requirements: List[str]

class WorkerResult(TypedDict):
    task_id: str
    output: str
    metadata: Dict[str, Any]
    status: Literal['success', 'failure', 'partial']

class OrchestratorPlan(TypedDict):
    tasks: List[WorkerTask]
    dependencies: Dict[str, List[str]]
    success_criteria: Dict[str, Any]

def create_execution_plan(goal: str) -> OrchestratorPlan:
    """
    Create a plan for achieving the goal using available workers
    
    Args:
        goal: The goal to achieve
        
    Returns:
        A structured plan with tasks and dependencies
    """
    print(colored("Creating execution plan...", "blue"))
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are a task orchestrator. Break down the given goal into subtasks
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
                }"""
            },
            {
                "role": "user",
                "content": f"Create an execution plan for this goal: {goal}"
            }
        ],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def assign_worker(task: WorkerTask) -> WorkerResult:
    """
    Assign a task to an appropriate worker LLM
    
    Args:
        task: The task to be performed
        
    Returns:
        The result from the worker
    """
    print(colored(f"Assigning task {task['id']} to {task['type']} worker", "cyan"))
    
    # Select appropriate model and configuration based on task type
    if task['type'] == 'researcher':
        model = "gpt-4o"
        system_prompt = """You are a research worker. Gather and analyze information thoroughly.
        Focus on accuracy and completeness. Provide citations where possible."""
    elif task['type'] == 'synthesizer':
        model = "gpt-4o"
        system_prompt = """You are a synthesis worker. Combine information from multiple sources
        into coherent, well-structured outputs. Focus on clarity and logical flow."""
    elif task['type'] == 'validator':
        model = "gpt-4o-mini"
        system_prompt = """You are a validation worker. Check outputs against specified requirements.
        Be thorough and explicit about any issues found."""
    else:  # specialist
        model = "gpt-4o"
        system_prompt = """You are a specialist worker. Apply domain expertise to solve specific problems.
        Focus on accuracy and practical applicability."""
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""Task: {task['input']}
                
                Requirements:
                {', '.join(task['requirements'])}
                
                Context:
                {task['context']}"""
            }
        ],
        temperature=0.5,
    )
    
    return {
        "task_id": task['id'],
        "output": response.choices[0].message.content,
        "metadata": {
            "type": task['type'],
            "tokens": response.usage.total_tokens if response.usage else 0
        },
        "status": "success"
    }

def synthesize_results(results: List[WorkerResult], goal: str) -> str:
    """
    Synthesize results from multiple workers into a final output
    
    Args:
        results: List of results from workers
        goal: The original goal
        
    Returns:
        Synthesized final output
    """
    print(colored("Synthesizing final results...", "green"))
    
    results_text = "\n\n".join([
        f"Task {r['task_id']} ({r['metadata']['type']}):\n{r['output']}"
        for r in results
    ])
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a final synthesizer. Combine all worker outputs into a
                coherent final result that achieves the original goal. Ensure all key points
                are included and properly connected."""
            },
            {
                "role": "user",
                "content": f"""Original Goal: {goal}

                Worker Results:
                {results_text}
                
                Please synthesize these results into a final output that achieves the goal."""
            }
        ],
        temperature=0.5,
    )
    
    return response.choices[0].message.content

def orchestrate_task(goal: str) -> Dict[str, Any]:
    """
    Main function that implements the Orchestrator-Workers pattern
    
    Args:
        goal: The goal to achieve
        
    Returns:
        Final results and execution metadata
    """
    print(colored(f"Starting orchestration for goal: {goal}", "yellow"))
    
    # Create execution plan
    plan = create_execution_plan(goal)
    
    # Track completed tasks and their results
    completed_tasks = {}
    all_results = []
    
    # Execute tasks in dependency order
    while len(completed_tasks) < len(plan['tasks']):
        # Find tasks whose dependencies are met
        available_tasks = [
            task for task in plan['tasks']
            if task['id'] not in completed_tasks and
            all(dep in completed_tasks for dep in plan['dependencies'].get(task['id'], []))
        ]
        
        if not available_tasks:
            raise ValueError("Circular dependency detected or invalid task configuration")
        
        # Process available tasks
        for task in available_tasks:
            # Add results from dependencies to task context
            task['context']['dependency_results'] = {
                dep: completed_tasks[dep]['output']
                for dep in plan['dependencies'].get(task['id'], [])
            }
            
            # Assign task to worker
            result = assign_worker(task)
            completed_tasks[task['id']] = result
            all_results.append(result)
    
    # Synthesize final results
    final_output = synthesize_results(all_results, goal)
    
    # Calculate metadata
    total_tokens = sum(r['metadata'].get('tokens', 0) for r in all_results)
    
    return {
        "goal": goal,
        "final_output": final_output,
        "worker_results": all_results,
        "execution_plan": plan,
        "metadata": {
            "total_tasks": len(all_results),
            "total_tokens": total_tokens,
            "success": all(r['status'] == 'success' for r in all_results)
        }
    }

if __name__ == "__main__":
    # Example usage
    goal = """Create a comprehensive market analysis report for a new smartphone app,
    including target audience analysis, competitor research, and pricing strategy."""
    
    try:
        result = orchestrate_task(goal)
        
        print("\nFinal Results:")
        print("=============")
        print(result['final_output'])
        
        print("\nExecution Statistics:")
        print("====================")
        print(f"Total tasks: {result['metadata']['total_tasks']}")
        print(f"Total tokens: {result['metadata']['total_tokens']}")
        print(f"Overall success: {result['metadata']['success']}")
        
    except Exception as e:
        print(colored(f"Error during orchestration: {e}", "red")) 