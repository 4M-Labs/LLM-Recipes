"""
Parallelization Agent Pattern Implementation

This file demonstrates the Parallelization pattern using the OpenAI API.
It showcases how to distribute tasks across multiple LLM calls simultaneously,
aggregating results to handle complex or large-scale operations efficiently.
"""

import os
import asyncio
from typing import List, Dict, Any, TypedDict, Optional
import openai
from termcolor import colored

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class SubTask(TypedDict):
    id: str
    content: str
    type: str
    dependencies: List[str]

class TaskResult(TypedDict):
    task_id: str
    result: str
    metadata: Dict[str, Any]

class AggregatedResult(TypedDict):
    results: List[TaskResult]
    summary: str
    metadata: Dict[str, Any]

async def process_subtask(subtask: SubTask) -> TaskResult:
    """
    Process a single subtask using the appropriate model
    
    Args:
        subtask: The subtask to process
        
    Returns:
        The result of the subtask processing
    """
    print(colored(f"Processing subtask {subtask['id']}: {subtask['type']}", "blue"))
    
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are processing a subtask of type: {subtask['type']}. Focus on this specific aspect only."
            },
            {
                "role": "user",
                "content": subtask['content']
            }
        ],
        temperature=0.7,
    )
    
    return {
        "task_id": subtask['id'],
        "result": response.choices[0].message.content,
        "metadata": {
            "type": subtask['type'],
            "tokens": response.usage.total_tokens if response.usage else 0
        }
    }

async def process_tasks_in_parallel(subtasks: List[SubTask]) -> List[TaskResult]:
    """
    Process multiple subtasks in parallel while respecting dependencies
    
    Args:
        subtasks: List of subtasks to process
        
    Returns:
        List of results from all subtasks
    """
    # Group tasks by dependency level
    dependency_groups: Dict[int, List[SubTask]] = {}
    processed_tasks = set()
    results = []
    
    while len(processed_tasks) < len(subtasks):
        # Find tasks whose dependencies are satisfied
        ready_tasks = [
            task for task in subtasks
            if task['id'] not in processed_tasks and
            all(dep in processed_tasks for dep in task['dependencies'])
        ]
        
        if not ready_tasks:
            raise ValueError("Circular dependency detected or invalid dependency configuration")
        
        # Process ready tasks in parallel
        print(colored(f"Processing {len(ready_tasks)} tasks in parallel", "yellow"))
        current_results = await asyncio.gather(
            *(process_subtask(task) for task in ready_tasks)
        )
        
        results.extend(current_results)
        processed_tasks.update(task['id'] for task in ready_tasks)
    
    return results

async def aggregate_results(results: List[TaskResult]) -> str:
    """
    Aggregate results from multiple subtasks into a coherent output
    
    Args:
        results: List of results from subtasks
        
    Returns:
        Aggregated and summarized result
    """
    print(colored("Aggregating results...", "green"))
    
    # Prepare results for aggregation
    results_text = "\n\n".join([
        f"Task {r['task_id']} ({r['metadata']['type']}):\n{r['result']}"
        for r in results
    ])
    
    response = await openai.chat.completions.create(
        model="gpt-4o",  # Using more capable model for aggregation
        messages=[
            {
                "role": "system",
                "content": """You are an aggregator that combines multiple task results into a coherent output.
                Synthesize the information, resolve any conflicts, and provide a clear summary."""
            },
            {
                "role": "user",
                "content": f"Aggregate and summarize these task results:\n\n{results_text}"
            }
        ],
        temperature=0.5,
    )
    
    return response.choices[0].message.content

async def parallel_process(
    task: str,
    subtask_definitions: List[Dict[str, Any]]
) -> AggregatedResult:
    """
    Main function that implements the Parallelization pattern
    
    Args:
        task: The main task to process
        subtask_definitions: List of subtask definitions with their dependencies
        
    Returns:
        Aggregated results from all subtasks
    """
    print(colored(f"Starting parallel processing for task: {task}", "yellow"))
    
    # Process subtasks in parallel
    results = await process_tasks_in_parallel(subtask_definitions)
    
    # Aggregate results
    summary = await aggregate_results(results)
    
    # Calculate metadata
    total_tokens = sum(r['metadata'].get('tokens', 0) for r in results)
    
    return {
        "results": results,
        "summary": summary,
        "metadata": {
            "total_tasks": len(results),
            "total_tokens": total_tokens
        }
    }

if __name__ == "__main__":
    # Example usage
    async def main():
        task = "Analyze a research paper and provide a comprehensive review"
        subtasks = [
            {
                "id": "methodology",
                "type": "analysis",
                "content": "Analyze the methodology section of the paper",
                "dependencies": []
            },
            {
                "id": "results",
                "type": "analysis",
                "content": "Analyze the results section of the paper",
                "dependencies": []
            },
            {
                "id": "discussion",
                "type": "analysis",
                "content": "Analyze the discussion section of the paper",
                "dependencies": ["methodology", "results"]
            },
            {
                "id": "conclusion",
                "type": "synthesis",
                "content": "Synthesize the findings and provide recommendations",
                "dependencies": ["discussion"]
            }
        ]
        
        try:
            result = await parallel_process(task, subtasks)
            
            print("\nFinal Results:")
            print("=============")
            print("\nIndividual Task Results:")
            for r in result['results']:
                print(f"\nTask {r['task_id']} ({r['metadata']['type']}):")
                print(r['result'])
            
            print("\nAggregated Summary:")
            print("==================")
            print(result['summary'])
            
            print("\nMetadata:")
            print("========")
            print(f"Total tasks: {result['metadata']['total_tasks']}")
            print(f"Total tokens: {result['metadata']['total_tokens']}")
            
        except Exception as e:
            print(colored(f"Error during parallel processing: {e}", "red"))
    
    asyncio.run(main()) 