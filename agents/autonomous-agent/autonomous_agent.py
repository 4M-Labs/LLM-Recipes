"""
Autonomous Agent Pattern Implementation

This file demonstrates the Autonomous Agent pattern using the OpenAI API.
It showcases how to create an agent that can operate independently to achieve
goals through a continuous cycle of perception, reasoning, and action.
"""

import os
import json
import time
from typing import Dict, List, TypedDict, Union, Any, Optional, Literal, Protocol
import openai
from termcolor import colored

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Type definitions
class Observation(TypedDict):
    timestamp: float
    content: str
    type: Literal["environment", "action_result", "user_input"]

class Action(TypedDict):
    type: str
    parameters: Dict[str, Any]
    reasoning: str

class ActionResult(TypedDict):
    status: Literal["success", "failure", "partial"]
    output: str
    goal_achieved: bool
    feedback: Optional[str]

class MemoryState(TypedDict):
    goal: str
    observations: List[Observation]
    actions: List[Dict[str, Union[Action, ActionResult]]]
    iterations: int
    current_state: str

# Environment protocol
class Environment(Protocol):
    async def observe(self) -> Observation:
        ...
    
    async def execute_action(self, action: Action) -> ActionResult:
        ...

# Memory class to store agent state
class Memory:
    def __init__(self):
        self.state: MemoryState = {
            "goal": "",
            "observations": [],
            "actions": [],
            "iterations": 0,
            "current_state": "initialized"
        }
    
    def set_goal(self, goal: str) -> None:
        self.state["goal"] = goal
        self.state["current_state"] = "goal_set"
    
    def add_observation(self, observation: Observation) -> None:
        self.state["observations"].append(observation)
        self.state["current_state"] = "observed"
    
    def add_action_result(self, action: Action, result: ActionResult) -> None:
        self.state["actions"].append({"action": action, "result": result})
        self.state["iterations"] += 1
        self.state["current_state"] = "action_taken"
    
    def get_recent_observations(self, count: int = 5) -> List[Observation]:
        return self.state["observations"][-count:]
    
    def get_recent_actions(self, count: int = 3) -> List[Dict[str, Union[Action, ActionResult]]]:
        return self.state["actions"][-count:]
    
    def get_full_state(self) -> MemoryState:
        return self.state.copy()
    
    def get_summary_state(self) -> str:
        recent_observations = "\n".join([o["content"] for o in self.get_recent_observations()])
        recent_actions = "\n".join([
            f"Action: {a['action']['type']} - Result: {a['result']['status']}"
            for a in self.get_recent_actions()
        ])
        
        return f"""Goal: {self.state["goal"]}
Iterations: {self.state["iterations"]}
Current State: {self.state["current_state"]}
Recent Observations:
{recent_observations}
Recent Actions:
{recent_actions}"""
    
    @property
    def iterations(self) -> int:
        return self.state["iterations"]
    
    def get_final_result(self) -> ActionResult:
        if not self.state["actions"]:
            return {
                "status": "failure",
                "output": "No actions were taken",
                "goal_achieved": False,
                "feedback": None
            }
        
        last_action = self.state["actions"][-1]
        return last_action["result"]

# Web search environment implementation
class WebSearchEnvironment:
    def __init__(self, initial_context: str):
        self.context = initial_context
        self.search_results: List[str] = []
    
    async def observe(self) -> Observation:
        # In a real implementation, this might fetch updated information
        # For this example, we'll just return the current context
        return {
            "timestamp": time.time(),
            "content": f"Current context: {self.context}\nSearch results: {' '.join(self.search_results)}",
            "type": "environment"
        }
    
    async def execute_action(self, action: Action) -> ActionResult:
        print(colored(f"Executing action: {action['type']}", "blue"))
        
        if action["type"] == "search":
            return await self.perform_search(action["parameters"]["query"])
        elif action["type"] == "analyze":
            return await self.analyze_information(action["parameters"]["topic"])
        elif action["type"] == "summarize":
            return await self.summarize_findings()
        elif action["type"] == "conclude":
            return self.conclude_task(action["parameters"]["conclusion"])
        else:
            return {
                "status": "failure",
                "output": f"Unknown action type: {action['type']}",
                "goal_achieved": False,
                "feedback": None
            }
    
    async def perform_search(self, query: str) -> ActionResult:
        print(colored(f"Performing search for: {query}", "cyan"))
        
        try:
            # Simulate a web search using the LLM
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search engine. Provide realistic search results for the given query. Return 3-5 relevant results with titles and brief descriptions."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=0.7,
            )
            
            search_result = response.choices[0].message.content
            self.search_results.append(f"Query: {query}\nResults: {search_result}")
            
            return {
                "status": "success",
                "output": search_result,
                "goal_achieved": False,
                "feedback": "Search completed successfully"
            }
        except Exception as e:
            print(colored(f"Search error: {str(e)}", "red"))
            return {
                "status": "failure",
                "output": f"Error performing search: {str(e)}",
                "goal_achieved": False,
                "feedback": None
            }
    
    async def analyze_information(self, topic: str) -> ActionResult:
        print(colored(f"Analyzing information about: {topic}", "cyan"))
        
        if not self.search_results:
            return {
                "status": "failure",
                "output": "No search results to analyze. Perform a search first.",
                "goal_achieved": False,
                "feedback": None
            }
        
        try:
            # Use the LLM to analyze the search results
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an analytical assistant. Analyze the provided search results and extract key insights about the topic."
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\nSearch Results:\n{' '.join(self.search_results)}\n\nPlease analyze these results and provide key insights."
                    }
                ],
                temperature=0.5,
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "status": "success",
                "output": analysis,
                "goal_achieved": False,
                "feedback": "Analysis completed successfully"
            }
        except Exception as e:
            print(colored(f"Analysis error: {str(e)}", "red"))
            return {
                "status": "failure",
                "output": f"Error analyzing information: {str(e)}",
                "goal_achieved": False,
                "feedback": None
            }
    
    async def summarize_findings(self) -> ActionResult:
        print(colored("Summarizing findings", "cyan"))
        
        if not self.search_results:
            return {
                "status": "failure",
                "output": "No search results to summarize. Perform a search first.",
                "goal_achieved": False,
                "feedback": None
            }
        
        try:
            # Use the LLM to summarize the search results
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a summarization assistant. Create a concise summary of the provided search results."
                    },
                    {
                        "role": "user",
                        "content": f"Search Results:\n{' '.join(self.search_results)}\n\nPlease provide a concise summary of these results."
                    }
                ],
                temperature=0.3,
            )
            
            summary = response.choices[0].message.content
            
            return {
                "status": "success",
                "output": summary,
                "goal_achieved": False,
                "feedback": "Summarization completed successfully"
            }
        except Exception as e:
            print(colored(f"Summarization error: {str(e)}", "red"))
            return {
                "status": "failure",
                "output": f"Error summarizing findings: {str(e)}",
                "goal_achieved": False,
                "feedback": None
            }
    
    def conclude_task(self, conclusion: str) -> ActionResult:
        print(colored("Concluding task", "green"))
        
        # Update the context with the conclusion
        self.context = conclusion
        
        return {
            "status": "success",
            "output": f"Task concluded with: {conclusion}",
            "goal_achieved": True,
            "feedback": "Task completed successfully"
        }

# Autonomous Agent class
class AutonomousAgent:
    def __init__(self, environment: WebSearchEnvironment, max_iterations: int = 10):
        self.memory = Memory()
        self.environment = environment
        self.max_iterations = max_iterations
    
    async def run(self, goal: str) -> ActionResult:
        print(colored(f"Starting autonomous agent with goal: {goal}", "yellow"))
        
        # Initialize the agent with a goal
        self.memory.set_goal(goal)
        is_complete = False
        
        # Main agent loop
        while not is_complete and self.memory.iterations < self.max_iterations:
            print(colored(f"Iteration {self.memory.iterations + 1}/{self.max_iterations}", "yellow"))
            
            # 1. Observe the environment
            observation = await self.environment.observe()
            print(colored("Observation received", "green"))
            
            # 2. Update memory with new observation
            self.memory.add_observation(observation)
            
            # 3. Reason about the current state and determine next action
            action = await self.reason()
            print(colored(f"Determined action: {action['type']}", "green"))
            
            # 4. Execute the action in the environment
            result = await self.environment.execute_action(action)
            print(colored(f"Action result: {result['status']}", "green"))
            
            # 5. Process feedback and determine if goal is complete
            self.memory.add_action_result(action, result)
            is_complete = self.evaluate_completion(result)
            
            if is_complete:
                print(colored("Goal achieved!", "green", attrs=["bold"]))
        
        if self.memory.iterations >= self.max_iterations and not is_complete:
            print(colored("Maximum iterations reached without completing the goal", "yellow"))
        
        return self.memory.get_final_result()
    
    async def reason(self) -> Action:
        print(colored("Reasoning about next action...", "magenta"))
        
        memory_state = self.memory.get_summary_state()
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """You are an autonomous agent that determines the next best action to take based on the current state and goal.
                        
                        Available actions:
                        1. search - Perform a web search (parameters: query)
                        2. analyze - Analyze information on a specific topic (parameters: topic)
                        3. summarize - Summarize current findings (no parameters)
                        4. conclude - Conclude the task with a final answer (parameters: conclusion)
                        
                        Respond with a JSON object containing:
                        - type: The action type (one of the available actions)
                        - parameters: An object with the required parameters for the action
                        - reasoning: Your reasoning for choosing this action"""
                    },
                    {
                        "role": "user",
                        "content": f"Current State:\n{memory_state}\n\nDetermine the next best action to take to achieve the goal."
                    }
                ],
                temperature=0.2,
            )
            
            action_plan = json.loads(response.choices[0].message.content)
            
            return {
                "type": action_plan.get("type", "search"),
                "parameters": action_plan.get("parameters", {}),
                "reasoning": action_plan.get("reasoning", "No reasoning provided")
            }
        except Exception as e:
            print(colored(f"Reasoning error: {str(e)}", "red"))
            
            # Fallback action if reasoning fails
            return {
                "type": "search",
                "parameters": {"query": self.memory.get_full_state()["goal"]},
                "reasoning": "Fallback action due to reasoning error"
            }
    
    def evaluate_completion(self, result: ActionResult) -> bool:
        return result["status"] == "success" and result["goal_achieved"]

async def main():
    # Example usage
    initial_context = "Starting research on a new topic"
    environment = WebSearchEnvironment(initial_context)
    agent = AutonomousAgent(environment, 5)
    
    goal = "Research the latest advancements in renewable energy and provide a summary of the most promising technologies"
    
    try:
        print(colored("=== STARTING AUTONOMOUS AGENT ===", "cyan", attrs=["bold"]))
        result = await agent.run(goal)
        
        print(colored("\n=== FINAL RESULT ===", "magenta", attrs=["bold"]))
        print(colored(f"Status: {result['status']}", "white"))
        print(colored(f"Goal Achieved: {result['goal_achieved']}", "white"))
        print(colored(f"Output: {result['output']}", "white"))
        
        if result.get("feedback"):
            print(colored(f"Feedback: {result['feedback']}", "white"))
        
        print(colored("\nPROCESSING COMPLETE", "cyan", attrs=["bold"]))
    except Exception as e:
        print(colored(f"Error running autonomous agent: {str(e)}", "red"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())