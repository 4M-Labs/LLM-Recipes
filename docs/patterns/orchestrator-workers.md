# Orchestrator-Workers Pattern

The Orchestrator-Workers pattern implements a coordinated system where a central orchestrator LLM directs multiple specialized worker LLMs to perform complex tasks. This pattern excels at managing complex workflows requiring different types of expertise.

## Key Concepts

1. **Central Coordination**: Orchestrator manages overall workflow
2. **Specialized Workers**: Different workers for different task types
3. **Task Planning**: Dynamic task breakdown and assignment
4. **Result Synthesis**: Combining worker outputs coherently

## When to Use

Use the Orchestrator-Workers pattern when:
- Tasks require multiple types of expertise
- Complex coordination is needed
- Quality control is critical
- Tasks have interdependencies

## Implementation

### Basic Structure

```python
async def orchestrate_task(goal: str) -> Dict[str, Any]:
    # Step 1: Create execution plan
    plan = await create_execution_plan(goal)
    
    # Step 2: Assign and execute tasks
    results = await execute_tasks(plan)
    
    # Step 3: Synthesize results
    final_output = await synthesize_results(results)
    
    return {
        "output": final_output,
        "plan": plan,
        "results": results
    }
```

### Key Components

1. **Orchestrator**:
   - Task planning
   - Worker assignment
   - Progress monitoring
   - Quality control

2. **Workers**:
   - Specialized processing
   - Result generation
   - Error handling
   - Status reporting

3. **Coordinator**:
   - Resource management
   - Task scheduling
   - State tracking
   - Error recovery

## Worker Types

1. **Researcher**:
   - Information gathering
   - Data analysis
   - Source verification
   - Fact checking

2. **Synthesizer**:
   - Information combination
   - Pattern recognition
   - Summary generation
   - Coherence checking

3. **Validator**:
   - Quality control
   - Requirement checking
   - Error detection
   - Consistency verification

4. **Specialist**:
   - Domain-specific tasks
   - Technical analysis
   - Expert review
   - Specialized processing

## Best Practices

1. **Task Management**:
   - Clear task definitions
   - Explicit dependencies
   - Progress tracking
   - Quality metrics

2. **Worker Management**:
   - Appropriate assignment
   - Load balancing
   - Performance monitoring
   - Error handling

3. **Result Management**:
   - Quality validation
   - Coherent synthesis
   - Error recovery
   - Version control

## Example Use Cases

1. **Market Research**:
   ```
   Goal → Research Plan → Data Collection → Analysis → Synthesis → Validation → Report
   ```

2. **Content Creation**:
   ```
   Brief → Planning → Research → Writing → Editing → Review → Publication
   ```

3. **Technical Analysis**:
   ```
   Problem → Research → Analysis → Solution Design → Implementation → Testing → Documentation
   ```

## Common Pitfalls

1. **Coordination Issues**:
   - Unclear responsibilities
   - Communication gaps
   - Dependency conflicts
   - Resource contention

2. **Quality Problems**:
   - Inconsistent outputs
   - Missing validations
   - Poor synthesis
   - Lost context

3. **Performance Issues**:
   - Bottlenecks
   - Resource waste
   - Poor scheduling
   - Excessive overhead

## Implementation Example

```python
from typing import Dict, Any, List, Literal
import openai
from dataclasses import dataclass
from enum import Enum

class WorkerType(Enum):
    RESEARCHER = "researcher"
    SYNTHESIZER = "synthesizer"
    VALIDATOR = "validator"
    SPECIALIST = "specialist"

@dataclass
class Task:
    id: str
    type: WorkerType
    description: str
    requirements: List[str]
    context: Dict[str, Any]

@dataclass
class WorkerResult:
    task_id: str
    output: str
    metadata: Dict[str, Any]
    status: Literal["success", "failure"]

class Worker:
    def __init__(self, worker_type: WorkerType):
        self.type = worker_type
    
    async def process(self, task: Task) -> WorkerResult:
        """Process a task according to worker specialty."""
        # Select appropriate model and prompt based on worker type
        if self.type == WorkerType.RESEARCHER:
            model = "gpt-4o"
            system_prompt = """You are a research specialist. Gather and analyze
            information thoroughly. Focus on accuracy and completeness."""
        elif self.type == WorkerType.SYNTHESIZER:
            model = "gpt-4o"
            system_prompt = """You are a synthesis specialist. Combine information
            from multiple sources into coherent outputs."""
        elif self.type == WorkerType.VALIDATOR:
            model = "gpt-4o-mini"
            system_prompt = """You are a validation specialist. Check outputs
            against requirements and ensure quality."""
        else:  # SPECIALIST
            model = "gpt-4o"
            system_prompt = """You are a domain specialist. Apply expert knowledge
            to solve specific problems."""
        
        try:
            response = await openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"""Task: {task.description}
                        
                        Requirements:
                        {', '.join(task.requirements)}
                        
                        Context:
                        {task.context}"""
                    }
                ],
                temperature=0.7,
            )
            
            return WorkerResult(
                task_id=task.id,
                output=response.choices[0].message.content,
                metadata={
                    "type": self.type.value,
                    "tokens": response.usage.total_tokens
                },
                status="success"
            )
            
        except Exception as e:
            return WorkerResult(
                task_id=task.id,
                output="",
                metadata={"error": str(e)},
                status="failure"
            )

class Orchestrator:
    def __init__(self):
        self.workers = {
            worker_type: Worker(worker_type)
            for worker_type in WorkerType
        }
    
    async def create_plan(self, goal: str) -> Dict[str, Any]:
        """Create an execution plan for the goal."""
        response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Create a detailed plan to achieve the goal.
                    Break it down into tasks for different worker types:
                    - researcher: information gathering
                    - synthesizer: information combination
                    - validator: quality control
                    - specialist: domain-specific tasks
                    
                    Return a JSON object with tasks and their dependencies."""
                },
                {
                    "role": "user",
                    "content": f"Create a plan for: {goal}"
                }
            ],
            temperature=0.7,
        )
        
        return response.choices[0].message.content
    
    async def execute_tasks(
        self,
        tasks: List[Task],
        dependencies: Dict[str, List[str]]
    ) -> List[WorkerResult]:
        """Execute tasks according to dependencies."""
        completed_tasks = set()
        all_results = []
        
        while len(completed_tasks) < len(tasks):
            # Find tasks whose dependencies are met
            available_tasks = [
                task for task in tasks
                if task.id not in completed_tasks and
                all(dep in completed_tasks for dep in dependencies.get(task.id, []))
            ]
            
            if not available_tasks:
                raise ValueError("Circular dependency detected")
            
            # Process available tasks
            for task in available_tasks:
                worker = self.workers[task.type]
                result = await worker.process(task)
                completed_tasks.add(task.id)
                all_results.append(result)
        
        return all_results
    
    async def synthesize_results(
        self,
        results: List[WorkerResult],
        goal: str
    ) -> str:
        """Synthesize results into final output."""
        results_text = "\n\n".join([
            f"Task {r.task_id} ({r.metadata.get('type', 'unknown')}):\n{r.output}"
            for r in results
        ])
        
        response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Synthesize all worker outputs into a coherent
                    final result that achieves the original goal."""
                },
                {
                    "role": "user",
                    "content": f"""Goal: {goal}

                    Worker Results:
                    {results_text}"""
                }
            ],
            temperature=0.5,
        )
        
        return response.choices[0].message.content
    
    async def orchestrate(self, goal: str) -> Dict[str, Any]:
        """Main orchestration function."""
        try:
            # Create execution plan
            plan = await self.create_plan(goal)
            
            # Execute tasks
            results = await self.execute_tasks(
                plan["tasks"],
                plan["dependencies"]
            )
            
            # Synthesize results
            final_output = await self.synthesize_results(results, goal)
            
            return {
                "goal": goal,
                "output": final_output,
                "results": results,
                "plan": plan,
                "metadata": {
                    "total_tasks": len(results),
                    "successful_tasks": len([
                        r for r in results
                        if r.status == "success"
                    ])
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "goal": goal,
                "metadata": {
                    "error_type": type(e).__name__
                }
            }
```

## Monitoring and Debugging

1. **System Monitoring**:
   - Worker performance
   - Task completion rates
   - Resource usage
   - Error patterns

2. **Quality Metrics**:
   - Output accuracy
   - Consistency scores
   - Validation results
   - User satisfaction

3. **Process Analytics**:
   - Task distribution
   - Worker utilization
   - Bottleneck identification
   - Error analysis

## References

- [Implementation Examples](../../agents/orchestrator-workers/)
- [Setup Guide](../setup/orchestrator-workers-setup.md)
- [Troubleshooting Guide](../troubleshooting.md) 