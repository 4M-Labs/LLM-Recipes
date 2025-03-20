# Chain-of-Thought Prompting

Chain-of-Thought (CoT) prompting is a technique that encourages Large Language Models (LLMs) to break down complex problems into intermediate steps before arriving at a final answer. This approach significantly improves performance on tasks requiring multi-step reasoning.

## How It Works

Chain-of-Thought prompting works by:

1. Explicitly asking the model to "think step by step"
2. Providing examples that demonstrate the reasoning process
3. Encouraging the model to show its work before giving a final answer

This technique leverages the model's ability to follow demonstrated patterns of reasoning.

## Types of Chain-of-Thought Prompting

### Zero-Shot CoT

Simply instructing the model to reason step by step without examples:

```javascript
import { generateText } from 'ai';

const zeroShotCoT = await generateText({
  model: "llama-3-70b-instruct",
  prompt: `
    Solve this problem step by step:
    
    If John has 5 apples and gives 2 to Mary, then buys 3 more apples and gives half of all his apples to Tom, how many apples does John have left?
  `,
  temperature: 0.2, // Lower temperature for reasoning tasks
  max_tokens: 500, // Allow sufficient space for detailed reasoning
});

console.log(zeroShotCoT.text);
```

### Few-Shot CoT

Providing examples of step-by-step reasoning:

```javascript
import { generateText } from 'ai';

const fewShotCoT = await generateText({
  model: "llama-3-70b-instruct",
  prompt: `
    Solve each problem step by step.
    
    Problem: If a shirt costs $15 and is on sale for 20% off, how much does it cost?
    Step 1: Calculate the discount amount. 20% of $15 is 0.2 × $15 = $3.
    Step 2: Subtract the discount from the original price. $15 - $3 = $12.
    Answer: $12
    
    Problem: A train travels at 60 miles per hour. How far will it travel in 2.5 hours?
    Step 1: To find the distance, multiply the speed by the time.
    Step 2: Distance = 60 miles per hour × 2.5 hours = 150 miles.
    Answer: 150 miles
    
    Problem: If John has 5 apples and gives 2 to Mary, then buys 3 more apples and gives half of all his apples to Tom, how many apples does John have left?
    Step 1:
  `,
  temperature: 0.2, // Lower temperature for reasoning tasks
  max_tokens: 500, // Allow sufficient space for detailed reasoning
});

console.log(fewShotCoT.text);
```

### Self-Consistency CoT

Generating multiple reasoning paths and taking the most consistent answer:

```javascript
import { generateText } from 'ai';

// Function to extract final answer from CoT response
function extractAnswer(response) {
  // Try to match patterns like "Answer: X" or "Therefore, John has X apples left"
  const answerMatch = response.match(/Answer: (\d+)/i) || 
                      response.match(/John has (\d+) apples? left/i) ||
                      response.match(/result is (\d+)/i) ||
                      response.match(/final answer is (\d+)/i);
  
  return answerMatch ? parseInt(answerMatch[1]) : null;
}

// Generate multiple reasoning paths
async function generateMultiplePaths(problem, numPaths = 5) {
  const paths = [];
  const prompt = `Solve this problem step by step:\n\n${problem}`;
  
  for (let i = 0; i < numPaths; i++) {
    const response = await generateText({
      model: "llama-3-70b-instruct",
      temperature: 0.7, // Higher temperature for diversity
      max_tokens: 500, // Allow sufficient space for detailed reasoning
      prompt,
    });
    
    paths.push({
      reasoning: response.text,
      answer: extractAnswer(response.text)
    });
  }
  
  return paths;
}

// Find the most consistent answer
async function selfConsistencyCoT(problem) {
  console.log(`Generating multiple reasoning paths for: "${problem}"`);
  const paths = await generateMultiplePaths(problem);
  
  // Filter out paths where we couldn't extract an answer
  const validPaths = paths.filter(path => path.answer !== null);
  
  if (validPaths.length === 0) {
    return {
      mostConsistentAnswer: null,
      confidence: 0,
      message: "Could not extract any valid answers from the reasoning paths.",
      paths
    };
  }
  
  // Count occurrences of each answer
  const answerCounts = {};
  validPaths.forEach(path => {
    answerCounts[path.answer] = (answerCounts[path.answer] || 0) + 1;
  });
  
  // Find the most frequent answer
  let mostConsistentAnswer = null;
  let highestCount = 0;
  
  for (const [answer, count] of Object.entries(answerCounts)) {
    if (count > highestCount) {
      highestCount = count;
      mostConsistentAnswer = parseInt(answer);
    }
  }
  
  return {
    mostConsistentAnswer,
    confidence: highestCount / validPaths.length,
    answerDistribution: answerCounts,
    paths: validPaths.map(p => p.reasoning)
  };
}

// Example usage
const problem = "If John has 5 apples and gives 2 to Mary, then buys 3 more apples and gives half of all his apples to Tom, how many apples does John have left?";

selfConsistencyCoT(problem).then(result => {
  console.log(`Most consistent answer: ${result.mostConsistentAnswer}`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  console.log("Answer distribution:", result.answerDistribution);
  
  // Log the first reasoning path as an example
  if (result.paths && result.paths.length > 0) {
    console.log("\nExample reasoning path:");
    console.log(result.paths[0]);
  }
});
```

## Advanced Techniques

### Least-to-Most Prompting

Breaking complex problems into simpler subproblems:

```javascript
import { generateText } from 'ai';

// Function to break down a complex problem into subproblems
async function breakDownProblem(problem) {
  const breakdownPrompt = await generateText({
    model: "llama-3-70b-instruct",
    system: "You are an expert at breaking down complex problems into simpler subproblems that can be solved sequentially.",
    prompt: `Break down the following problem into 3-5 simpler subproblems that can be solved one at a time to reach the final answer. For each subproblem, provide a clear question that needs to be answered.

Problem: ${problem}

Format your response as:
Subproblem 1: [question]
Subproblem 2: [question]
...and so on.`,
    temperature: 0.3,
  });
  
  // Extract the subproblems
  const subproblems = breakdownPrompt.text
    .split('\n')
    .filter(line => line.startsWith('Subproblem'))
    .map(line => {
      const [_, subproblemText] = line.split(':');
      return subproblemText.trim();
    });
  
  return subproblems;
}

// Function to solve each subproblem sequentially
async function solveSubproblems(problem, subproblems) {
  let context = `Original problem: ${problem}\n\n`;
  let currentAnswer = '';
  
  for (let i = 0; i < subproblems.length; i++) {
    const subproblem = subproblems[i];
    
    // Add the current subproblem to the context
    context += `Subproblem ${i+1}: ${subproblem}\n`;
    
    // If we have a previous answer, include it
    if (currentAnswer) {
      context += `Previous answer: ${currentAnswer}\n`;
    }
    
    // Solve the current subproblem
    const subproblemSolution = await generateText({
      model: "llama-3-70b-instruct",
      prompt: `${context}

Solve Subproblem ${i+1} step by step, using the previous answers if available.`,
      temperature: 0.2,
    });
    
    // Extract the answer from the solution
    const answerMatch = subproblemSolution.text.match(/Answer:(.+?)(?:\n|$)/);
    currentAnswer = answerMatch ? answerMatch[1].trim() : subproblemSolution.text.trim();
    
    // Add the solution to the context
    context += `Solution to Subproblem ${i+1}: ${subproblemSolution.text}\n\n`;
  }
  
  // Generate the final answer based on all subproblem solutions
  const finalSolution = await generateText({
    model: "llama-3-70b-instruct",
    prompt: `${context}

Based on all the subproblems solved above, what is the final answer to the original problem: "${problem}"?`,
    temperature: 0.2,
  });
  
  return {
    subproblems,
    context,
    finalAnswer: finalSolution.text
  };
}

// Main function for least-to-most prompting
async function leastToMostPrompting(problem) {
  console.log(`Breaking down problem: "${problem}"`);
  const subproblems = await breakDownProblem(problem);
  
  console.log("Subproblems identified:");
  subproblems.forEach((subproblem, i) => {
    console.log(`${i+1}. ${subproblem}`);
  });
  
  console.log("\nSolving subproblems sequentially...");
  const solution = await solveSubproblems(problem, subproblems);
  
  return solution;
}

// Example usage
const complexProblem = "In a class of 30 students, 60% are girls. If 20% of the girls and 25% of the boys participate in sports, how many students don't participate in sports?";

leastToMostPrompting(complexProblem).then(result => {
  console.log("\nFinal answer:");
  console.log(result.finalAnswer);
});
```

### Tree of Thoughts (ToT)

Exploring multiple reasoning branches:

```javascript
import { generateText } from 'ai';

// Function to generate multiple initial thoughts for a problem
async function generateInitialThoughts(problem, numThoughts = 3) {
  const thoughtsPrompt = await generateText({
    model: "llama-3-70b-instruct",
    system: "You are an expert problem solver who can think of different approaches to solve problems.",
    prompt: `Generate ${numThoughts} different initial approaches to solve this problem. Each approach should start with a different perspective or method.

Problem: ${problem}

Format your response as:
Approach 1: [brief description of the first approach]
Approach 2: [brief description of the second approach]
...and so on.`,
    temperature: 0.8, // Higher temperature for diverse approaches
  });
  
  // Extract the approaches
  const approaches = thoughtsPrompt.text
    .split('\n')
    .filter(line => line.startsWith('Approach'))
    .map(line => {
      const [_, approachText] = line.split(':');
      return approachText.trim();
    });
  
  return approaches;
}

// Function to evaluate and expand a thought
async function evaluateAndExpandThought(problem, thought, depth = 0, maxDepth = 2) {
  if (depth >= maxDepth) {
    // At max depth, evaluate the final answer
    const evaluation = await generateText({
      model: "llama-3-70b-instruct",
      prompt: `Problem: ${problem}

Reasoning approach: ${thought}

Based on this reasoning approach, what is the final answer to the problem? Explain why this answer is correct.`,
      temperature: 0.3,
    });
    
    return {
      thought,
      evaluation: evaluation.text,
      children: [],
      depth
    };
  }
  
  // Expand the thought into next steps
  const expansion = await generateText({
    model: "llama-3-70b-instruct",
    prompt: `Problem: ${problem}

Current reasoning: ${thought}

Generate 2 possible next steps in this reasoning process. Each next step should build on the current reasoning but take it in a slightly different direction.

Format your response as:
Next step 1: [description]
Next step 2: [description]`,
    temperature: 0.7,
  });
  
  // Extract the next steps
  const nextSteps = expansion.text
    .split('\n')
    .filter(line => line.startsWith('Next step'))
    .map(line => {
      const [_, stepText] = line.split(':');
      return stepText.trim();
    });
  
  // Recursively evaluate each next step
  const children = [];
  for (const nextStep of nextSteps) {
    const expandedThought = `${thought}\n→ ${nextStep}`;
    const childResult = await evaluateAndExpandThought(
      problem, 
      expandedThought, 
      depth + 1, 
      maxDepth
    );
    children.push(childResult);
  }
  
  return {
    thought,
    children,
    depth
  };
}

// Function to find the best solution from the tree
async function findBestSolution(problem, thoughtTree) {
  // Collect all leaf nodes (final evaluations)
  const leaves = [];
  
  function collectLeaves(node) {
    if (node.children.length === 0 && node.evaluation) {
      leaves.push(node);
    } else {
      for (const child of node.children) {
        collectLeaves(child);
      }
    }
  }
  
  for (const root of thoughtTree) {
    collectLeaves(root);
  }
  
  if (leaves.length === 0) {
    return null;
  }
  
  // Have the model evaluate all solutions and pick the best one
  let evaluationPrompt = `Problem: ${problem}\n\nI have explored multiple reasoning paths to solve this problem. Help me determine which solution is most likely correct.\n\n`;
  
  leaves.forEach((leaf, i) => {
    evaluationPrompt += `Solution ${i+1}:\n${leaf.evaluation}\n\n`;
  });
  
  evaluationPrompt += "Which solution (by number) is most likely correct and why?";
  
  const evaluation = await generateText({
    model: "llama-3-70b-instruct",
    prompt: evaluationPrompt,
    temperature: 0.2,
  });
  
  return {
    bestSolutionIndex: evaluation.text.match(/Solution (\d+)/i)?.[1] || "1",
    evaluationReasoning: evaluation.text,
    allSolutions: leaves.map(leaf => leaf.evaluation)
  };
}

// Main function for Tree of Thoughts
async function treeOfThoughts(problem, maxDepth = 2) {
  console.log(`Generating initial thoughts for: "${problem}"`);
  const initialThoughts = await generateInitialThoughts(problem);
  
  console.log("Initial approaches:");
  initialThoughts.forEach((thought, i) => {
    console.log(`${i+1}. ${thought}`);
  });
  
  console.log("\nExpanding the reasoning tree...");
  const thoughtTree = [];
  
  for (const thought of initialThoughts) {
    const expandedThought = await evaluateAndExpandThought(
      problem, 
      thought, 
      0, 
      maxDepth
    );
    thoughtTree.push(expandedThought);
  }
  
  console.log("\nEvaluating the best solution...");
  const bestSolution = await findBestSolution(problem, thoughtTree);
  
  return {
    initialThoughts,
    thoughtTree,
    bestSolution
  };
}

// Example usage
const problem = "A store has a 25% off sale. After the discount, a shirt costs $18. What was the original price?";

treeOfThoughts(problem).then(result => {
  console.log("\nBest solution evaluation:");
  console.log(result.bestSolution.evaluationReasoning);
});
```

### Verification-Augmented CoT

Adding a verification step to check the reasoning:

```javascript
import { generateText } from 'ai';

async function verificationAugmentedCoT(problem) {
  // Step 1: Generate the initial reasoning
  const initialReasoning = await generateText({
    model: "llama-3-70b-instruct",
    prompt: `Solve this problem step by step:

${problem}`,
    temperature: 0.2,
  });
  
  // Step 2: Verify the reasoning
  const verification = await generateText({
    model: "llama-3-70b-instruct",
    system: "You are a critical reviewer who carefully checks mathematical reasoning for errors.",
    prompt: `Problem: ${problem}

Proposed solution:
${initialReasoning.text}

Carefully check this solution step by step. Are there any errors in the reasoning or calculations? If so, identify them specifically. If not, confirm that the solution is correct.`,
    temperature: 0.2,
  });
  
  // Step 3: If errors were found, generate a corrected solution
  const errorDetected = verification.text.toLowerCase().includes("error") || 
                        verification.text.toLowerCase().includes("incorrect") ||
                        verification.text.toLowerCase().includes("mistake");
  
  let finalSolution;
  
  if (errorDetected) {
    finalSolution = await generateText({
      model: "llama-3-70b-instruct",
      prompt: `Problem: ${problem}

Initial solution attempt:
${initialReasoning.text}

Review of the solution:
${verification.text}

Please provide a corrected solution that addresses the issues identified in the review. Solve the problem step by step.`,
      temperature: 0.2,
    });
  } else {
    finalSolution = { text: initialReasoning.text };
  }
  
  return {
    problem,
    initialReasoning: initialReasoning.text,
    verification: verification.text,
    errorDetected,
    finalSolution: finalSolution.text
  };
}

// Example usage
const problem = "If a recipe calls for 2/3 cup of flour and I want to make 1.5 batches, how much flour do I need?";

verificationAugmentedCoT(problem).then(result => {
  console.log("Initial reasoning:");
  console.log(result.initialReasoning);
  
  console.log("\nVerification:");
  console.log(result.verification);
  
  if (result.errorDetected) {
    console.log("\nErrors were detected. Corrected solution:");
    console.log(result.finalSolution);
  } else {
    console.log("\nNo errors detected. Solution is correct.");
  }
});
```

## Best Practices

1. **Explicitly request step-by-step reasoning** with phrases like "Let's think step by step" or "Let's solve this problem step by step"
2. **Break down complex problems** into smaller, manageable steps
3. **Use clear, logical transitions** between reasoning steps
4. **Encourage the model to verify its work** by checking intermediate results
5. **For mathematical problems**, include units and clear calculations
6. **For logical reasoning**, explicitly state assumptions and inferences
7. **Use self-consistency** for problems with high uncertainty
8. **Combine with few-shot examples** for optimal performance
9. **Provide feedback loops** where the model can critique its own reasoning
10. **Use structured formats** (numbered steps, bullet points) to organize the reasoning process
11. **Adjust temperature settings** based on the task (lower for precise reasoning, higher for exploring multiple paths)
12. **Set appropriate max_tokens** to allow for detailed reasoning
13. **Implement verification steps** to catch and correct errors
14. **Use system messages** to establish the model's role as a careful reasoner
15. **Consider multiple reasoning paths** for complex problems

## When to Use

Chain-of-Thought prompting is particularly effective for:

- Mathematical word problems
- Logical reasoning tasks
- Multi-step planning
- Complex decision-making scenarios
- Tasks requiring careful analysis of constraints
- Problems where the final answer depends on intermediate calculations
- Situations where explanation of the reasoning process is as important as the answer
- Debugging or error analysis
- Teaching or educational contexts
- Scientific or technical problem-solving

## Limitations

- Increases token usage significantly
- May introduce reasoning errors that propagate to the final answer
- Can be overly verbose for simple tasks
- Performance varies based on the complexity of the problem
- May struggle with problems requiring specialized domain knowledge
- Can sometimes follow flawed reasoning paths confidently
- Requires careful prompt engineering to avoid biases
- Not all models have equal reasoning capabilities
- May hallucinate intermediate steps that seem plausible but are incorrect
- Verification steps add computational overhead

## Research Insights

The Chain-of-Thought technique was formally introduced by Wei et al. (2022) in their paper "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." Their research demonstrated that:

- CoT prompting significantly improves performance on arithmetic, commonsense, and symbolic reasoning tasks
- The benefits of CoT increase with model size (emerging most strongly in models with >100B parameters)
- Few-shot CoT outperforms standard few-shot prompting across various reasoning tasks

Subsequent research has expanded on these findings:

- Kojima et al. (2022) showed that simply adding "Let's think step by step" can elicit reasoning in zero-shot settings
- Wang et al. (2022) introduced "Self-Consistency" to improve CoT by generating multiple reasoning paths
- Zhou et al. (2023) developed "Least-to-Most Prompting" for breaking down complex problems
- Yao et al. (2023) proposed "Tree of Thoughts" for exploring multiple reasoning branches
- Zhang et al. (2023) demonstrated that "Verification-Augmented CoT" can significantly reduce reasoning errors

The 2023 paper "Towards Understanding Chain-of-Thought Prompting" by Fu et al. analyzed why CoT works and found that:
- It helps models organize their "thoughts" in a structured way
- It reduces the working memory burden by externalizing intermediate reasoning
- It allows models to catch and correct their own errors during the reasoning process

## Implementation Strategies

### For Simple Problems

```javascript
const simpleCoT = await generateText({
  model: "llama-3-70b-instruct",
  prompt: "Let's solve this step by step: What is 17 × 24?",
  temperature: 0.2,
});
```

### For Complex Problems

```javascript
const complexCoT = await generateText({
  model: "llama-3-70b-instruct",
  system: "You are a careful problem solver who breaks down complex problems into manageable steps.",
  prompt: `Solve this problem by breaking it down into steps and showing your work:
  
  A cylindrical water tank has a radius of 4 meters and a height of 10 meters. If water is flowing into the tank at a rate of 2 cubic meters per minute, how long will it take to fill the tank to 80% of its capacity?`,
  temperature: 0.2,
  max_tokens: 800,
});
```

### For Debugging Code

```javascript
const codeDebuggingCoT = await generateText({
  model: "llama-3-70b-instruct",
  system: "You are an expert programmer who carefully analyzes code to find and fix bugs.",
  prompt: `Debug this Python code by thinking step by step about what might be causing the error:
  
  ```python
  def calculate_average(numbers):
      total = 0
      for num in numbers:
          total += num
      return total / len(numbers)
      
  result = calculate_average([])
  print(result)
  ```
  
  Error message: ZeroDivisionError: division by zero`,
  temperature: 0.2,
});
``` 