# Tree of Thoughts (ToT) Prompting

Tree of Thoughts (ToT) is an advanced prompting technique that extends Chain-of-Thought reasoning by exploring multiple reasoning paths simultaneously. Instead of following a single line of reasoning, ToT creates a tree-like structure of thoughts, evaluates different branches, and selects the most promising paths to continue exploration.

## How It Works

Tree of Thoughts works through the following process:

1. **Generate multiple initial thoughts** for approaching a problem
2. **Evaluate each thought** based on its potential to lead to a correct solution
3. **Expand promising thoughts** into further reasoning steps
4. **Prune less promising branches** to focus computational resources
5. **Continue exploration** until reaching a satisfactory solution

This approach enables more thorough exploration of the solution space and helps avoid getting stuck in suboptimal reasoning paths.

## Example Implementation

```javascript
import { generateText } from 'ai';

async function treeOfThoughts(problem, maxDepth = 3, branchingFactor = 3) {
  // Initialize the tree with the problem as the root
  const root = {
    content: problem,
    children: [],
    depth: 0,
    score: 0
  };
  
  // Queue for breadth-first exploration
  const queue = [root];
  let bestSolution = null;
  let bestScore = -Infinity;
  
  while (queue.length > 0) {
    const node = queue.shift();
    
    // If we've reached max depth, evaluate this as a potential solution
    if (node.depth === maxDepth) {
      const evaluation = await evaluateSolution(node.content, problem);
      if (evaluation.score > bestScore) {
        bestScore = evaluation.score;
        bestSolution = node;
      }
      continue;
    }
    
    // Generate multiple thoughts/next steps from this node
    const thoughts = await generateThoughts(node.content, problem, branchingFactor);
    
    // Evaluate each thought
    const evaluatedThoughts = await Promise.all(
      thoughts.map(async (thought) => {
        const evaluation = await evaluateThought(thought, problem, node.depth + 1);
        return {
          content: thought,
          score: evaluation.score,
          reasoning: evaluation.reasoning,
          depth: node.depth + 1,
          children: []
        };
      })
    );
    
    // Sort thoughts by score and keep only the top k
    const topThoughts = evaluatedThoughts
      .sort((a, b) => b.score - a.score)
      .slice(0, branchingFactor);
    
    // Add the top thoughts as children to the current node
    node.children = topThoughts;
    
    // Add promising thoughts to the queue for further exploration
    for (const thought of topThoughts) {
      if (thought.score > 0.5) { // Only explore promising thoughts
        queue.push(thought);
      }
    }
  }
  
  return reconstructSolution(bestSolution);
}

async function generateThoughts(currentState, originalProblem, count) {
  const response = await generateText({
    model: yourModel,
    system: `You are an expert problem solver exploring multiple approaches to solve a complex problem.`,
    prompt: `Original Problem: ${originalProblem}
    
    Current reasoning state: ${currentState}
    
    Generate ${count} different next steps or thoughts that could help solve this problem. 
    Each thought should be distinct and explore a different approach or aspect of the problem.
    
    Format your response as a numbered list with one thought per line:
    1. [First thought]
    2. [Second thought]
    3. [Third thought]`,
  });
  
  // Parse the numbered list into an array of thoughts
  const thoughts = response
    .split('\n')
    .filter(line => /^\d+\./.test(line))
    .map(line => line.replace(/^\d+\.\s*/, '').trim());
  
  return thoughts.slice(0, count);
}

async function evaluateThought(thought, originalProblem, depth) {
  const response = await generateText({
    model: yourModel,
    system: `You are an expert evaluator assessing the quality and promise of different problem-solving approaches.`,
    prompt: `Original Problem: ${originalProblem}
    
    Thought to evaluate: ${thought}
    
    Evaluate this thought on a scale from 0 to 1, where:
    - 0 means the thought is completely irrelevant or incorrect
    - 0.5 means the thought has some merit but may not lead directly to a solution
    - 1 means the thought is highly promising and likely to lead to a correct solution
    
    Format your response as:
    Score: [number between 0 and 1]
    Reasoning: [brief explanation of your evaluation]`,
  });
  
  // Parse the score and reasoning
  const scoreMatch = response.match(/Score:\s*(0(\.\d+)?|1(\.0+)?)/);
  const reasoningMatch = response.match(/Reasoning:\s*(.*)/);
  
  return {
    score: scoreMatch ? parseFloat(scoreMatch[1]) : 0.5,
    reasoning: reasoningMatch ? reasoningMatch[1] : "No reasoning provided"
  };
}

async function evaluateSolution(solution, originalProblem) {
  const response = await generateText({
    model: yourModel,
    system: `You are an expert evaluator assessing the quality of solutions to complex problems.`,
    prompt: `Original Problem: ${originalProblem}
    
    Proposed solution: ${solution}
    
    Evaluate this solution on a scale from 0 to 1, where:
    - 0 means the solution is completely incorrect
    - 0.5 means the solution partially addresses the problem
    - 1 means the solution is completely correct and optimal
    
    Format your response as:
    Score: [number between 0 and 1]
    Reasoning: [brief explanation of your evaluation]`,
  });
  
  // Parse the score and reasoning
  const scoreMatch = response.match(/Score:\s*(0(\.\d+)?|1(\.0+)?)/);
  const reasoningMatch = response.match(/Reasoning:\s*(.*)/);
  
  return {
    score: scoreMatch ? parseFloat(scoreMatch[1]) : 0.5,
    reasoning: reasoningMatch ? reasoningMatch[1] : "No reasoning provided"
  };
}

function reconstructSolution(node) {
  if (!node) return "No solution found";
  
  // Reconstruct the path from root to this node
  let current = node;
  const path = [current.content];
  
  while (current.parent) {
    current = current.parent;
    path.unshift(current.content);
  }
  
  return path.join("\n\nNext step:\n\n");
}
```

## Using the Vercel AI SDK

```javascript
import { streamText } from 'ai';

async function treeOfThoughtsWithVercel(problem) {
  // Step 1: Generate multiple initial approaches
  const initialThoughtsResponse = await streamText({
    model: yourModel,
    system: `You are an expert problem solver using Tree of Thoughts methodology.`,
    prompt: `Problem: ${problem}
    
    Generate 3 different initial approaches to solve this problem. Each approach should represent a distinct way of thinking about the problem.
    
    Format your response as:
    Approach 1: [description]
    Approach 2: [description]
    Approach 3: [description]`,
  });
  
  let initialThoughtsText = '';
  for await (const chunk of initialThoughtsResponse) {
    initialThoughtsText += chunk;
  }
  
  // Parse the initial approaches
  const approaches = initialThoughtsText
    .split('\n')
    .filter(line => line.startsWith('Approach'))
    .map(line => {
      const [label, ...description] = line.split(': ');
      return {
        label,
        description: description.join(': '),
        expanded: false,
        score: 0
      };
    });
  
  // Step 2: Evaluate each approach
  for (let i = 0; i < approaches.length; i++) {
    const evaluationResponse = await streamText({
      model: yourModel,
      system: `You are an expert evaluator of problem-solving approaches.`,
      prompt: `Problem: ${problem}
      
      Approach: ${approaches[i].description}
      
      Evaluate this approach on a scale from 0 to 10, where 10 is extremely promising.
      
      Format your response as:
      Score: [number]
      Reasoning: [explanation]`,
    });
    
    let evaluationText = '';
    for await (const chunk of evaluationResponse) {
      evaluationText += chunk;
    }
    
    // Parse the score
    const scoreMatch = evaluationText.match(/Score:\s*(\d+)/);
    approaches[i].score = scoreMatch ? parseInt(scoreMatch[1]) : 5;
  }
  
  // Step 3: Expand the most promising approach
  const bestApproach = approaches.sort((a, b) => b.score - a.score)[0];
  
  const expansionResponse = await streamText({
    model: yourModel,
    system: `You are an expert problem solver developing a solution using Tree of Thoughts methodology.`,
    prompt: `Problem: ${problem}
    
    Selected approach: ${bestApproach.description}
    
    Expand this approach into 3 more detailed sub-steps or considerations. For each sub-step, provide:
    1. A detailed description
    2. Potential challenges
    3. How this sub-step contributes to solving the overall problem
    
    Format your response as:
    Sub-step 1: [title]
    Description: [detailed description]
    Challenges: [potential challenges]
    Contribution: [how this helps]
    
    Sub-step 2: [title]
    ...
    
    Sub-step 3: [title]
    ...`,
  });
  
  let expansionText = '';
  for await (const chunk of expansionResponse) {
    expansionText += chunk;
  }
  
  // Step 4: Generate final solution based on the expanded approach
  const solutionResponse = await streamText({
    model: yourModel,
    system: `You are an expert problem solver synthesizing a final solution.`,
    prompt: `Problem: ${problem}
    
    Selected approach: ${bestApproach.description}
    
    Expanded thinking:
    ${expansionText}
    
    Based on this tree of thoughts exploration, provide a comprehensive solution to the original problem.
    
    Format your response as:
    Final Solution: [detailed solution]
    Reasoning: [explanation of how the solution was derived through the tree of thoughts process]`,
  });
  
  let solutionText = '';
  for await (const chunk of solutionResponse) {
    solutionText += chunk;
  }
  
  return {
    problem,
    approaches,
    bestApproach,
    expansion: expansionText,
    solution: solutionText
  };
}
```

## Advanced ToT Patterns

### Breadth-First vs. Depth-First Exploration

```javascript
async function treeOfThoughtsWithStrategy(problem, strategy = 'breadth-first') {
  // Initial thoughts generation
  const initialThoughts = await generateInitialThoughts(problem);
  
  if (strategy === 'breadth-first') {
    return breadthFirstExploration(problem, initialThoughts);
  } else if (strategy === 'depth-first') {
    return depthFirstExploration(problem, initialThoughts);
  } else if (strategy === 'best-first') {
    return bestFirstExploration(problem, initialThoughts);
  } else {
    throw new Error(`Unknown strategy: ${strategy}`);
  }
}

async function breadthFirstExploration(problem, initialThoughts) {
  // Explore all thoughts at each level before moving deeper
  let currentLevel = initialThoughts;
  const maxDepth = 3;
  
  for (let depth = 0; depth < maxDepth; depth++) {
    const nextLevel = [];
    
    for (const thought of currentLevel) {
      // Generate child thoughts
      const children = await expandThought(thought, problem);
      nextLevel.push(...children);
    }
    
    // Evaluate and prune
    const evaluatedThoughts = await evaluateThoughts(nextLevel, problem);
    currentLevel = pruneThoughts(evaluatedThoughts);
  }
  
  // Select best final thought
  const finalEvaluations = await evaluateThoughts(currentLevel, problem, true);
  return finalEvaluations.sort((a, b) => b.score - a.score)[0];
}

async function depthFirstExploration(problem, initialThoughts) {
  // Evaluate initial thoughts
  const evaluatedThoughts = await evaluateThoughts(initialThoughts, problem);
  const bestInitialThought = evaluatedThoughts.sort((a, b) => b.score - a.score)[0];
  
  // Recursively explore the best thought first, to a maximum depth
  return exploreThoughtDepthFirst(bestInitialThought, problem, 1, 3);
}

async function exploreThoughtDepthFirst(thought, problem, currentDepth, maxDepth) {
  if (currentDepth >= maxDepth) {
    return thought;
  }
  
  // Expand this thought
  const children = await expandThought(thought, problem);
  
  // Evaluate children
  const evaluatedChildren = await evaluateThoughts(children, problem);
  const bestChild = evaluatedChildren.sort((a, b) => b.score - a.score)[0];
  
  // Recursively explore the best child
  return exploreThoughtDepthFirst(bestChild, problem, currentDepth + 1, maxDepth);
}
```

### Monte Carlo Tree Search for ToT

```javascript
async function monteCarloTreeOfThoughts(problem, iterations = 10) {
  // Root node represents the initial problem
  const root = {
    content: problem,
    children: [],
    visits: 0,
    value: 0
  };
  
  for (let i = 0; i < iterations; i++) {
    // Selection: Select a promising node to expand
    let node = select(root);
    
    // Expansion: Add a new child node
    if (node.visits > 0) {
      node = expand(node, problem);
    }
    
    // Simulation: Simulate a random playout from this node
    const reward = await simulate(node, problem);
    
    // Backpropagation: Update statistics for all nodes in the path
    backpropagate(node, reward);
  }
  
  // Return the best child of the root
  return getBestChild(root);
}

function select(node) {
  // If node has unvisited children, return the node itself
  if (node.children.length === 0 || node.children.some(child => child.visits === 0)) {
    return node;
  }
  
  // Otherwise, select the best child according to UCT formula
  let bestChild = null;
  let bestUCT = -Infinity;
  
  for (const child of node.children) {
    // UCT formula: exploitation + exploration
    const exploitation = child.value / child.visits;
    const exploration = Math.sqrt(2 * Math.log(node.visits) / child.visits);
    const uct = exploitation + exploration;
    
    if (uct > bestUCT) {
      bestUCT = uct;
      bestChild = child;
    }
  }
  
  // Recursively select from the best child
  return select(bestChild);
}

async function expand(node, problem) {
  // Generate a new thought
  const newThought = await generateSingleThought(node.content, problem);
  
  // Create a new child node
  const childNode = {
    content: newThought,
    parent: node,
    children: [],
    visits: 0,
    value: 0
  };
  
  // Add the child to the parent
  node.children.push(childNode);
  
  return childNode;
}

async function simulate(node, problem) {
  // Simulate a random playout from this node
  let currentContent = node.content;
  const maxSteps = 3;
  
  for (let step = 0; step < maxSteps; step++) {
    // Generate a random next thought
    currentContent = await generateRandomThought(currentContent, problem);
  }
  
  // Evaluate the final state
  const evaluation = await evaluateSolution(currentContent, problem);
  return evaluation.score;
}

function backpropagate(node, reward) {
  // Update statistics for all nodes in the path from node to root
  let current = node;
  while (current) {
    current.visits += 1;
    current.value += reward;
    current = current.parent;
  }
}

function getBestChild(node) {
  // Return the child with the highest value
  return node.children.sort((a, b) => b.value / b.visits - a.value / a.visits)[0];
}
```

## Best Practices

1. **Start with diverse initial thoughts** to explore different parts of the solution space
2. **Use appropriate evaluation criteria** for the specific problem type
3. **Balance exploration vs. exploitation** when deciding which branches to expand
4. **Adjust branching factor and depth** based on problem complexity
5. **Consider computational efficiency** by pruning unpromising branches early
6. **Combine with other techniques** like Chain-of-Thought for individual reasoning steps
7. **Provide clear evaluation metrics** in your prompts to help the model assess thoughts

## When to Use

Tree of Thoughts is particularly effective for:
- Complex reasoning problems with multiple valid approaches
- Problems where the initial approach might lead to dead ends
- Tasks requiring creative problem-solving
- Multi-step planning problems
- Situations where exploring alternative solutions is valuable
- Problems where the optimal solution path is not immediately obvious 