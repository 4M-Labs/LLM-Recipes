# Automatic Prompt Engineering (APE)

Automatic Prompt Engineering (APE) is a technique that uses language models to generate, evaluate, and optimize prompts automatically. Instead of manually crafting prompts through trial and error, APE leverages the capabilities of LLMs to create effective prompts for specific tasks.

## How It Works

APE typically follows these steps:

1. **Task Definition**: Clearly define the task and desired output format
2. **Prompt Generation**: Use an LLM to generate candidate prompts
3. **Prompt Evaluation**: Test the generated prompts against evaluation criteria
4. **Prompt Refinement**: Iteratively improve prompts based on performance
5. **Selection**: Choose the best-performing prompt for deployment

This approach can discover prompting strategies that might not be obvious to human engineers and can systematically optimize for specific performance metrics.

## Example Implementation

```javascript
import { generateText } from 'ai';

async function automaticPromptEngineering(task, examples, evaluationCriteria, numCandidates = 5) {
  // Step 1: Generate candidate prompts
  const promptGenerationResponse = await generateText({
    model: yourModel,
    system: `You are an expert prompt engineer. Your task is to generate effective prompts for language models.`,
    prompt: `Task: ${task}
    
    Examples of input-output pairs:
    ${examples.map(ex => `Input: ${ex.input}\nExpected Output: ${ex.output}`).join('\n\n')}
    
    Evaluation criteria:
    ${evaluationCriteria.join('\n')}
    
    Generate ${numCandidates} different candidate prompts that would help a language model perform this task effectively.
    Each prompt should be designed to elicit the expected output given the input.
    
    Format your response as:
    Prompt 1: [prompt text]
    
    Prompt 2: [prompt text]
    
    ...and so on.`,
  });
  
  // Parse the candidate prompts
  const candidatePrompts = promptGenerationResponse
    .split('\n\n')
    .filter(block => block.trim().startsWith('Prompt'))
    .map(block => {
      const promptText = block.substring(block.indexOf(':') + 1).trim();
      return promptText;
    });
  
  // Step 2: Evaluate each candidate prompt
  const evaluationResults = [];
  
  for (const prompt of candidatePrompts) {
    const promptPerformance = await evaluatePrompt(prompt, examples);
    evaluationResults.push({
      prompt,
      performance: promptPerformance
    });
  }
  
  // Step 3: Analyze results and select the best prompt
  const analysisResponse = await generateText({
    model: yourModel,
    system: `You are an expert at analyzing prompt performance and selecting the best prompts.`,
    prompt: `Task: ${task}
    
    Candidate prompts and their performance:
    ${evaluationResults.map(result => 
      `Prompt: ${result.prompt}\nPerformance:\n${JSON.stringify(result.performance, null, 2)}`
    ).join('\n\n')}
    
    Based on the performance data, analyze each prompt's strengths and weaknesses.
    Then select the best prompt for this task and explain your reasoning.
    
    Format your response as:
    Analysis:
    [Your analysis of each prompt's performance]
    
    Best Prompt:
    [The selected prompt]
    
    Reasoning:
    [Your reasoning for selecting this prompt]`,
  });
  
  // Step 4: Refine the best prompt
  const bestPromptMatch = analysisResponse.match(/Best Prompt:\s*(.*?)(?=\n\n|$)/s);
  const bestPrompt = bestPromptMatch ? bestPromptMatch[1].trim() : candidatePrompts[0];
  
  const refinedPromptResponse = await generateText({
    model: yourModel,
    system: `You are an expert prompt engineer focused on refining and improving prompts.`,
    prompt: `Task: ${task}
    
    Current best prompt:
    ${bestPrompt}
    
    Performance of this prompt:
    ${JSON.stringify(evaluationResults.find(r => r.prompt === bestPrompt)?.performance || {}, null, 2)}
    
    Please refine this prompt to improve its performance. Consider:
    1. Adding more specific instructions
    2. Improving clarity
    3. Adding examples if helpful
    4. Addressing any weaknesses identified in the analysis
    
    Provide the refined prompt and explain your changes.`,
  });
  
  return {
    task,
    candidatePrompts,
    evaluationResults,
    analysis: analysisResponse,
    refinedPrompt: refinedPromptResponse
  };
}

async function evaluatePrompt(prompt, examples) {
  const results = [];
  
  for (const example of examples) {
    // Test the prompt with each example
    const response = await generateText({
      model: yourModel,
      prompt: `${prompt}\n\nInput: ${example.input}`
    });
    
    // Calculate similarity/accuracy metrics
    const similarity = calculateSimilarity(response, example.output);
    
    results.push({
      input: example.input,
      expectedOutput: example.output,
      actualOutput: response,
      similarity
    });
  }
  
  // Calculate aggregate metrics
  const averageSimilarity = results.reduce((sum, r) => sum + r.similarity, 0) / results.length;
  
  return {
    individualResults: results,
    averageSimilarity,
    successRate: results.filter(r => r.similarity > 0.8).length / results.length
  };
}

function calculateSimilarity(text1, text2) {
  // Simple implementation - in practice, use more sophisticated metrics
  // like BLEU, ROUGE, or embedding similarity
  const words1 = text1.toLowerCase().split(/\s+/);
  const words2 = text2.toLowerCase().split(/\s+/);
  
  const intersection = words1.filter(word => words2.includes(word)).length;
  const union = new Set([...words1, ...words2]).size;
  
  return intersection / union;
}
```

## Using the Vercel AI SDK

```javascript
import { streamText } from 'ai';

async function vercelAPE(task, examples) {
  // Step 1: Generate candidate prompts
  const promptGenerationStream = await streamText({
    model: yourModel,
    system: `You are an expert prompt engineer who creates effective prompts for language models.`,
    prompt: `Task: ${task}
    
    Examples of input-output pairs:
    ${examples.map(ex => `Input: ${ex.input}\nExpected Output: ${ex.output}`).join('\n\n')}
    
    Generate 3 different candidate prompts that would help a language model perform this task effectively.
    Each prompt should be designed to elicit the expected output given the input.
    
    Format your response as:
    Prompt 1: [prompt text]
    
    Prompt 2: [prompt text]
    
    Prompt 3: [prompt text]`,
  });
  
  let promptGenerationText = '';
  for await (const chunk of promptGenerationStream) {
    promptGenerationText += chunk;
  }
  
  // Parse the candidate prompts
  const candidatePrompts = promptGenerationText
    .split('\n\n')
    .filter(block => block.trim().startsWith('Prompt'))
    .map(block => {
      const promptText = block.substring(block.indexOf(':') + 1).trim();
      return promptText;
    });
  
  // Step 2: Test each prompt with examples
  const testResults = [];
  
  for (const prompt of candidatePrompts) {
    const promptResults = [];
    
    for (const example of examples) {
      const testStream = await streamText({
        model: yourModel,
        prompt: `${prompt}\n\nInput: ${example.input}`
      });
      
      let response = '';
      for await (const chunk of testStream) {
        response += chunk;
      }
      
      promptResults.push({
        input: example.input,
        expectedOutput: example.output,
        actualOutput: response
      });
    }
    
    testResults.push({
      prompt,
      results: promptResults
    });
  }
  
  // Step 3: Analyze results and select the best prompt
  const analysisStream = await streamText({
    model: yourModel,
    system: `You are an expert at analyzing prompt performance and selecting the best prompts.`,
    prompt: `Task: ${task}
    
    Candidate prompts and their results:
    ${testResults.map(result => 
      `Prompt: ${result.prompt}\n\nResults:\n${result.results.map(r => 
        `Input: ${r.input}\nExpected: ${r.expectedOutput}\nActual: ${r.actualOutput}`
      ).join('\n\n')}`
    ).join('\n\n')}
    
    Analyze each prompt's performance and select the best one for this task.
    
    Format your response as:
    Analysis:
    [Your analysis of each prompt's performance]
    
    Best Prompt:
    [The selected prompt]
    
    Reasoning:
    [Your reasoning for selecting this prompt]`,
  });
  
  let analysisText = '';
  for await (const chunk of analysisStream) {
    analysisText += chunk;
  }
  
  // Extract the best prompt
  const bestPromptMatch = analysisText.match(/Best Prompt:\s*(.*?)(?=\n\n|$)/s);
  const bestPrompt = bestPromptMatch ? bestPromptMatch[1].trim() : candidatePrompts[0];
  
  return {
    task,
    candidatePrompts,
    testResults,
    analysis: analysisText,
    bestPrompt
  };
}
```

## Advanced APE Patterns

### Meta-Prompt Optimization

```javascript
async function metaPromptOptimization(task, examples) {
  // Generate a meta-prompt that will be used to generate task-specific prompts
  const metaPromptResponse = await generateText({
    model: yourModel,
    system: `You are an expert at creating meta-prompts that generate effective task-specific prompts.`,
    prompt: `Task: ${task}
    
    Examples of input-output pairs:
    ${examples.map(ex => `Input: ${ex.input}\nExpected Output: ${ex.output}`).join('\n\n')}
    
    Create a meta-prompt that, when given to a language model, will instruct it to generate 
    an effective prompt for the task described above.
    
    The meta-prompt should include:
    1. Instructions for analyzing the task
    2. Guidelines for creating effective prompts
    3. Considerations specific to this type of task
    4. Format requirements for the generated prompt
    
    Format your response as:
    Meta-Prompt: [your meta-prompt]`,
  });
  
  // Extract the meta-prompt
  const metaPromptMatch = metaPromptResponse.match(/Meta-Prompt:\s*(.*)/s);
  const metaPrompt = metaPromptMatch ? metaPromptMatch[1].trim() : "";
  
  // Use the meta-prompt to generate a task-specific prompt
  const taskPromptResponse = await generateText({
    model: yourModel,
    prompt: metaPrompt
  });
  
  // Test the generated prompt
  const testResults = [];
  
  for (const example of examples) {
    const response = await generateText({
      model: yourModel,
      prompt: `${taskPromptResponse}\n\nInput: ${example.input}`
    });
    
    testResults.push({
      input: example.input,
      expectedOutput: example.output,
      actualOutput: response
    });
  }
  
  return {
    task,
    metaPrompt,
    generatedPrompt: taskPromptResponse,
    testResults
  };
}
```

### Evolutionary Prompt Optimization

```javascript
async function evolutionaryPromptOptimization(task, examples, generations = 3, populationSize = 4) {
  // Initial population
  let population = await generateInitialPrompts(task, examples, populationSize);
  
  for (let generation = 0; generation < generations; generation++) {
    // Evaluate fitness
    const evaluatedPopulation = await Promise.all(
      population.map(async prompt => {
        const fitness = await evaluatePromptFitness(prompt, examples);
        return { prompt, fitness };
      })
    );
    
    // Sort by fitness
    evaluatedPopulation.sort((a, b) => b.fitness - a.fitness);
    
    // Select the top half as parents
    const parents = evaluatedPopulation
      .slice(0, Math.ceil(populationSize / 2))
      .map(item => item.prompt);
    
    // Generate new population through crossover and mutation
    const offspring = await generateOffspring(parents, task, examples, populationSize - parents.length);
    
    // New population is top parents + offspring
    population = [...parents, ...offspring];
    
    console.log(`Generation ${generation + 1} best fitness: ${evaluatedPopulation[0].fitness}`);
  }
  
  // Final evaluation
  const finalEvaluation = await Promise.all(
    population.map(async prompt => {
      const fitness = await evaluatePromptFitness(prompt, examples);
      return { prompt, fitness };
    })
  );
  
  // Return the best prompt
  finalEvaluation.sort((a, b) => b.fitness - a.fitness);
  return finalEvaluation[0].prompt;
}

async function generateInitialPrompts(task, examples, count) {
  const response = await generateText({
    model: yourModel,
    system: `You are an expert prompt engineer.`,
    prompt: `Task: ${task}
    
    Examples of input-output pairs:
    ${examples.map(ex => `Input: ${ex.input}\nExpected Output: ${ex.output}`).join('\n\n')}
    
    Generate ${count} diverse and effective prompts for this task.
    Each prompt should be designed to elicit the expected output given the input.
    
    Format your response as:
    Prompt 1: [prompt text]
    
    Prompt 2: [prompt text]
    
    ...and so on.`,
  });
  
  // Parse the prompts
  const prompts = response
    .split('\n\n')
    .filter(block => block.trim().startsWith('Prompt'))
    .map(block => block.substring(block.indexOf(':') + 1).trim());
  
  return prompts.slice(0, count);
}

async function evaluatePromptFitness(prompt, examples) {
  const results = [];
  
  for (const example of examples) {
    const response = await generateText({
      model: yourModel,
      prompt: `${prompt}\n\nInput: ${example.input}`
    });
    
    // Calculate similarity score
    const similarity = calculateSimilarity(response, example.output);
    results.push(similarity);
  }
  
  // Average similarity across all examples
  return results.reduce((sum, score) => sum + score, 0) / results.length;
}

async function generateOffspring(parents, task, examples, count) {
  const promptsDescription = parents.map((p, i) => `Parent ${i + 1}: ${p}`).join('\n\n');
  
  const response = await generateText({
    model: yourModel,
    system: `You are an expert at evolving and improving prompts.`,
    prompt: `Task: ${task}
    
    Parent prompts:
    ${promptsDescription}
    
    Generate ${count} new prompts by combining elements from the parent prompts and introducing beneficial mutations.
    The new prompts should maintain the strengths of the parents while addressing their weaknesses.
    
    Format your response as:
    Offspring 1: [prompt text]
    
    Offspring 2: [prompt text]
    
    ...and so on.`,
  });
  
  // Parse the offspring prompts
  const offspring = response
    .split('\n\n')
    .filter(block => block.trim().startsWith('Offspring'))
    .map(block => block.substring(block.indexOf(':') + 1).trim());
  
  return offspring.slice(0, count);
}
```

## Best Practices

1. **Define clear evaluation criteria** for prompt performance
2. **Use diverse examples** to ensure robust prompt generation
3. **Test across multiple models** if the prompt will be used with different LLMs
4. **Consider both accuracy and efficiency** in your evaluation metrics
5. **Iterate multiple times** to refine and improve prompts
6. **Combine automatic and manual refinement** for best results
7. **Document the prompt engineering process** for future reference
8. **Test edge cases** to ensure prompt robustness

## When to Use

Automatic Prompt Engineering is particularly effective for:
- Complex tasks where manual prompt engineering is challenging
- Applications requiring optimal performance across many examples
- Scenarios where prompt performance needs to be quantitatively measured
- Projects with sufficient examples to evaluate prompt quality
- Tasks where slight improvements in prompt effectiveness have significant value
- Situations requiring systematic exploration of different prompting strategies 