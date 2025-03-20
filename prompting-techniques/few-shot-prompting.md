# Few-Shot Prompting

Few-shot prompting is a technique where you provide the LLM with a few examples of the task you want it to perform before asking it to complete a similar task. This helps guide the model's responses by demonstrating the expected format and reasoning.

## How It Works

In few-shot prompting, you provide:
- A set of example inputs and their corresponding outputs
- The new input for which you want an output
- Optional instructions to clarify the task

The model then uses pattern recognition to generate an appropriate response for the new input.

## Basic Example

```javascript
import { generateText } from 'ai';

const result = await generateText({
  model: "llama-3-70b-instruct",
  prompt: `
    Classify the sentiment of the following movie reviews as positive, negative, or neutral.
    
    Review: "This movie was terrible. I hated every minute of it."
    Sentiment: Negative
    
    Review: "The film was okay, but nothing special."
    Sentiment: Neutral
    
    Review: "I absolutely loved this movie! The acting was superb."
    Sentiment: Positive
    
    Review: "The special effects were amazing, but the plot made no sense."
    Sentiment: 
  `,
  temperature: 0.1, // Lower temperature for classification tasks
});

console.log(result.text); // Expected: "Negative" or "Mixed"
```

## Advanced Few-Shot Techniques

### Structured Few-Shot

Using a consistent structure with clear delimiters:

```javascript
const structuredResult = await generateText({
  model: "llama-3-70b-instruct",
  prompt: `
    # Task: Convert the given sentence into a question.
    
    ## Example 1
    Sentence: The capital of France is Paris.
    Question: What is the capital of France?
    
    ## Example 2
    Sentence: Dogs are mammals.
    Question: Are dogs mammals?
    
    ## Example 3
    Sentence: The Earth orbits around the Sun.
    Question: What does the Earth orbit around?
    
    ## Your Turn
    Sentence: Photosynthesis is the process by which plants make food.
    Question:
  `,
  temperature: 0.3,
  max_tokens: 50, // Limit response length for concise answers
});

console.log(structuredResult.text);
```

### Few-Shot with Reasoning

Including the reasoning process in examples:

```javascript
const reasoningResult = await generateText({
  model: "llama-3-70b-instruct",
  prompt: `
    Solve the following word problems by reasoning step by step.
    
    Problem: If a shirt costs $15 and is on sale for 20% off, how much does it cost?
    Reasoning: To find the sale price, I need to subtract the discount from the original price.
    The discount is 20% of $15, which is 0.2 × $15 = $3.
    So the sale price is $15 - $3 = $12.
    Answer: $12
    
    Problem: A train travels at 60 miles per hour. How far will it travel in 2.5 hours?
    Reasoning: To find the distance, I multiply the speed by the time.
    Distance = 60 miles per hour × 2.5 hours = 150 miles.
    Answer: 150 miles
    
    Problem: If a recipe calls for 2/3 cup of flour and I want to make 1.5 batches, how much flour do I need?
    Reasoning:
  `,
  temperature: 0.2, // Lower temperature for reasoning tasks
});

console.log(reasoningResult.text);
```

### Dynamic Few-Shot Selection

Selecting the most relevant examples for each query:

```javascript
// Example database of input-output pairs
const exampleDatabase = [
  {
    input: "The food was delicious and the service was excellent.",
    output: "Positive",
    category: "restaurant"
  },
  {
    input: "The hotel room was spacious and clean.",
    output: "Positive",
    category: "hotel"
  },
  {
    input: "The flight was delayed and the staff was unhelpful.",
    output: "Negative",
    category: "airline"
  },
  {
    input: "The product arrived damaged and customer service was unresponsive.",
    output: "Negative",
    category: "retail"
  },
  // More examples...
];

// Function to select the most relevant examples based on the query
async function selectRelevantExamples(query, numExamples = 3) {
  // In a real implementation, you would use embeddings to find similar examples
  // For simplicity, we'll use a keyword-based approach here
  const categories = {
    restaurant: ["food", "restaurant", "meal", "dinner", "lunch", "chef", "waiter"],
    hotel: ["hotel", "room", "stay", "accommodation", "lobby", "resort"],
    airline: ["flight", "plane", "airport", "airline", "travel", "pilot", "attendant"],
    retail: ["product", "purchase", "delivery", "store", "shop", "item", "customer"]
  };
  
  // Determine the likely category of the query
  let queryCategory = "general";
  let maxMatches = 0;
  
  for (const [category, keywords] of Object.entries(categories)) {
    const matches = keywords.filter(keyword => 
      query.toLowerCase().includes(keyword.toLowerCase())).length;
    
    if (matches > maxMatches) {
      maxMatches = matches;
      queryCategory = category;
    }
  }
  
  // Select examples from the matching category, or general examples if no match
  const relevantExamples = exampleDatabase
    .filter(example => example.category === queryCategory)
    .slice(0, numExamples);
  
  // If we don't have enough examples from the category, add some general ones
  if (relevantExamples.length < numExamples) {
    const generalExamples = exampleDatabase
      .filter(example => example.category !== queryCategory)
      .slice(0, numExamples - relevantExamples.length);
    
    relevantExamples.push(...generalExamples);
  }
  
  return relevantExamples;
}

// Function to generate a few-shot prompt with dynamically selected examples
async function dynamicFewShotPrompt(query) {
  const relevantExamples = await selectRelevantExamples(query);
  
  let prompt = "Classify the sentiment of the following text as positive, negative, or neutral.\n\n";
  
  // Add the selected examples to the prompt
  for (const example of relevantExamples) {
    prompt += `Text: "${example.input}"\nSentiment: ${example.output}\n\n`;
  }
  
  // Add the query
  prompt += `Text: "${query}"\nSentiment:`;
  
  const result = await generateText({
    model: "llama-3-70b-instruct",
    prompt,
    temperature: 0.1,
  });
  
  return result.text;
}

// Example usage
const query = "The hotel restaurant served amazing food but the room was tiny.";
dynamicFewShotPrompt(query).then(sentiment => {
  console.log(`Query: ${query}`);
  console.log(`Sentiment: ${sentiment}`);
});
```

## Using LangChain for Few-Shot Prompting

```javascript
import {
  ChatPromptTemplate,
  FewShotChatMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  temperature: 0.1, // Lower temperature for more consistent results
});

const examples = [
  {
    input: "The food was delicious and the service was excellent.",
    output: "Positive",
  },
  {
    input: "The room was dirty and the staff was rude.",
    output: "Negative",
  },
  {
    input: "The hotel was adequate for our needs.",
    output: "Neutral",
  },
  {
    input: "The concert was amazing but the venue was too crowded.",
    output: "Mixed",
  },
];

// Create a system message to set the context
const systemTemplate = "You are a sentiment analysis expert. Classify the sentiment of the text as positive, negative, neutral, or mixed.";
const systemMessagePrompt = SystemMessagePromptTemplate.fromTemplate(systemTemplate);

// Create the example prompt template
const examplePrompt = ChatPromptTemplate.fromTemplate(`Human: {input}
AI: {output}`);

// Create the few-shot prompt with examples
const fewShotPrompt = new FewShotChatMessagePromptTemplate({
  prefix: "Here are some examples of sentiment classification:",
  suffix: "Human: {input}\nAI:",
  examplePrompt,
  examples,
  inputVariables: ["input"],
});

// Combine the system message and few-shot examples
const finalPrompt = ChatPromptTemplate.fromMessages([
  systemMessagePrompt,
  fewShotPrompt,
]);

// Generate the response
const formattedPrompt = await finalPrompt.formatMessages({
  input: "The movie had great actors but a confusing plot.",
});

const response = await model.invoke(formattedPrompt);

console.log(response.content);
```

## Best Practices

1. **Choose diverse, representative examples** that cover different aspects of the task
2. **Order matters**: The sequence of examples can influence the model's understanding
3. **Use consistent formatting** across all examples
4. **Include edge cases** when relevant
5. **Keep examples concise** but informative
6. **Use 3-5 examples** for optimal results (more isn't always better)
7. **Maintain consistent complexity** across examples
8. **Include both positive and negative examples** when applicable
9. **Use clear delimiters** between examples (e.g., numbered lists, headers, separators)
10. **Test different example sets** to find the most effective combinations
11. **Consider example relevance** to the specific query or task
12. **Balance example diversity** with task specificity
13. **Use structured formats** with clear input-output separation
14. **Incorporate reasoning steps** for complex tasks
15. **Adjust temperature settings** based on the desired consistency

## When to Use

Few-shot prompting is ideal for:
- Tasks requiring specific output formats
- Classification or categorization tasks
- When zero-shot prompting produces inconsistent results
- Teaching the model a pattern it might not have seen during pre-training
- Fine-tuning the model's response style without formal fine-tuning
- Domain-specific tasks where the model needs guidance on terminology or conventions
- Tasks with clear input-output mappings
- Situations where you need consistent formatting across multiple responses
- Applications requiring specialized knowledge representation

## Limitations

- Examples take up token space, reducing the available context for complex tasks
- Performance depends on the quality and relevance of the chosen examples
- May not work well for tasks requiring deep reasoning beyond the patterns shown
- Can introduce biases based on the selected examples
- Not as effective as fine-tuning for tasks requiring consistent performance at scale
- Sensitive to example ordering and presentation
- May struggle with tasks that differ significantly from the provided examples
- Token limitations restrict the number of examples you can include

## Research Insights

Research by Brown et al. (2020) in the GPT-3 paper demonstrated that few-shot learning enables language models to perform tasks with minimal examples, without parameter updates. More recent studies have shown that:

- The quality of examples matters more than quantity
- Strategically selected examples outperform randomly chosen ones
- Including reasoning steps in examples improves performance on complex tasks
- Models can be sensitive to the order and presentation of examples

The 2022 paper "Rethinking the Role of Demonstrations" by Min et al. found that the label space and format of examples are often more important than the specific content of the examples themselves.

Research by Liu et al. (2023) in "Lost in the Middle" demonstrated that examples placed in the middle of the context window are less effective than those at the beginning or end, suggesting that example placement matters.

The 2023 paper "Active Few-Shot Learning" by Diao et al. showed that dynamically selecting the most relevant examples for each query can significantly improve performance compared to using a fixed set of examples.

## Real-World Applications

- **Custom classifiers**: Creating specialized categorization systems without training data
- **Format conversion**: Transforming data from one format to another
- **Language translation**: Providing examples of translation pairs for specific domains
- **Code generation**: Showing patterns for generating code in specific styles or frameworks
- **Content creation**: Guiding the tone, style, and structure of generated content
- **Named entity recognition**: Identifying specific types of information in text
- **Data extraction**: Pulling structured information from unstructured text
- **Text summarization**: Demonstrating the desired summary style and length
- **Question answering**: Showing how to format answers for different question types
- **Dialogue systems**: Setting the conversational style and response patterns 