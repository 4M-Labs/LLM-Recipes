# LLM Prompting Techniques

This directory contains a collection of advanced prompting techniques for Large Language Models (LLMs). Each technique is documented with explanations, examples, and best practices to help you get the most out of your interactions with LLMs.

## Techniques Overview

### Basic Techniques

- [Zero-Shot Prompting](./zero-shot-prompting.md) - Getting LLMs to perform tasks without examples
- [Few-Shot Prompting](./few-shot-prompting.md) - Providing examples to guide LLM responses
- [Chain-of-Thought (CoT)](./chain-of-thought.md) - Encouraging step-by-step reasoning
- [System Prompts](./system-prompts.md) - Setting the behavior and context for the LLM
- [Structured Output](./structured-output.md) - Getting responses in specific formats (JSON, XML, etc.)
- [Role Prompting](./role-prompting.md) - Assigning specific roles to guide LLM behavior

### Advanced Techniques

- [Self-Consistency](./self-consistency.md) - Generating multiple reasoning paths and selecting the most consistent answer
- [ReAct Prompting](./react-prompting.md) - Combining reasoning and acting for complex problem-solving
- [Tree of Thoughts (ToT)](./tree-of-thoughts.md) - Exploring multiple reasoning paths simultaneously
- [Automatic Reasoning and Tool-use (ART)](./art-prompting.md) - Systematic task decomposition and tool use
- [Multimodal Prompting](./multimodal-prompting.md) - Working with multiple types of media (text, images, etc.)
- [Automatic Prompt Engineering (APE)](./automatic-prompt-engineering.md) - Using LLMs to generate and optimize prompts
- [Directional Stimulus Prompting](./directional-stimulus-prompting.md) - Guiding responses in specific directions
- [Prompt Chaining](./prompt-chaining.md) - Breaking complex tasks into sequences of simpler prompts
- [Retrieval-Augmented Generation (RAG)](./retrieval-augmented-generation.md) - Enhancing responses with retrieved information
- [Agent Prompting](./agent-prompting.md) - Creating autonomous agents that can perform tasks

## How to Use This Repository

Each technique is documented in its own markdown file with the following structure:

1. **Definition** - What the technique is and how it works
2. **Examples** - Code samples demonstrating implementation
3. **Advanced Patterns** - More sophisticated applications of the technique
4. **Best Practices** - Tips for effective use
5. **When to Use** - Scenarios where the technique is most effective

## Choosing the Right Technique

The effectiveness of each technique depends on your specific use case:

- For complex reasoning tasks, consider **Chain-of-Thought**, **Tree of Thoughts**, or **Self-Consistency**
- For tasks requiring external information, use **ReAct**, **ART**, or **RAG**
- For consistent, structured outputs, use **Structured Output** or **System Prompts**
- For creative or diverse responses, try **Directional Stimulus Prompting**
- For optimizing prompts automatically, use **APE**
- For autonomous task completion, use **Agent Prompting**

## Combining Techniques

Many of these techniques can be combined for even more powerful results. Common combinations include:

- **Few-Shot** + **Chain-of-Thought** for guided reasoning
- **ReAct** + **Structured Output** for tool use with formatted results
- **System Prompts** + **Role Prompting** for consistent persona-based responses
- **RAG** + **Self-Consistency** for factual accuracy with consistent reasoning
- **Agent Prompting** + **Tree of Thoughts** for agents that explore multiple solution paths

## Resources

- [Together AI Cookbook](https://github.com/togethercomputer/cookbook) - Source for many of these techniques
- [Anthropic Claude Documentation](https://docs.anthropic.com/claude/docs/introduction-to-prompting)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## Contributing

Feel free to suggest improvements or additional techniques by opening an issue or pull request.

## License

This collection is provided under the MIT License. 