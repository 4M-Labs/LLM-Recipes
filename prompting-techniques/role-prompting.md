# Role Prompting

Role prompting is a technique where you assign a specific persona, role, or character to the LLM to guide its responses. This approach helps shape the model's tone, expertise level, perspective, and response style by framing its identity within a particular context.

## How It Works

In role prompting, you:
1. Define a specific role for the model to assume
2. Optionally provide characteristics of that role
3. Frame your request within the context of that role
4. Optionally include examples of how that role would respond

The model then responds from the perspective of the assigned role, drawing on its pre-trained understanding of how someone in that position might communicate.

## Basic Example

```javascript
import { generateText } from 'ai';

const doctorResponse = await generateText({
  model: "llama-3-70b-instruct",
  system: "You are an experienced medical doctor with a specialization in cardiology. You explain medical concepts in clear, accessible language while maintaining scientific accuracy.",
  prompt: "What are the warning signs of a heart attack that people often ignore?",
  temperature: 0.4, // Lower temperature for factual medical information
});

console.log(doctorResponse.text);
```

## Advanced Role Prompting Techniques

### Multi-dimensional Role Definition

Defining a role across multiple dimensions for a richer persona:

```javascript
const financialAdvisorResponse = await generateText({
  model: "llama-3-70b-instruct",
  system: `You are a certified financial advisor with the following characteristics:
  
  - Expertise: 15+ years in personal finance, retirement planning, and investment management
  - Communication style: Concise, straightforward, and jargon-free
  - Approach: Conservative and risk-aware, focused on long-term growth
  - Values: Transparency, education, and client empowerment
  - Background: Former economics professor with a real-world focus
  
  Always disclose when financial advice is general in nature and may not apply to specific situations. Avoid making specific investment recommendations about individual securities.`,
  prompt: "I'm 30 years old and just started my first job with a 401(k) option. How should I think about allocating my investments?",
  temperature: 0.3, // Conservative for financial advice
});

console.log(financialAdvisorResponse.text);
```

### Historical or Fictional Character Roles

Adopting the perspective of a specific character:

```javascript
const einsteinResponse = await generateText({
  model: "llama-3-70b-instruct",
  system: `You are Albert Einstein in 1935. Respond as Einstein would, incorporating:
  
  - His communication style and speech patterns
  - References to his theories and scientific work up to 1935
  - His philosophical views on science, pacifism, and humanism
  - Occasional German expressions he might use
  - His playful sense of humor and thoughtful analogies
  
  However, avoid the use of complex mathematical formulas as Einstein would typically simplify complex ideas for general audiences.`,
  prompt: "Could you explain your concept of space-time to a high school student?",
  temperature: 0.7, // Higher temperature for creative expression
});

console.log(einsteinResponse.text);
```

### Expert Collaboration Role

Creating a panel of experts for multi-perspective analysis:

```javascript
const expertPanelResponse = await generateText({
  model: "llama-3-70b-instruct",
  system: `You are a panel of three different experts analyzing the same problem:
  
  - Dr. Chen, Economist: Focuses on economic implications, market forces, and financial impacts. Data-driven and analytical.
  - Professor Rodriguez, Social Scientist: Examines social factors, community effects, equity concerns, and human behavior. Compassionate and systems-oriented.
  - Taylor Kim, Technology Futurist: Considers technological trends, innovation opportunities, and future scenarios. Forward-thinking and solution-oriented.
  
  For each question, provide perspectives from all three experts, clearly labeled. Highlight both agreements and differences in their viewpoints.`,
  prompt: "What are the implications of widespread adoption of remote work after the pandemic?",
  temperature: 0.5, // Balanced for diverse but grounded perspectives
});

console.log(expertPanelResponse.text);
```

### Role with Constrained Knowledge

Limiting a role to a specific time period or knowledge domain:

```javascript
const historicalDoctorResponse = await generateText({
  model: "llama-3-70b-instruct",
  system: `You are a physician from the year 1850. Your knowledge and beliefs about medicine are limited to what was known in that era. You should:
  
  - Only reference medical theories, treatments, and terminology available in 1850
  - Express beliefs common to medical practitioners of that time period
  - Not reference modern medical knowledge, technologies, or discoveries that occurred after 1850
  - Use period-appropriate language and terminology
  
  Important: This is for educational purposes about historical medical perspectives. Include a disclaimer that this represents historical views only and should not be taken as medical advice.`,
  prompt: "What would you recommend for a patient with a high fever and persistent cough?",
  temperature: 0.6, // Moderate temperature for historical perspective
});

console.log(historicalDoctorResponse.text);
```

### Anti-Role Prompting

Defining what the role is explicitly NOT:

```javascript
const technicalWriterResponse = await generateText({
  model: "llama-3-70b-instruct",
  system: `You are a technical writer for developer documentation with these specific characteristics:
  
  - You ARE: Precise, concise, and technically accurate
  - You ARE: Focused on practical examples and use cases
  - You ARE: Organized and structured in your explanations
  
  - You are NOT: Marketing-oriented or promotional in tone
  - You are NOT: Using unnecessary jargon to sound impressive
  - You are NOT: Verbose or repetitive
  
  Your goal is to create documentation that helps developers implement features quickly and correctly, not to sell or promote the technology.`,
  prompt: "Explain how to implement JWT authentication in a Node.js API.",
  temperature: 0.2, // Low temperature for precise technical writing
});

console.log(technicalWriterResponse.text);
```

### Evolving Role

A role that changes during the interaction:

```javascript
const mentorResponse = await generateText({
  model: "llama-3-70b-instruct",
  system: `You are a programming mentor who adapts your teaching style based on the learner's progress. You evolve through three stages:
  
  1. Beginner stage: Very supportive, explains basic concepts thoroughly, avoids jargon, uses simple examples, focuses on fundamentals, and encourages experimentation.
  
  2. Intermediate stage: More technical language, introduces best practices, provides more nuanced explanations, encourages problem-solving before giving answers, and introduces edge cases.
  
  3. Advanced stage: Discusses system design, points to documentation rather than explaining basics, asks probing questions, introduces advanced concepts, and treats the learner as a peer.
  
  Determine the appropriate stage based on the complexity and sophistication of the learner's questions. Adjust your teaching style accordingly during the conversation.`,
  prompt: "I'm trying to build my first React app but I'm confused about how state works. Can you help me understand it?",
  temperature: 0.4, // Moderate temperature for teaching
});

console.log(mentorResponse.text);
```

## Using LangChain for Role Prompting

```javascript
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

async function getRoleResponse(role, prompt) {
  const chat = new ChatOpenAI({
    temperature: 0.5,
    modelName: "gpt-4",
  });
  
  const messages = [
    new SystemMessage(role),
    new HumanMessage(prompt),
  ];
  
  const response = await chat.invoke(messages);
  return response.content;
}

// Example usage
const teacherRole = `You are a high school physics teacher with 20 years of experience. 
You excel at creating engaging explanations with real-world examples that teenagers can relate to.
You use analogies, simple experiments students can do at home, and connections to popular movies or video games.
While you maintain scientific accuracy, you prioritize building intuition over mathematical rigor for introductory concepts.`;

const response = await getRoleResponse(
  teacherRole,
  "How do Newton's laws of motion apply to everyday life?"
);

console.log(response);
```

## Best Practices

1. **Be specific about expertise level** - Define not just the role but the level of expertise (e.g., "a senior software engineer with 15 years of experience in distributed systems")

2. **Include communication style** - Specify how the role should communicate (e.g., "explains complex topics using simple analogies")

3. **Define constraints** - Set boundaries on what the role does and doesn't do (e.g., "provides general advice but does not offer specific investment recommendations")

4. **Add personality elements** - Include personality traits that influence the responses (e.g., "methodical, detail-oriented, and likes to consider edge cases")

5. **Specify audience adaptation** - Instruct the role on how to adapt to the audience (e.g., "adjusts technical depth based on the questioner's apparent knowledge level")

6. **Include ethical guidelines** - Add ethical constraints appropriate to the role (e.g., "prioritizes patient safety and well-being above all else")

7. **Consider cultural and temporal context** - Specify the cultural background or time period if relevant (e.g., "a chef specializing in traditional Szechuan cuisine" or "a historian with expertise in Victorian England")

8. **Avoid stereotyping** - Craft roles based on expertise and communication style rather than stereotypical characteristics

9. **Test different phrasings** - The exact wording of role descriptions can significantly impact results

10. **Refine iteratively** - Start with a basic role description and refine based on the responses

## When to Use

Role prompting is particularly effective for:

- **Educational content**: When you need explanations from a specific perspective or expertise level
- **Creative writing**: To maintain consistent character voice and perspective
- **Professional guidance**: When you need advice from a particular professional viewpoint
- **Specialized knowledge domains**: To focus responses on a specific field of expertise
- **Balanced perspectives**: When you want to explore multiple viewpoints on a topic
- **Historical context**: To get period-appropriate responses
- **Technical vs. non-technical communication**: To control the level of technical detail
- **Emotional tone management**: To set a specific emotional tone for responses

## Role Prompting Templates

### Expert Advisor Template

```
You are a [PROFESSION] with [NUMBER] years of experience specializing in [SPECIALIZATION].

Your communication style is [STYLE CHARACTERISTICS].

When giving advice, you prioritize [PRIORITIES].

You typically address problems by [METHODOLOGY].

Important limitations:
- [LIMITATION 1]
- [LIMITATION 2]
- [LIMITATION 3]

Question: [USER QUESTION]
```

### Character Role-Play Template

```
You are [CHARACTER NAME] from [SOURCE/WORLD/TIME PERIOD].

Key personality traits:
- [TRAIT 1]
- [TRAIT 2]
- [TRAIT 3]

Your background includes:
- [BACKGROUND ELEMENT 1]
- [BACKGROUND ELEMENT 2]

When speaking, you typically:
- [SPEECH PATTERN 1]
- [SPEECH PATTERN 2]

Your worldview is characterized by [WORLDVIEW].

Respond to the following as this character would:
[USER PROMPT]
```

### Multi-Perspective Role Template

```
Analyze the following from three different perspectives:

1. As a [ROLE 1] focused on [FOCUS AREA 1]:
2. As a [ROLE 2] focused on [FOCUS AREA 2]:
3. As a [ROLE 3] focused on [FOCUS AREA 3]:

Topic to analyze:
[TOPIC]
```

## Research Insights

Research has shown that role prompting can significantly influence:

- **Response quality**: Appropriate roles can improve factual accuracy in domain-specific responses
- **Creativity**: Character-based roles can enhance creative writing outputs
- **Consistency**: Well-defined roles lead to more consistent response patterns
- **Ethical boundaries**: Roles can help reinforce ethical guidelines and limitations
- **Readability**: Professional roles often produce more organized and clear responses

A 2023 study on "The Effect of Persona Design on Output Quality in Large Language Models" found that:
- Detailed, multi-dimensional persona definitions outperformed simple role labels
- Including communication style preferences had the largest positive impact on response quality
- Different personas worked better for different tasks, with no "one-size-fits-all" best persona

## Real-World Applications

- **Educational content creation**: Role-playing experts in different fields to create learning materials
- **Content writing**: Adopting brand voices or character perspectives
- **UX writing**: Creating consistent voice and tone for product interfaces
- **Healthcare communication**: Simplifying complex medical information through a patient-friendly physician role
- **Legal document simplification**: Using a "legal translator" role to make complex legal concepts accessible
- **Technical support**: Creating step-by-step guides with appropriate technical depth
- **Market research**: Analyzing products or ideas from different stakeholder perspectives
- **Conflict resolution**: Exploring different viewpoints on controversial topics
- **Historical analysis**: Examining events through period-appropriate lenses
- **Creative writing assistance**: Maintaining consistent character voices in fiction

## Limitations

- **Stereotyping risk**: Poorly designed roles can reinforce stereotypes or biases
- **False authority**: The model has no actual credentials regardless of the role assigned
- **Knowledge limitations**: The model can only simulate expertise based on its training data
- **Temporal limitations**: Historical role-playing is limited by training data accuracy
- **Inconsistent adherence**: Models may occasionally break character or mix role characteristics
- **Hallucination risk**: Strong character roles might increase the risk of fabricated information
- **Cultural nuance challenges**: Cross-cultural roles may lack authenticity or nuance
