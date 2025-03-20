/**
 * Routing Agent Pattern Implementation
 * 
 * This file demonstrates the Routing Agent pattern using the Vercel AI SDK.
 * It showcases how to route customer support queries to different specialized agents
 * based on query classification.
 */
import { openai } from '@ai-sdk/openai';
import { generateObject, generateText } from 'ai';
import { z } from 'zod';

/**
 * Schema for structured output from the classification step
 */
const classificationSchema = z.object({
  reasoning: z.string().min(10).describe('Explanation of the classification reasoning'),
  type: z.enum(['technical', 'billing', 'general', 'complaint']).describe('The category of customer query'),
  complexity: z.enum(['simple', 'complex']).describe('How complex the query is to resolve'),
  urgency: z.enum(['low', 'medium', 'high']).describe('How urgent the customer query is'),
  sentimentScore: z.number().min(-1).max(1).describe('Customer sentiment score from -1 (negative) to 1 (positive)')
});

/**
 * Main function to handle a customer support query using the routing pattern
 * 
 * @param {string} query The customer's query text
 * @returns {Promise<Object>} Object containing the response and the classification data
 */
export async function handleCustomerQuery(query) {
  console.log('Processing customer query:', query);
  
  // Step 1: Classify the query
  const classification = await classifyCustomerQuery(query);
  console.log('Query classified as:', classification);
  
  // Step 2: Route to appropriate specialized agent based on classification
  const response = await routeQueryToAgent(query, classification);
  
  // Return both the response and the classification for transparency
  return {
    response,
    classification
  };
}

/**
 * Classifies a customer query into different categories
 * 
 * @param {string} query The customer's query text
 * @returns {Promise<Object>} Classification of the query
 */
async function classifyCustomerQuery(query) {
  const model = openai('gpt-4o-mini'); // Using a smaller model for classification
  
  // Use generateObject to get structured classification data
  const { object: classification } = await generateObject({
    model,
    schema: classificationSchema,
    prompt: `Classify this customer query:
    "${query}"
    
    Determine:
    1. Query type (technical, billing, general, or complaint)
    2. Complexity (simple or complex)
    3. Urgency level (low, medium, high)
    4. Customer sentiment score (-1 to 1)
    5. Brief reasoning for your classification
    
    Be objective and thorough in your assessment.`,
  });
  
  return classification;
}

/**
 * Routes the query to an appropriate specialized agent based on classification
 * 
 * @param {string} query The original customer query
 * @param {Object} classification The query classification
 * @returns {Promise<string>} Response text for the customer
 */
async function routeQueryToAgent(query, classification) {
  // Select appropriate model based on complexity and urgency
  let model;
  if (classification.complexity === 'simple' && classification.urgency !== 'high') {
    // Use smaller model for simple, non-urgent queries
    model = openai('gpt-4o-mini');
    console.log('Using lightweight model for simple query');
  } else {
    // Use more capable model for complex or urgent queries
    model = openai('gpt-4o');
    console.log('Using advanced model for complex/urgent query');
  }
  
  // Route to the appropriate specialized agent based on query type
  switch (classification.type) {
    case 'technical':
      return await technicalSupportAgent(query, classification, model);
    case 'billing':
      return await billingAgent(query, classification, model);
    case 'complaint':
      return await complaintAgent(query, classification, model);
    case 'general':
    default:
      return await generalInformationAgent(query, classification, model);
  }
}

/**
 * Technical Support Agent - Specialized for technical issues
 */
async function technicalSupportAgent(query, classification, model) {
  const { text: response } = await generateText({
    model,
    system: `You are a technical support specialist with deep product knowledge.
      Provide clear, step-by-step troubleshooting instructions.
      Use simple language and avoid technical jargon when possible.
      If you need more information, ask specific diagnostic questions.
      Current customer sentiment: ${classification.sentimentScore < 0 ? 'negative' : 'positive'}
      
      ${classification.urgency === 'high' ? "This is a high-priority issue. Provide the most direct solution available." : ""}`,
    prompt: query,
  });
  
  return response;
}

/**
 * Billing Agent - Specialized for billing and payment issues
 */
async function billingAgent(query, classification, model) {
  const { text: response } = await generateText({
    model,
    system: `You are a billing specialist who handles payment and subscription inquiries.
      Be precise about financial information and policies.
      Explain charges clearly and provide specific next steps for resolution.
      Maintain a professional and reassuring tone.
      Current customer sentiment: ${classification.sentimentScore < 0 ? 'negative' : 'positive'}
      
      ${classification.urgency === 'high' ? "This is a high-priority issue. Provide the most direct solution available." : ""}`,
    prompt: query,
  });
  
  return response;
}

/**
 * Complaint Agent - Specialized for handling customer complaints
 */
async function complaintAgent(query, classification, model) {
  const { text: response } = await generateText({
    model,
    system: `You are a customer relations specialist who excels at handling complaints.
      Express genuine empathy for the customer's frustration.
      Acknowledge the issue without admitting fault or making promises you can't keep.
      Focus on what can be done to improve the situation.
      Offer specific next steps for resolution.
      Current customer sentiment: ${classification.sentimentScore < 0 ? 'very negative' : 'somewhat negative'}
      
      ${classification.urgency === 'high' ? "This is a high-priority complaint. Provide the most direct path to resolution." : ""}`,
    prompt: query,
  });
  
  return response;
}

/**
 * General Information Agent - Handles general inquiries
 */
async function generalInformationAgent(query, classification, model) {
  const { text: response } = await generateText({
    model,
    system: `You are a friendly customer service agent handling general inquiries.
      Be concise but thorough. Always maintain a professional tone.
      Provide accurate information about products, services, and policies.
      If you don't know something, be honest about it.
      Current customer sentiment: ${classification.sentimentScore < 0 ? 'negative' : 'positive'}`,
    prompt: query,
  });
  
  return response;
}

// Example usage
if (require.main === module) {
  const exampleQueries = [
    "I can't log into my account. I've tried resetting my password twice already.",
    "I was charged twice for my last month's subscription. I need a refund immediately!",
    "I'd like to know your store hours for this weekend.",
    "This is the third time I've contacted you about this issue. Your service is terrible!"
  ];
  
  (async () => {
    for (const query of exampleQueries) {
      console.log("\n-----------------------------------------------");
      console.log(`QUERY: ${query}`);
      const result = await handleCustomerQuery(query);
      console.log(`CLASSIFICATION: ${JSON.stringify(result.classification, null, 2)}`);
      console.log(`RESPONSE: ${result.response}`);
      console.log("-----------------------------------------------\n");
    }
  })();
} 