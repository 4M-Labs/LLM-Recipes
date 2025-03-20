"""
Routing Agent Pattern Implementation

This file demonstrates the Routing Agent pattern using the OpenAI API.
It showcases how to route customer support queries to different specialized agents
based on query classification.
"""

import os
from typing import Dict, Any, TypedDict, Literal
import openai
from termcolor import colored

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Type definitions
class QueryClassification(TypedDict):
    reasoning: str
    type: Literal['technical', 'billing', 'general', 'complaint']
    complexity: Literal['simple', 'complex']
    urgency: Literal['low', 'medium', 'high']
    sentiment_score: float  # -1 to 1, where -1 is very negative, 1 is very positive

class CustomerQueryResult(TypedDict):
    response: str
    classification: QueryClassification

def classify_customer_query(query: str) -> QueryClassification:
    """
    Classifies a customer query into different categories
    
    Args:
        query: The customer's query text
        
    Returns:
        Classification of the query
    """
    print(colored("Classifying customer query...", "blue"))
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Using a smaller model for classification
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are a query classifier. Analyze the customer query and provide:
                1. Query type (technical, billing, general, or complaint)
                2. Complexity (simple or complex)
                3. Urgency level (low, medium, high)
                4. Customer sentiment score (-1 to 1)
                5. Brief reasoning for your classification
                
                Return your analysis in JSON format with the following fields:
                {
                    "reasoning": string,
                    "type": "technical" | "billing" | "general" | "complaint",
                    "complexity": "simple" | "complex",
                    "urgency": "low" | "medium" | "high",
                    "sentiment_score": number (-1 to 1)
                }"""
            },
            {
                "role": "user",
                "content": f'Classify this customer query: "{query}"'
            }
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content

def route_query_to_agent(query: str, classification: QueryClassification) -> str:
    """
    Routes the query to an appropriate specialized agent based on classification
    
    Args:
        query: The original customer query
        classification: The query classification
        
    Returns:
        Response text for the customer
    """
    # Select appropriate model based on complexity and urgency
    if classification['complexity'] == 'simple' and classification['urgency'] != 'high':
        # Use smaller model for simple, non-urgent queries
        model = "gpt-4o-mini"
        print(colored("Using lightweight model for simple query", "cyan"))
    else:
        # Use more capable model for complex or urgent queries
        model = "gpt-4o"
        print(colored("Using advanced model for complex/urgent query", "cyan"))
    
    # Route to the appropriate specialized agent based on query type
    if classification['type'] == 'technical':
        return technical_support_agent(query, classification, model)
    elif classification['type'] == 'billing':
        return billing_agent(query, classification, model)
    elif classification['type'] == 'complaint':
        return complaint_agent(query, classification, model)
    else:
        return general_information_agent(query, classification, model)

def technical_support_agent(query: str, classification: QueryClassification, model: str) -> str:
    """Technical Support Agent - Specialized for technical issues"""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are a technical support specialist with deep product knowledge.
                Provide clear, step-by-step troubleshooting instructions.
                Use simple language and avoid technical jargon when possible.
                If you need more information, ask specific diagnostic questions.
                Current customer sentiment: {'negative' if classification['sentiment_score'] < 0 else 'positive'}
                
                {f"This is a high-priority issue. Provide the most direct solution available." if classification['urgency'] == 'high' else ""}"""
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def billing_agent(query: str, classification: QueryClassification, model: str) -> str:
    """Billing Agent - Specialized for billing and payment issues"""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are a billing specialist who handles payment and subscription inquiries.
                Be precise about financial information and policies.
                Explain charges clearly and provide specific next steps for resolution.
                Maintain a professional and reassuring tone.
                Current customer sentiment: {'negative' if classification['sentiment_score'] < 0 else 'positive'}
                
                {f"This is a high-priority issue. Provide the most direct solution available." if classification['urgency'] == 'high' else ""}"""
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def complaint_agent(query: str, classification: QueryClassification, model: str) -> str:
    """Complaint Agent - Specialized for handling customer complaints"""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are a customer relations specialist who excels at handling complaints.
                Express genuine empathy for the customer's frustration.
                Acknowledge the issue without admitting fault or making promises you can't keep.
                Focus on what can be done to improve the situation.
                Offer specific next steps for resolution.
                Current customer sentiment: {'very negative' if classification['sentiment_score'] < -0.5 else 'somewhat negative'}
                
                {f"This is a high-priority complaint. Provide the most direct path to resolution." if classification['urgency'] == 'high' else ""}"""
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def general_information_agent(query: str, classification: QueryClassification, model: str) -> str:
    """General Information Agent - Handles general inquiries"""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are a friendly customer service agent handling general inquiries.
                Be concise but thorough. Always maintain a professional tone.
                Provide accurate information about products, services, and policies.
                If you don't know something, be honest about it.
                Current customer sentiment: {'negative' if classification['sentiment_score'] < 0 else 'positive'}"""
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def handle_customer_query(query: str) -> CustomerQueryResult:
    """
    Main function to handle a customer support query using the routing pattern
    
    Args:
        query: The customer's query text
        
    Returns:
        Object containing the response and the classification data
    """
    print(colored(f"Processing customer query: {query}", "yellow"))
    
    # Step 1: Classify the query
    classification = classify_customer_query(query)
    print(colored("Query classified as:", "green"), classification)
    
    # Step 2: Route to appropriate specialized agent based on classification
    response = route_query_to_agent(query, classification)
    
    # Return both the response and the classification for transparency
    return {
        "response": response,
        "classification": classification
    }

if __name__ == "__main__":
    # Example usage
    example_queries = [
        "I can't log into my account. I've tried resetting my password twice already.",
        "I was charged twice for my last month's subscription. I need a refund immediately!",
        "I'd like to know your store hours for this weekend.",
        "This is the third time I've contacted you about this issue. Your service is terrible!"
    ]
    
    for query in example_queries:
        print("\n" + "="*50)
        print(colored(f"QUERY: {query}", "yellow"))
        result = handle_customer_query(query)
        print(colored("\nCLASSIFICATION:", "cyan"), result["classification"])
        print(colored("\nRESPONSE:", "green"), result["response"])
        print("="*50 + "\n") 