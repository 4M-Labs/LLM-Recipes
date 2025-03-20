# LLM Architecture

Explore common architectural patterns for building robust and scalable LLM-powered applications.

## Transformer Architecture (Foundation)

The foundational architecture using self-attention mechanisms that revolutionized NLP.

```mermaid
graph LR
    A[Input] --> B[Self-Attention]
    B --> C[Output]
    
    style A fill:#4285f4,stroke:#4285f4,color:white
    style B fill:#4285f4,stroke:#4285f4,color:white
    style C fill:#4285f4,stroke:#4285f4,color:white
```

## Decoder-only Models (GPT-style)

GPT-style architectures focused on text generation (GPT-4, Claude, Llama).

```mermaid
graph LR
    A[Input] --> B[Decoder]
    B --> C[Output]
    
    style A fill:#34A853,stroke:#34A853,color:white
    style B fill:#34A853,stroke:#34A853,color:white
    style C fill:#34A853,stroke:#34A853,color:white
```

## Encoder-only Models (BERT-style)

BERT-style architectures focused on understanding (good for embeddings).

```mermaid
graph LR
    A[Input] --> B[Encoder]
    B --> C[Embeddings]
    
    style A fill:#9334EA,stroke:#9334EA,color:white
    style B fill:#9334EA,stroke:#9334EA,color:white
    style C fill:#9334EA,stroke:#9334EA,color:white
```

## Encoder-Decoder Models (T5-style)

T5-style architectures for translation and summarization tasks.

```mermaid
graph LR
    A[Input] --> B[Encoder]
    B --> C[Decoder]
    C --> D[Output]
    
    style A fill:#EA580C,stroke:#EA580C,color:white
    style B fill:#EA580C,stroke:#EA580C,color:white
    style C fill:#EA580C,stroke:#EA580C,color:white
    style D fill:#EA580C,stroke:#EA580C,color:white
```

## Mixture of Experts (Mixtral)

Models like Mixtral with specialized subnetworks activated based on input.

```mermaid
graph TD
    A[Input] --> B[Router]
    B --> C[Expert 1]
    B --> D[Expert 2]
    B --> E[Expert 3]
    C --> F[Output]
    D --> F
    E --> F
    
    style A fill:#DC2626,stroke:#DC2626,color:white
    style B fill:#DC2626,stroke:#DC2626,color:white
    style C fill:#DC2626,stroke:#DC2626,color:white
    style D fill:#DC2626,stroke:#DC2626,color:white
    style E fill:#DC2626,stroke:#DC2626,color:white
    style F fill:#DC2626,stroke:#DC2626,color:white
```

## Sparse Attention Mechanisms (Efficiency)

Alternate attention patterns to reduce computational complexity.

```mermaid
graph LR
    A[Input] --> B[Sparse Attention]
    B --> C[Output]
    
    style A fill:#0D9488,stroke:#0D9488,color:white
    style B fill:#0D9488,stroke:#0D9488,color:white
    style C fill:#0D9488,stroke:#0D9488,color:white
```

Each pattern is optimized for specific use cases:
- **Transformer**: General-purpose foundation for NLP tasks
- **Decoder-only**: Text generation and completion
- **Encoder-only**: Understanding and embeddings
- **Encoder-Decoder**: Translation and summarization
- **Mixture of Experts**: Specialized task handling
- **Sparse Attention**: Efficient processing of long sequences
