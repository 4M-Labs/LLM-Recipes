import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExpertLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.feed_forward(x)

class Router(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        
        self.gate = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.top_k = top_k
        
    def forward(self, x):
        # Calculate routing weights
        routing_weights = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Get top-k experts for each token
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        return top_k_weights, top_k_indices

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        
        self.router = Router(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertLayer(d_model, d_ff)
            for _ in range(num_experts)
        ])
        
        self.num_experts = num_experts
        self.top_k = top_k
        
    def forward(self, x):
        batch_size, seq_length, d_model = x.size()
        
        # Get routing weights and expert assignments
        routing_weights, expert_indices = self.router(x)
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Process input through experts
        for k in range(self.top_k):
            # Get k-th expert indices and weights
            k_expert_indices = expert_indices[..., k]  # [batch_size, seq_length]
            k_routing_weights = routing_weights[..., k].unsqueeze(-1)  # [batch_size, seq_length, 1]
            
            # Process through each expert
            for expert_idx in range(self.num_experts):
                # Create mask for current expert
                expert_mask = (k_expert_indices == expert_idx)
                if not expert_mask.any():
                    continue
                
                # Select tokens for current expert
                expert_input = x[expert_mask]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Add weighted output to result
                output[expert_mask] += k_routing_weights[expert_mask] * expert_output
        
        return output

class MixtralBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Mixture of experts
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts)
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        x = residual + self.dropout(x)
        
        # Mixture of experts
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.moe(x))
        
        return x

class MixtralModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, num_experts, 
                 max_seq_length, vocab_size, dropout=0.1):
        super().__init__()
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Mixtral blocks
        self.blocks = nn.ModuleList([
            MixtralBlock(d_model, num_heads, d_ff, num_experts, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        assert seq_length <= self.max_seq_length, "Sequence length exceeds maximum length"
        
        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Apply embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Apply Mixtral blocks
        for block in self.blocks:
            x = block(x, attention_mask)
            
        # Apply final normalization and projection
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, prompt, max_new_tokens, temperature=1.0):
        self.eval()
        with torch.no_grad():
            # Start with the prompt
            x = prompt
            
            for _ in range(max_new_tokens):
                # Take the last max_seq_length tokens if input is too long
                x_cropped = x[:, -self.max_seq_length:]
                
                # Get predictions
                logits = self(x_cropped)
                logits = logits[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to the sequence
                x = torch.cat([x, next_token], dim=1)
                
                # Stop if end token is generated
                if next_token.item() == 1:  # Assuming 1 is the end token
                    break
            
            return x

# Example usage
def main():
    # Model parameters
    d_model = 1024       # Embedding dimension
    num_heads = 16       # Number of attention heads
    num_layers = 32      # Number of Mixtral blocks
    d_ff = 4096         # Feed-forward dimension
    num_experts = 8      # Number of experts
    max_seq_length = 2048 # Maximum sequence length
    vocab_size = 32000   # Vocabulary size
    
    # Create model
    model = MixtralModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_experts=num_experts,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size
    )
    
    # Example input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length).bool()
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Generation example
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.7)
    print(f"Generated sequence shape: {generated.shape}")

if __name__ == "__main__":
    main()
