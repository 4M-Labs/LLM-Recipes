import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def create_causal_mask(self, seq_length):
        # Create causal mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        causal_mask = self.create_causal_mask(seq_length).to(x.device)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax and attention
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        # Reshape and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Causal self-attention
        self.attention = CausalSelfAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT uses GELU instead of ReLU
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture (used in GPT)
        attn_input = self.norm1(x)
        x = x + self.dropout(self.attention(attn_input))
        
        ff_input = self.norm2(x)
        x = x + self.dropout(self.feed_forward(ff_input))
        
        return x

class GPTModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, vocab_size, dropout=0.1):
        super().__init__()
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length
        
    def forward(self, x):
        batch_size, seq_length = x.size()
        assert seq_length <= self.max_seq_length, "Sequence length exceeds maximum length"
        
        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Apply embeddings
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        
        # Apply decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Project to vocabulary
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
                logits = logits[:, -1, :] / temperature  # Only take the last token's predictions
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to the sequence
                x = torch.cat([x, next_token], dim=1)
            
        return x

# Example usage
def main():
    # Model parameters
    d_model = 768        # Embedding dimension
    num_heads = 12       # Number of attention heads
    num_layers = 12      # Number of decoder blocks
    d_ff = 3072         # Feed-forward dimension
    max_seq_length = 1024 # Maximum sequence length
    vocab_size = 50257   # GPT-2 vocabulary size
    
    # Create model
    model = GPTModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size
    )
    
    # Example input (batch_size=2, seq_length=10)
    x = torch.randint(0, vocab_size, (2, 10))
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Example generation
    prompt = torch.randint(0, vocab_size, (1, 5))  # Small prompt for demonstration
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.7)
    print(f"Generated sequence shape: {generated.shape}")

if __name__ == "__main__":
    main()
