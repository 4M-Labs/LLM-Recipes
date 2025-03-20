import torch
import torch.nn as nn
import math

class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64, sparsity_factor=4):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size
        self.sparsity_factor = sparsity_factor
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def create_sparse_mask(self, seq_length):
        """Create a sparse attention mask with local and strided patterns."""
        # Initialize mask
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
        
        # Add local attention (each token attends to nearby tokens)
        for i in range(seq_length):
            start = max(0, i - self.block_size // 2)
            end = min(seq_length, i + self.block_size // 2 + 1)
            mask[i, start:end] = True
        
        # Add strided attention (each token attends to tokens at regular intervals)
        for i in range(seq_length):
            for j in range(i % self.sparsity_factor, seq_length, self.sparsity_factor):
                mask[i, j] = True
        
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply sparse attention mask
        sparse_mask = self.create_sparse_mask(seq_length).to(x.device)
        attn_scores = attn_scores.masked_fill(~sparse_mask, float('-inf'))
        
        # Apply softmax and attention
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        # Reshape and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output

class SparseTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, block_size=64, sparsity_factor=4, dropout=0.1):
        super().__init__()
        
        # Sparse self-attention
        self.attention = SparseAttention(d_model, num_heads, block_size, sparsity_factor)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SparseTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, 
                 vocab_size, block_size=64, sparsity_factor=4, dropout=0.1):
        super().__init__()
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Sparse transformer blocks
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                d_model, num_heads, d_ff, block_size, sparsity_factor, dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length
        
    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        assert seq_length <= self.max_seq_length, "Sequence length exceeds maximum length"
        
        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Apply embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
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

class LSHAttention(nn.Module):
    """Alternative sparse attention using Locality-Sensitive Hashing."""
    def __init__(self, d_model, num_heads, num_hashes=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_hashes = num_hashes
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Random projection matrix for hashing
        self.R = nn.Parameter(torch.randn(self.num_hashes, self.d_k))
        
    def hash_vectors(self, vectors):
        # Project vectors and get hash buckets
        projections = torch.matmul(vectors, self.R.t())
        buckets = torch.argmax(projections, dim=-1)
        return buckets
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Hash queries and keys
        Q_buckets = self.hash_vectors(Q.view(-1, self.d_k)).view(batch_size, seq_length, self.num_heads)
        K_buckets = self.hash_vectors(K.view(-1, self.d_k)).view(batch_size, seq_length, self.num_heads)
        
        # Create attention mask based on bucket assignments
        mask = (Q_buckets.unsqueeze(-1) == K_buckets.unsqueeze(-2))
        mask = mask.unsqueeze(1)  # Add head dimension
        
        # Compute attention scores only for tokens in same bucket
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax and attention
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        # Reshape and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output

# Example usage
def main():
    # Model parameters
    d_model = 512        # Embedding dimension
    num_heads = 8        # Number of attention heads
    num_layers = 6       # Number of transformer blocks
    d_ff = 2048         # Feed-forward dimension
    max_seq_length = 4096 # Maximum sequence length
    vocab_size = 32000   # Vocabulary size
    block_size = 64      # Local attention window size
    sparsity_factor = 8  # Stride for global attention
    
    # Create model
    model = SparseTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size,
        block_size=block_size,
        sparsity_factor=sparsity_factor
    )
    
    # Example input
    batch_size = 2
    seq_length = 1024
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Generation example
    prompt = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.7)
    print(f"Generated sequence shape: {generated.shape}")
    
    # Example with LSH attention
    lsh_attention = LSHAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_length, d_model)
    output = lsh_attention(x)
    print(f"LSH attention output shape: {output.shape}")

if __name__ == "__main__":
    main()
