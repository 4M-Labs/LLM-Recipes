import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
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
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (for padding tokens)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and attention
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        
        # Reshape and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # BERT uses GELU
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class BERTModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, vocab_size, num_segments=2, dropout=0.1):
        super().__init__()
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # MLM prediction head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # NSP prediction head
        self.nsp_head = nn.Linear(d_model, 2)
        
        self.max_seq_length = max_seq_length
        
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        assert seq_length <= self.max_seq_length, "Sequence length exceeds maximum length"
        
        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        
        # Add segment embeddings if provided
        if segment_ids is not None:
            segment_embeddings = self.segment_embedding(segment_ids)
            embeddings = token_embeddings + position_embeddings + segment_embeddings
        else:
            embeddings = token_embeddings + position_embeddings
            
        x = self.dropout(embeddings)
        
        # Apply encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask)
            
        x = self.norm(x)
        
        # Get sequence output and pooled output
        sequence_output = x
        pooled_output = sequence_output[:, 0]  # Use [CLS] token representation
        
        # Get MLM and NSP predictions
        mlm_logits = self.mlm_head(sequence_output)
        nsp_logits = self.nsp_head(pooled_output)
        
        return mlm_logits, nsp_logits, sequence_output

    def get_embeddings(self, input_ids, attention_mask=None):
        """Get contextual embeddings for input tokens."""
        with torch.no_grad():
            _, _, sequence_output = self.forward(input_ids, attention_mask=attention_mask)
        return sequence_output

# Example usage
def main():
    # Model parameters
    d_model = 768        # Embedding dimension
    num_heads = 12       # Number of attention heads
    num_layers = 12      # Number of encoder blocks
    d_ff = 3072         # Feed-forward dimension
    max_seq_length = 512 # Maximum sequence length
    vocab_size = 30522   # BERT vocabulary size
    
    # Create model
    model = BERTModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size
    )
    
    # Example inputs
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    segment_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    attention_mask = torch.ones(batch_size, 1, seq_length, seq_length)
    
    # Forward pass
    mlm_logits, nsp_logits, sequence_output = model(
        input_ids=input_ids,
        segment_ids=segment_ids,
        attention_mask=attention_mask
    )
    
    print(f"Input shape: {input_ids.shape}")
    print(f"MLM logits shape: {mlm_logits.shape}")
    print(f"NSP logits shape: {nsp_logits.shape}")
    print(f"Sequence output shape: {sequence_output.shape}")
    
    # Get embeddings example
    embeddings = model.get_embeddings(input_ids, attention_mask)
    print(f"Embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()
