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
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and reshape
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
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
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
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
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class T5Model(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, vocab_size, dropout=0.1):
        super().__init__()
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length
        
    def create_causal_mask(self, seq_length):
        # Create causal mask for decoder self-attention
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)
        
    def encode(self, src, src_mask=None):
        batch_size, seq_length = src.size()
        
        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=src.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Apply embeddings
        x = self.token_embedding(src) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Apply encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
            
        return x
        
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        batch_size, seq_length = tgt.size()
        
        # Create position indices
        positions = torch.arange(0, seq_length, dtype=torch.long, device=tgt.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Apply embeddings
        x = self.token_embedding(tgt) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Apply decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, memory, src_mask, tgt_mask)
            
        return x
        
    def forward(self, src, tgt):
        # Create attention masks
        src_mask = None  # Could be implemented for padding tokens
        tgt_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
        
        # Encode source sequence
        encoder_output = self.encode(src, src_mask)
        
        # Decode target sequence
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def generate(self, src, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            # Encode source sequence
            encoder_output = self.encode(src)
            
            # Initialize target sequence with start token
            tgt = torch.zeros(src.size(0), 1, dtype=torch.long, device=src.device)
            
            for _ in range(max_length - 1):
                # Decode current sequence
                decoder_output = self.decode(tgt, encoder_output)
                
                # Get next token predictions
                logits = self.output_projection(decoder_output[:, -1:]) / temperature
                next_token = torch.argmax(logits, dim=-1)
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if end token is generated
                if next_token.item() == 1:  # Assuming 1 is the end token
                    break
                    
            return tgt

# Example usage
def main():
    # Model parameters
    d_model = 512        # Embedding dimension
    num_heads = 8        # Number of attention heads
    num_layers = 6       # Number of encoder/decoder blocks
    d_ff = 2048         # Feed-forward dimension
    max_seq_length = 128 # Maximum sequence length
    vocab_size = 32000   # T5 vocabulary size
    
    # Create model
    model = T5Model(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        vocab_size=vocab_size
    )
    
    # Example inputs
    batch_size = 2
    src_length = 64
    tgt_length = 32
    src = torch.randint(0, vocab_size, (batch_size, src_length))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_length))
    
    # Forward pass
    logits = model(src, tgt)
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Generation example
    generated = model.generate(src, max_length=50)
    print(f"Generated sequence shape: {generated.shape}")

if __name__ == "__main__":
    main()
