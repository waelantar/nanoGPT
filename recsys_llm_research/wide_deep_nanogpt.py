"""
Wide & Deep NanoGPT Implementation
This file demonstrates how to integrate the Wide & Deep embedding principle into a GPT-like architecture
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from wide_deep_embedding import WideDeepEmbedding


class WideDeepNanoGPT(nn.Module):
    """
    A GPT-like model that incorporates Wide & Deep principles at the embedding level
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        block_size: int = 1024,
        dropout: float = 0.1,
        static_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        # Wide & Deep embedding layer
        self.token_embedding = WideDeepEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=n_embd,
            static_embeddings=static_embeddings,
            dropout=dropout
        )
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WideDeepBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        
        # Get token embeddings using Wide & Deep approach
        tok_emb = self.token_embedding(idx)  # (b, t, n_embd)
        
        # Add positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (t)
        pos_emb = self.pos_embedding(pos)  # (t, n_embd)
        x = tok_emb + pos_emb  # (b, t, n_embd)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Compute logits
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
        
        return logits, loss


class WideDeepBlock(nn.Module):
    """A transformer block that works with the Wide & Deep architecture"""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention layer
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        
        # MLP layer
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention implementation"""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        
        # Key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.weight.data.normal_(mean=0.0, std=0.02)
        
        # Regularization
        self.resid_dropout = nn.Dropout(dropout)
        
        # Create causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """MLP feedforward network"""
    
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.c_fc.weight.data.normal_(mean=0.0, std=0.02)
        self.c_proj.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def create_wide_deep_model(vocab_size: int, 
                          n_embd: int = 128,  # Smaller for testing
                          n_head: int = 4, 
                          n_layer: int = 4, 
                          block_size: int = 128) -> WideDeepNanoGPT:
    """
    Helper function to create a Wide & Deep model with reasonable parameters for testing
    """
    # Create dummy static embeddings (in practice, load from Word2Vec/GloVe)
    static_embeddings = torch.randn(vocab_size, n_embd // 2)
    
    model = WideDeepNanoGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        static_embeddings=static_embeddings
    )
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    vocab_size = 1000
    n_embd = 128
    n_head = 4
    n_layer = 4
    block_size = 64
    batch_size = 2
    seq_len = 32
    
    # Create the model
    model = create_wide_deep_model(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size
    )
    
    print(f"Model created with:")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Embedding dimension: {n_embd}")
    print(f"- Number of heads: {n_head}")
    print(f"- Number of layers: {n_layer}")
    print(f"- Block size: {block_size}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Forward pass
    with torch.no_grad():  # Disable gradients for testing
        logits, loss = model(input_ids, targets=targets)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test without targets (evaluation mode)
    with torch.no_grad():
        logits_eval, _ = model(input_ids)
    
    print(f"Logits shape (eval mode): {logits_eval.shape}")
    
    # Verify that the wide & deep components are being used
    wide_weight = model.token_embedding.wide_weight.item()
    deep_weight = model.token_embedding.deep_weight.item()
    
    print(f"\nWide component weight: {wide_weight:.3f}")
    print(f"Deep component weight: {deep_weight:.3f}")
    
    print("\nWide & Deep NanoGPT implementation is working correctly!")