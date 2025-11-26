"""
Wide & Deep Embedding Layer for LLMs
Based on the RecSys architecture principle of separating memorization (wide) from generalization (deep)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WideDeepEmbedding(nn.Module):
    """
    Wide & Deep Embedding Layer that combines static (wide) and dynamic (deep) embeddings.
    
    The wide component uses pre-trained static embeddings (like Word2Vec/GloVe) for fixed semantics,
    while the deep component uses learnable embeddings for contextual understanding.
    
    Reference: Wide & Deep Learning for Recommender Systems (Cheng et al., 2016)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        static_embeddings: Optional[torch.Tensor] = None,
        static_embedding_dim: Optional[int] = None,
        dropout: float = 0.1,
        freeze_static: bool = True
    ):
        """
        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of the final combined embedding
            static_embeddings: Pre-trained static embeddings (e.g., Word2Vec, GloVe)
            static_embedding_dim: Dimension of static embeddings (if different from embedding_dim)
            dropout: Dropout rate
            freeze_static: Whether to freeze the static embeddings (recommended)
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.freeze_static = freeze_static
        
        # Dynamic (deep) component: learnable embeddings for contextual understanding
        self.dynamic_embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # Wide component: static embeddings for memorization of fixed semantics
        if static_embeddings is not None:
            self.static_embedding_dim = static_embeddings.size(1)
            self.static_embeddings = nn.Embedding.from_pretrained(
                static_embeddings, 
                freeze=freeze_static
            )
            
            # If static embedding dim is different from target dim, use projection
            if self.static_embedding_dim != embedding_dim:
                self.static_projection = nn.Linear(self.static_embedding_dim, embedding_dim)
            else:
                self.static_projection = nn.Identity()
        else:
            # If no static embeddings provided, create a smaller learnable component
            self.static_embedding_dim = embedding_dim // 2  # Use half dim for static
            self.static_embeddings = nn.Embedding(num_embeddings, self.static_embedding_dim)
            self.static_projection = nn.Linear(self.static_embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable weights for combining static and dynamic components
        self.wide_weight = nn.Parameter(torch.tensor(0.5))  # Weight for static component
        self.deep_weight = nn.Parameter(torch.tensor(0.5))  # Weight for dynamic component
        
        # Optional MLP to transform dynamic embeddings before combination
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining wide (static) and deep (dynamic) embeddings
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Combined embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        # Get dynamic (deep) embeddings
        dynamic_emb = self.dynamic_embedding(input_ids)  # (B, S, D)
        
        # Get static (wide) embeddings
        static_emb = self.static_embeddings(input_ids)  # (B, S, D_static)
        
        # Project static embeddings to target dimension if needed
        static_emb = self.static_projection(static_emb)  # (B, S, D)
        
        # Transform dynamic embeddings with MLP
        dynamic_emb_transformed = self.mlp(dynamic_emb)  # (B, S, D)
        
        # Combine wide and deep components with learnable weights
        combined_emb = (self.wide_weight * static_emb + 
                       self.deep_weight * dynamic_emb_transformed)
        
        # Apply dropout and return
        return self.dropout(combined_emb)


class WideDeepTransformerBlock(nn.Module):
    """
    A Transformer block that incorporates Wide & Deep principles at the embedding level
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        static_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        # Wide & Deep embedding layer
        self.token_embedding = WideDeepEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            static_embeddings=static_embeddings,
            dropout=dropout
        )
        
        # Positional embedding (learnable)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        
        # Standard transformer components
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, n_head, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, idx: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = idx.size()
        
        # Get token embeddings using Wide & Deep approach
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        
        # Add positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Apply layer norm
        x = self.ln1(x)
        
        # Self-attention with optional mask
        attn_out, _ = self.attn(x, x, x, attn_mask=attention_mask)
        x = x + attn_out  # Residual connection
        
        # Apply layer norm before MLP
        x = self.ln2(x)
        
        # MLP
        mlp_out = self.mlp(x)
        x = x + mlp_out  # Residual connection
        
        return x


def create_static_embeddings_from_glove(vocab_size: int, embedding_dim: int = 100) -> torch.Tensor:
    """
    Helper function to create dummy static embeddings (in practice, these would come from Word2Vec/GloVe)
    """
    # In real implementation, you'd load pre-trained embeddings
    # This creates random embeddings for demonstration purposes
    return torch.randn(vocab_size, embedding_dim)


# Example usage
if __name__ == "__main__":
    # Example parameters
    vocab_size = 1000
    embedding_dim = 256
    batch_size = 4
    seq_len = 16
    
    # Create dummy static embeddings (in practice load from Word2Vec/GloVe)
    static_embs = create_static_embeddings_from_glove(vocab_size, embedding_dim=100)
    
    # Initialize the Wide & Deep embedding layer
    wide_deep_emb = WideDeepEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        static_embeddings=static_embs,
        freeze_static=True
    )
    
    # Create some dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = wide_deep_emb(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Static component weight: {wide_deep_emb.wide_weight.item():.3f}")
    print(f"Dynamic component weight: {wide_deep_emb.deep_weight.item():.3f}")
    
    # Test the full transformer block
    transformer_block = WideDeepTransformerBlock(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        n_head=8,
        block_size=seq_len
    )
    
    block_output = transformer_block(input_ids)
    print(f"Transformer block output shape: {block_output.shape}")