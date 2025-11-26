import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import pickle
import os


class PMISparseAttention(nn.Module):
    """
    Implements PMI (Pointwise Mutual Information) based sparse attention.
    Instead of computing attention between all tokens, this mechanism
    pre-computes token co-occurrence statistics and only attends to tokens
    with high PMI scores.
    """
    
    def __init__(self, embed_dim, num_heads, vocab_size, max_seq_len=2048, top_k=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.top_k = top_k  # Number of tokens to attend to per position
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Pre-computed PMI mask - will be computed during initialization
        self.register_buffer('pmi_mask', None)  # Shape: [vocab_size, top_k]
        
    def compute_pmi_matrix(self, token_cooc_counts, vocab_size, top_k=64):
        """
        Compute PMI matrix from co-occurrence counts.
        
        Args:
            token_cooc_counts: Dictionary of {(token_i, token_j): count}
            vocab_size: Size of vocabulary
            top_k: Number of highest-PMI tokens to keep for each token
            
        Returns:
            pmi_matrix: Sparse matrix of shape [vocab_size, top_k] with top-k PMI neighbors
        """
        # Calculate marginal probabilities
        token_counts = defaultdict(int)
        total_pairs = 0
        
        for (i, j), count in token_cooc_counts.items():
            token_counts[i] += count
            token_counts[j] += count
            total_pairs += count
        
        # Calculate PMI for each pair
        pmi_scores = defaultdict(float)
        
        for (i, j), count in token_cooc_counts.items():
            p_ij = count / total_pairs
            p_i = token_counts[i] / (2 * total_pairs)  # Account for (i,j) and (j,i)
            p_j = token_counts[j] / (2 * total_pairs)
            
            if p_i > 0 and p_j > 0 and p_ij > 0:
                pmi = np.log(p_ij / (p_i * p_j))
                pmi_scores[(i, j)] = pmi
                pmi_scores[(j, i)] = pmi  # Symmetric
        
        # For each token, find top-k tokens with highest PMI
        pmi_top_k = torch.zeros(vocab_size, top_k, dtype=torch.long)
        
        for token_id in range(vocab_size):
            # Get all PMI scores for this token
            token_pmi_scores = [(other_token, pmi) for (t1, t2), pmi in pmi_scores.items() 
                               if t1 == token_id or t2 == token_id
                               for other_token in [t1 if t2 == token_id else t2 if t1 == token_id else None] 
                               if other_token is not None]
            
            # Sort by PMI and keep top-k
            token_pmi_scores.sort(key=lambda x: x[1], reverse=True)
            top_tokens = [token for token, _ in token_pmi_scores[:top_k]]
            
            # Pad if necessary
            while len(top_tokens) < top_k:
                top_tokens.append(token_id)  # Self-attention if not enough neighbors
                
            pmi_top_k[token_id] = torch.tensor(top_tokens[:top_k])
        
        return pmi_top_k
    
    def build_pmi_mask(self, token_cooc_counts):
        """
        Build the PMI mask based on co-occurrence statistics.
        """
        self.pmi_mask = self.compute_pmi_matrix(token_cooc_counts, self.vocab_size, self.top_k)
    
    def forward(self, x, token_ids=None):
        """
        Forward pass with PMI sparse attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            token_ids: Token IDs of shape [batch_size, seq_len] for PMI lookup
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # If no token IDs provided, fall back to dense attention
        if token_ids is None or self.pmi_mask is None:
            # Dense attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # Sparse attention based on PMI
            attn_output = torch.zeros_like(q)
            
            for batch_idx in range(batch_size):
                # For each batch, find the positions of each vocabulary token in the sequence
                # This creates a mapping from vocab tokens to their positions in the current sequence
                token_to_positions = {}
                for pos in range(seq_len):
                    token_id = token_ids[batch_idx, pos].item()
                    if token_id not in token_to_positions:
                        token_to_positions[token_id] = []
                    token_to_positions[token_id].append(pos)
                
                for pos in range(seq_len):
                    # Get the token ID for this position
                    current_token_id = token_ids[batch_idx, pos].item()
                    
                    # Get top-k tokens with highest PMI to current token
                    if self.pmi_mask is not None and current_token_id < self.pmi_mask.size(0):
                        pmi_related_tokens = self.pmi_mask[current_token_id]
                        
                        # Find which of these PMI-related tokens actually appear in the current sequence
                        valid_positions = []
                        for pmi_token in pmi_related_tokens:
                            pmi_token_id = pmi_token.item()
                            if pmi_token_id in token_to_positions:
                                # Add all positions where this PMI-related token appears
                                valid_positions.extend(token_to_positions[pmi_token_id])
                        
                        # If no related tokens are in the sequence, fall back to attending to all
                        if len(valid_positions) == 0:
                            # Fallback to dense attention for this position
                            k_dense = k[batch_idx, :, :, :]  # [num_heads, seq_len, head_dim]
                            v_dense = v[batch_idx, :, :, :]  # [num_heads, seq_len, head_dim]
                            q_single = q[batch_idx, :, pos:pos+1, :]  # [num_heads, 1, head_dim]
                            attn_weights = torch.matmul(q_single, k_dense.transpose(-2, -1)) / (self.head_dim ** 0.5)
                            attn_weights = F.softmax(attn_weights, dim=-1)
                            attn_result = torch.matmul(attn_weights, v_dense)  # [num_heads, 1, head_dim]
                            attn_output[batch_idx, :, pos:pos+1, :] = attn_result
                        else:
                            # Remove duplicates and limit the number of positions to prevent excessive computation
                            valid_positions = list(set(valid_positions))[:self.top_k]
                            
                            # Get K and V for these specific positions
                            pos_tensor = torch.tensor(valid_positions, dtype=torch.long, device=k.device)
                            k_sparse = k[batch_idx, :, pos_tensor, :]  # [num_heads, num_valid_pos, head_dim]
                            v_sparse = v[batch_idx, :, pos_tensor, :]  # [num_heads, num_valid_pos, head_dim]
                            
                            # Compute attention only with these tokens
                            q_single = q[batch_idx, :, pos:pos+1, :]  # [num_heads, 1, head_dim]
                            attn_weights = torch.matmul(q_single, k_sparse.transpose(-2, -1)) / (self.head_dim ** 0.5)
                            attn_weights = F.softmax(attn_weights, dim=-1)
                            
                            # Apply attention to values
                            attn_result = torch.matmul(attn_weights, v_sparse)  # [num_heads, 1, head_dim]
                            attn_output[batch_idx, :, pos:pos+1, :] = attn_result
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(attn_output)


class PMISparseGPTBlock(nn.Module):
    """
    A GPT-style transformer block with PMI sparse attention.
    """
    
    def __init__(self, embed_dim, num_heads, vocab_size, max_seq_len=2048, top_k=64, ff_multiplier=4):
        super().__init__()
        self.attention = PMISparseAttention(embed_dim, num_heads, vocab_size, max_seq_len, top_k)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(ff_multiplier * embed_dim, embed_dim)
        )
    
    def forward(self, x, token_ids=None):
        # Attention with residual connection
        attn_out = self.attention(self.norm1(x), token_ids)
        x = x + attn_out
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x


class PMISparseGPT(nn.Module):
    """
    Full GPT model with PMI sparse attention.
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len=2048, top_k=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer blocks with PMI sparse attention
        self.blocks = nn.ModuleList([
            PMISparseGPTBlock(embed_dim, num_heads, vocab_size, max_seq_len, top_k)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        
        # Embeddings
        x = self.token_embedding(token_ids) + self.pos_embedding[:, :seq_len, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, token_ids)
        
        # Final norm and output
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def build_pmi_mask_from_data(self, token_cooc_counts):
        """
        Build PMI mask from co-occurrence counts for all attention layers.
        """
        for block in self.blocks:
            block.attention.build_pmi_mask(token_cooc_counts)


# Example usage and testing
if __name__ == "__main__":
    # Test the PMI Sparse Attention
    vocab_size = 1000
    embed_dim = 256
    num_heads = 4
    seq_len = 32
    batch_size = 2
    top_k = 16
    
    # Create a simple co-occurrence dictionary for testing
    # In practice, this would come from analyzing your training corpus
    token_cooc_counts = {
        (0, 1): 100, (0, 2): 80, (1, 2): 90, (1, 3): 70, (2, 3): 85,
        (4, 5): 120, (4, 6): 95, (5, 6): 110, (5, 7): 75, (6, 7): 100,
        # Add more for a real implementation
    }
    
    # Create model
    model = PMISparseGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=3,
        max_seq_len=64,
        top_k=top_k
    )
    
    # Build PMI mask
    model.build_pmi_mask_from_data(token_cooc_counts)
    
    # Test forward pass
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(token_ids)
    
    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {output.shape}")
    print("PMI Sparse Attention implementation works!")