"""
Generate synthetic but learnable language modeling data.
CRITICAL FIX: Creates next-token prediction task with Markov structure
"""
import torch
import numpy as np

def create_language_modeling_data(vocab_size, seq_len, num_batches, transition_strength=0.7):
    """
    Generate synthetic data with controlled statistical structure.
    transition_strength: 0.0=uniform random, 1.0=fully deterministic
    """
    # Create stochastic transition matrix with less deterministic behavior
    base = torch.ones(vocab_size, vocab_size) / vocab_size  # Uniform base
    identity = torch.eye(vocab_size)  # Self-transition
    # Blend base and identity with transition_strength
    transitions = (1 - transition_strength) * base + transition_strength * identity
    transitions = transitions / transitions.sum(dim=1, keepdim=True)  # Normalize rows
    
    data = []
    for _ in range(num_batches):
        batch = []
        # Start with random token
        current = torch.randint(0, vocab_size, (1,))
        
        for _ in range(seq_len + 1):  # +1 for shifted target
            batch.append(current.item())
            # Sample next token based on transition probabilities
            current = torch.multinomial(transitions[current.item()], 1)
        
        seq = torch.tensor(batch)
        # CRITICAL: Create next-token prediction pairs
        x = seq[:-1]  # Input: tokens 0..n-1
        y = seq[1:]   # Target: tokens 1..n
        
        data.append((x, y))
    
    return data

def verify_data_quality(data, vocab_size, sample_batches=5):
    """Verify data has learnable structure"""
    print("\nData Quality Check:")
    for i in range(min(sample_batches, len(data))):
        x, y = data[i]
        # Calculate next-token accuracy - this should be meaningful since y is x shifted by 1
        next_token_acc = (x[1:] == y[:-1]).float().mean().item()
        print(f"  Batch {i}: Next-token match={next_token_acc:.3f} (should be <1.0 for learnable task)")
        
        # Check if x and y are different (they should be for next-token prediction)
        diff_ratio = (x != y).float().mean().item()
        print(f"    X!=Y ratio: {diff_ratio:.3f} (should be high for next-token task)")
    
    # Check entropy
    all_tokens = torch.cat([y for _, y in data])
    token_counts = torch.bincount(all_tokens, minlength=vocab_size).float()
    token_probs = token_counts / token_counts.sum()
    entropy = -(token_probs * token_probs.log()).sum().item()
    entropy = entropy if not torch.isnan(torch.tensor(entropy)) else 0.0
    print(f"  Token entropy: {entropy:.3f} bits (random={np.log2(vocab_size):.3f})")
    
    # Additional check: verify the relationship between x and y
    x0, y0 = data[0]
    print(f"  First sequence: x[:10] = {x0[:10]}")
    print(f"  First sequence: y[:10] = {y0[:10]}")
    print(f"  X shifted by 1: {x0[1:11]} (should match y[:10] if copy)")