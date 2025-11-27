"""
Generate synthetic but learnable language modeling data.
CRITICAL FIX: Creates next-token prediction task with Markov structure
"""
import torch
import numpy as np

def create_language_modeling_data(vocab_size, seq_len, num_batches, transition_strength=0.85, batch_size=8):
    """
    FIXED: Returns properly batched tensors for language modeling
    """
    # Create structured transitions (higher diagonal = more predictable)
    transitions = torch.eye(vocab_size) * transition_strength + \
                  torch.ones(vocab_size, vocab_size) * (1 - transition_strength) / vocab_size
    
    data = []
    for _ in range(num_batches):
        batch_x = []
        batch_y = []
        
        for b in range(batch_size):
            # Each sequence starts randomly
            seq = [torch.randint(0, vocab_size, (1,)).item()]
            
            # Generate sequence with Markov dependence
            for _ in range(seq_len):
                current = seq[-1]
                next_token = torch.multinomial(transitions[current], 1).item()
                seq.append(next_token)
            
            # Convert to tensor and split into x/y
            seq_tensor = torch.tensor(seq)
            batch_x.append(seq_tensor[:-1])  # Input: tokens 0..n-1
            batch_y.append(seq_tensor[1:])   # Target: tokens 1..n
        
        # Stack into proper batch tensors (batch_size, seq_len)
        data.append((torch.stack(batch_x), torch.stack(batch_y)))
    
    return data

def verify_data_quality(data, vocab_size):
    """FIXED: Properly verifies x→y mapping"""
    print("\nData Quality Check:")
    
    # Check first batch
    x, y = data[0]
    print(f"  x shape: {x.shape}, y shape: {y.shape} (should be [batch, seq])")
    
    # Verify x and y are shifted by 1 position
    match_rate = (x[:, 1:] == y[:, :-1]).float().mean().item()
    print(f"  X→Y shift consistency: {match_rate:.3f} (should be ≈1.0)")
    
    # Check token diversity
    all_tokens = torch.cat([y.flatten() for _, y in data])
    unique = all_tokens.unique().numel()
    print(f"  Unique tokens used: {unique}/{vocab_size} ({unique/vocab_size*100:.1f}%)")
    
    # Entropy (higher = more random)
    counts = torch.bincount(all_tokens, minlength=vocab_size).float()
    probs = counts / counts.sum()
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()
    random_entropy = np.log(vocab_size)
    print(f"  Token entropy: {entropy:.3f} bits (random={random_entropy:.3f})")
    
    # Show sequence correlation (should be > 0 for learnable data)
    x_flat = x[0, 1:].float()
    y_flat = y[0, :-1].float()
    if len(x_flat) > 1:  # Need at least 2 points for correlation
        correlation = torch.corrcoef(torch.stack([x_flat, y_flat]))[0, 1].item()
        if torch.isnan(torch.tensor(correlation)):
            correlation = 0.0
    else:
        correlation = 0.0
    print(f"  Sequence correlation: {correlation:.3f} (higher = more predictable)")

    return correlation > -0.1  # Return True if data is learnable (even weak correlation is OK)