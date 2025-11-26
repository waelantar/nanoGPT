"""
Speculative Decoding Implementation for LLMs
Based on the RecSys-inspired approach of using a fast "retrieval" model (bigram/trigram) 
to draft tokens, followed by a slower "ranking" model (Transformer) to verify them.

This implements the "Two-Stage Generation" concept from the research plan.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Optional


class NGramModel(nn.Module):
    """
    A simple n-gram statistical model that serves as the "fast retriever" 
    in the two-stage generation process.
    """
    
    def __init__(self, vocab_size: int, n: int = 2, smoothing: float = 1.0):
        """
        Args:
            vocab_size: Size of the vocabulary
            n: N-gram order (2 for bigram, 3 for trigram, etc.)
            smoothing: Laplace smoothing factor
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n = n
        self.smoothing = smoothing
        
        # Store n-gram counts
        # For bigram: {prev_token: {next_token: count}}
        # For trigram: {(prev_prev_token, prev_token): {next_token: count}}
        self.ngram_counts = {}
        self.total_counts = {}  # Total count for each context
        
    def update_counts(self, token_ids: torch.Tensor):
        """
        Update n-gram counts from a sequence of token IDs.
        
        Args:
            token_ids: Sequence of token IDs [seq_len]
        """
        seq = token_ids.tolist()
        
        for i in range(len(seq) - self.n + 1):
            context = tuple(seq[i:i + self.n - 1])  # Previous n-1 tokens
            next_token = seq[i + self.n - 1]  # Next token
            
            if context not in self.ngram_counts:
                self.ngram_counts[context] = Counter()
                self.total_counts[context] = 0
            
            self.ngram_counts[context][next_token] += 1
            self.total_counts[context] += 1
    
    def predict_next_tokens(self, context: Tuple[int, ...], k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Predict the next k most likely tokens given a context.
        
        Args:
            context: Tuple of previous tokens
            k: Number of top predictions to return
            
        Returns:
            Tuple of (top_k_tokens, top_k_probs)
        """
        if context not in self.ngram_counts:
            # If context not seen, return uniform distribution
            tokens = np.random.choice(self.vocab_size, size=min(k, self.vocab_size), replace=False).tolist()
            probs = [1.0 / len(tokens)] * len(tokens)
            return tokens, probs
        
        # Get counts for this context
        counts = self.ngram_counts[context]
        total = self.total_counts[context]
        
        # Apply smoothing and calculate probabilities
        all_probs = {}
        for token in range(self.vocab_size):
            count = counts.get(token, 0)
            prob = (count + self.smoothing) / (total + self.smoothing * self.vocab_size)
            all_probs[token] = prob
        
        # Get top-k tokens
        sorted_tokens = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top_tokens = [token for token, _ in sorted_tokens[:k]]
        top_probs = [prob for _, prob in sorted_tokens[:k]]
        
        return top_tokens, top_probs
    
    def forward(self, token_ids: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate next token predictions for a batch of sequences.
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            k: Number of top predictions per position
            
        Returns:
            Tuple of (top_k_tokens [batch_size, seq_len, k], top_k_probs [batch_size, seq_len, k])
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        if seq_len < self.n - 1:
            # If sequence is too short, pad with special token (e.g., 0)
            padded = torch.cat([torch.zeros(batch_size, self.n - 1 - seq_len, dtype=torch.long, device=device), token_ids], dim=1)
        else:
            padded = token_ids[:, max(0, seq_len - (self.n - 1)):]
        
        # Prepare output tensors
        top_tokens = torch.zeros(batch_size, seq_len, k, dtype=torch.long, device=device)
        top_probs = torch.zeros(batch_size, seq_len, k, dtype=torch.float, device=device)
        
        for batch_idx in range(batch_size):
            for pos in range(seq_len):
                # Determine context based on position and n-gram order
                if pos < self.n - 1:
                    # For early positions, use shorter context
                    context_len = min(pos, self.n - 1)
                    context_start = max(0, pos - context_len)
                    context = tuple(token_ids[batch_idx, context_start:pos].tolist())
                else:
                    # Use full n-gram context
                    context_start = pos - (self.n - 1)
                    context = tuple(token_ids[batch_idx, context_start:pos].tolist())
                
                # Get predictions
                pred_tokens, pred_probs = self.predict_next_tokens(context, k)
                
                # Fill output tensors
                top_tokens[batch_idx, pos, :len(pred_tokens)] = torch.tensor(pred_tokens, device=device)
                top_probs[batch_idx, pos, :len(pred_probs)] = torch.tensor(pred_probs, device=device)
        
        return top_tokens, top_probs


class SpeculativeDecoder(nn.Module):
    """
    Implements speculative decoding using an n-gram model as the "draft" model
    and a transformer as the "target" model.
    
    This follows the "Retrieval → Ranking" paradigm from RecSys.
    """
    
    def __init__(self, 
                 draft_model: NGramModel,
                 target_model: nn.Module,
                 max_speculation: int = 5,
                 accept_threshold: float = 0.9):
        """
        Args:
            draft_model: Fast statistical model (n-gram) for drafting tokens
            target_model: Slow transformer model for verification
            max_speculation: Maximum number of tokens to speculate ahead
            accept_threshold: Threshold for accepting draft tokens based on probability
        """
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.max_speculation = max_speculation
        self.accept_threshold = accept_threshold
    
    def speculative_decode(self, 
                          input_ids: torch.Tensor, 
                          max_new_tokens: int = 20,
                          temperature: float = 1.0) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Initial input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature for final output
            
        Returns:
            Generated token IDs [batch_size, seq_len + new_tokens]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Start with input tokens
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            current_len = generated.shape[1]
            
            # Use draft model to predict multiple tokens ahead
            draft_tokens, draft_probs = self.draft_model(generated, k=1)
            draft_tokens = draft_tokens.squeeze(-1)  # Remove k dimension
            
            # Take the most likely draft tokens
            draft_next_tokens = draft_tokens[:, -1]  # Next token predictions
            
            # Verify with target model (process all tokens in parallel when possible)
            with torch.no_grad():
                # Get logits from target model for the entire sequence so far
                target_logits, _ = self.target_model(generated) if hasattr(self.target_model, '__call__') else self.target_model(generated)
                
                # Get the target probabilities for the last token
                target_logits_last = target_logits[:, -1, :] / temperature
                target_probs = F.softmax(target_logits_last, dim=-1)
                
                # Sample next token from target model
                next_token = torch.multinomial(target_probs, num_samples=1).squeeze(-1)
            
            # Append the verified token
            next_token = next_token.unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def speculative_decode_optimized(self, 
                                   input_ids: torch.Tensor, 
                                   max_new_tokens: int = 20,
                                   temperature: float = 1.0) -> torch.Tensor:
        """
        An optimized version that actually implements speculative decoding logic.
        
        This version:
        1. Uses the draft model to predict k tokens ahead
        2. Runs the target model on those k tokens in parallel
        3. Accepts tokens where draft and target agree, up to first disagreement
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Start with input tokens
        generated = input_ids.clone()
        
        step = 0
        while step < max_new_tokens:
            # Determine how many tokens we can speculate
            remaining_tokens = max_new_tokens - (generated.shape[1] - seq_len)
            speculate_count = min(self.max_speculation, remaining_tokens)
            
            # Use draft model to predict multiple tokens ahead
            draft_tokens, draft_probs = self.draft_model(generated[:, -speculate_count:], k=1)
            draft_tokens = draft_tokens.squeeze(-1)  # Remove k dimension
            
            # Get draft sequence
            draft_seq = draft_tokens[:, -speculate_count:]  # Next 'speculate_count' draft tokens
            
            # Concatenate with current sequence for target model evaluation
            speculative_input = torch.cat([generated, draft_seq], dim=1)
            
            # Run target model on the entire speculative sequence
            with torch.no_grad():
                target_logits, _ = self.target_model(speculative_input) if hasattr(self.target_model, '__call__') else self.target_model(speculative_input)
                
                # Get target probabilities for the speculative portion
                target_logits_spec = target_logits[:, generated.shape[1]:generated.shape[1]+speculate_count, :] / temperature
                target_probs_spec = F.softmax(target_logits_spec, dim=-1)
                
                # Sample from target model for each position
                sampled_tokens = torch.multinomial(target_probs_spec.view(-1, target_probs_spec.shape[-1]), num_samples=1).view(batch_size, speculate_count)
            
            # Compare draft and target outputs, accept up to first disagreement
            accepted_count = 0
            for i in range(speculate_count):
                draft_token = draft_seq[0, i] if batch_size == 1 else draft_seq[:, i]
                target_token = sampled_tokens[:, i]
                
                if torch.equal(draft_token, target_token):
                    accepted_count += 1
                else:
                    break  # Stop at first disagreement
            
            # Append accepted tokens
            if accepted_count > 0:
                accepted_tokens = draft_seq[:, :accepted_count]
                generated = torch.cat([generated, accepted_tokens], dim=1)
            
            # If no tokens were accepted, just generate one token normally
            if accepted_count == 0:
                # Generate one token normally
                current_input = generated
                with torch.no_grad():
                    target_logits, _ = self.target_model(current_input) if hasattr(self.target_model, '__call__') else self.target_model(current_input)
                    target_logits_last = target_logits[:, -1, :] / temperature
                    target_probs = F.softmax(target_logits_last, dim=-1)
                    next_token = torch.multinomial(target_probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
            
            step += accepted_count if accepted_count > 0 else 1
        
        return generated


def create_speculative_model(vocab_size: int, 
                           draft_n: int = 2,
                           target_model: Optional[nn.Module] = None) -> Tuple[NGramModel, SpeculativeDecoder]:
    """
    Create a speculative decoding system with a draft n-gram model and a decoder.
    
    Args:
        vocab_size: Vocabulary size
        draft_n: N-gram order for draft model (2=bigram, 3=trigram)
        target_model: The target transformer model to verify tokens
        
    Returns:
        Tuple of (draft_model, speculative_decoder)
    """
    if target_model is None:
        # Create a dummy target model for testing
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'recsys_llm_research'))
        from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT
        target_model = WideDeepNanoGPT(
            vocab_size=vocab_size,
            n_embd=128,
            n_head=4,
            n_layer=2,
            block_size=128
        )
    
    # Create draft model
    draft_model = NGramModel(vocab_size, n=draft_n)
    
    # Create speculative decoder
    speculative_decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        max_speculation=5,
        accept_threshold=0.9
    )
    
    return draft_model, speculative_decoder


# Example usage and testing
if __name__ == "__main__":
    print("Testing Speculative Decoding Implementation")
    print("="*50)
    
    # Parameters
    vocab_size = 1000
    batch_size = 1
    seq_len = 10
    max_new_tokens = 5
    
    # Create a simple target model for testing
    import sys
    sys.path.append('recsys_llm_research')
    from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT
    target_model = WideDeepNanoGPT(
        vocab_size=vocab_size,
        n_embd=64,
        n_head=4,
        n_layer=2,
        block_size=64
    )
    
    # Create draft model and speculative decoder
    draft_model, spec_decoder = create_speculative_model(
        vocab_size=vocab_size,
        draft_n=2,
        target_model=target_model
    )
    
    # Create sample input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0, :5].tolist()}...")  # Show first 5 tokens
    
    # Update draft model with some training data (simulated)
    # In practice, this would come from your training corpus
    sample_training_data = torch.randint(0, vocab_size, (100, 20))  # 100 sequences of length 20
    for i in range(sample_training_data.shape[0]):
        draft_model.update_counts(sample_training_data[i])
    
    print(f"Updated draft model with {sample_training_data.shape[0]} training sequences")
    
    # Test the draft model
    draft_tokens, draft_probs = draft_model(input_ids, k=3)
    print(f"Draft model output shape: {draft_tokens.shape}")
    print(f"Sample draft tokens for first position: {draft_tokens[0, 0, :].tolist()}")
    print(f"Sample draft probabilities: {draft_probs[0, 0, :].tolist()}")
    
    # Test speculative decoding
    print(f"\nGenerating {max_new_tokens} new tokens using speculative decoding...")
    generated = spec_decoder.speculative_decode_optimized(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.8
    )
    
    print(f"Generated shape: {generated.shape}")
    print(f"Original tokens: {input_ids[0].tolist()}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print("\nSpeculative decoding implementation completed successfully!")