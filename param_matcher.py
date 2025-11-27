"""
Automatically match model parameters to target count
"""
from model import GPT, GPTConfig
from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT
from efficient_pmi_sparse_attention import create_efficient_pmi_model

TARGET_PARAMS = 731_264
VOCAB_SIZE = 1000
SEQ_LEN = 64

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def find_wide_deep_config(target_params=TARGET_PARAMS, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Find n_embd that gives exact parameter count"""
    for n_embd in range(64, 256, 4):  # Search space
        model = WideDeepNanoGPT(vocab_size, n_embd, 4, 3, seq_len)
        params = count_params(model)
        if abs(params - target_params) / target_params < 0.02:  # Within 2%
            return n_embd, params
    return 96, count_params(WideDeepNanoGPT(vocab_size, 96, 4, 3, seq_len))

def find_pmi_sparse_config(target_params=TARGET_PARAMS, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
    """Find n_layer that gives exact parameter count"""
    for n_layer in [2, 3, 4]:
        model = create_efficient_pmi_model(vocab_size, 128, 4, n_layer, seq_len, 32)
        params = count_params(model)
        if abs(params - target_params) / target_params < 0.02:
            return n_layer, params
    return 2, count_params(create_efficient_pmi_model(vocab_size, 128, 4, 2, seq_len, 32))

if __name__ == "__main__":
    print("Finding parameter-matched configurations...")
    
    n_embd, wd_params = find_wide_deep_config()
    print(f"Wide & Deep: n_embd={n_embd} → {wd_params:,} params ({wd_params/TARGET_PARAMS:.3f}x)")
    
    n_layer, pmi_params = find_pmi_sparse_config()
    print(f"PMI Sparse: n_layer={n_layer} → {pmi_params:,} params ({pmi_params/TARGET_PARAMS:.3f}x)")