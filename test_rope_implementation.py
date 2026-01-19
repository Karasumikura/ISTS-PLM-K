#!/usr/bin/env python3
"""
Test script to verify TimeRotaryEmbedding implementation in GPT2Model_wope
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope, TimeRotaryEmbedding
from transformers import GPT2Config


def test_time_rotary_embedding():
    """Test TimeRotaryEmbedding class"""
    print("Testing TimeRotaryEmbedding class...")
    
    dim = 64
    batch_size = 2
    seq_len = 10
    
    rope = TimeRotaryEmbedding(dim)
    time_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    cos, sin = rope(time_ids, seq_len, 'cpu')
    
    assert cos.shape == (batch_size, seq_len, dim), f"Expected shape ({batch_size}, {seq_len}, {dim}), got {cos.shape}"
    assert sin.shape == (batch_size, seq_len, dim), f"Expected shape ({batch_size}, {seq_len}, {dim}), got {sin.shape}"
    
    print("✓ TimeRotaryEmbedding class works correctly")
    return True


def test_gpt2_with_time_ids():
    """Test GPT2Model_wope with time_ids parameter"""
    print("\nTesting GPT2Model_wope with time_ids...")
    
    # Create a small GPT2 config for testing
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,
        n_head=12,
    )
    
    model = GPT2Model_wope(config)
    model.eval()
    
    batch_size = 2
    seq_len = 10
    
    # Create dummy inputs
    inputs_embeds = torch.randn(batch_size, seq_len, config.n_embd)
    time_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Test forward pass with time_ids
    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, time_ids=time_ids)
    
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.n_embd), \
        f"Expected shape ({batch_size}, {seq_len}, {config.n_embd}), got {outputs.last_hidden_state.shape}"
    
    print("✓ GPT2Model_wope works with time_ids parameter")
    return True


def test_gpt2_without_time_ids():
    """Test GPT2Model_wope works without time_ids (backward compatibility)"""
    print("\nTesting GPT2Model_wope without time_ids (backward compatibility)...")
    
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,
        n_head=12,
    )
    
    model = GPT2Model_wope(config)
    model.eval()
    
    batch_size = 2
    seq_len = 10
    
    inputs_embeds = torch.randn(batch_size, seq_len, config.n_embd)
    
    # Test forward pass without time_ids
    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds)
    
    assert outputs.last_hidden_state.shape == (batch_size, seq_len, config.n_embd), \
        f"Expected shape ({batch_size}, {seq_len}, {config.n_embd}), got {outputs.last_hidden_state.shape}"
    
    print("✓ GPT2Model_wope works without time_ids (backward compatible)")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing TimeRotaryEmbedding Implementation")
    print("=" * 60)
    
    try:
        test_time_rotary_embedding()
        test_gpt2_with_time_ids()
        test_gpt2_without_time_ids()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
