#!/usr/bin/env python3
"""
Simpler test to verify the core functionality works
"""

import torch
import torch.nn as nn


# Copy the core classes directly for testing
class TimeRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for 1D time sequences.
    Applies rotation to query and key vectors based on time_ids.
    """
    def __init__(self, dim, max_time_steps=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_time_steps = max_time_steps

    def forward(self, time_ids, seq_len, device):
        """
        Args:
            time_ids: (batch_size, seq_len) tensor of time indices
            seq_len: sequence length
            device: device to create tensors on
        
        Returns:
            cos, sin: (batch_size, seq_len, dim) tensors for rotary embeddings
        """
        # time_ids shape: (batch_size, seq_len)
        # Ensure time_ids is on the same device as inv_freq
        time_ids = time_ids.to(device).float()
        
        # Compute frequencies: (batch_size, seq_len, dim/2)
        freqs = torch.einsum('bs,d->bsd', time_ids, self.inv_freq.to(device))
        
        # Compute embeddings: (batch_size, seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: (batch_size, num_heads, seq_len, head_dim) query tensor
        k: (batch_size, num_heads, seq_len, head_dim) key tensor
        cos: (batch_size, seq_len, head_dim) cosine embeddings
        sin: (batch_size, seq_len, head_dim) sine embeddings
    
    Returns:
        q, k with rotary embeddings applied
    """
    # Reshape cos and sin to match q and k dimensions
    # cos, sin: (batch_size, seq_len, head_dim) -> (batch_size, 1, seq_len, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    # Apply rotation: [cos, cos] * [q] + [-sin, sin] * [q_rotate]
    # where q_rotate is q with adjacent pairs swapped
    # This implements the rotation matrix multiplication
    
    # Create rotated versions by swapping adjacent pairs
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    
    return q_rotated, k_rotated


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
    print(f"  - cos shape: {cos.shape}")
    print(f"  - sin shape: {sin.shape}")
    return True


def test_apply_rotary_pos_emb():
    """Test apply_rotary_pos_emb function"""
    print("\nTesting apply_rotary_pos_emb...")
    
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64
    
    # Create dummy q and k
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Create dummy cos and sin
    rope = TimeRotaryEmbedding(head_dim)
    time_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    cos, sin = rope(time_ids, seq_len, 'cpu')
    
    # Apply rotary embeddings
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_rotated.shape == q.shape, f"Expected q shape {q.shape}, got {q_rotated.shape}"
    assert k_rotated.shape == k.shape, f"Expected k shape {k.shape}, got {k_rotated.shape}"
    
    print("✓ apply_rotary_pos_emb works correctly")
    print(f"  - q_rotated shape: {q_rotated.shape}")
    print(f"  - k_rotated shape: {k_rotated.shape}")
    return True


def test_time_ids_generation():
    """Test time_ids generation like in plm4ts.py"""
    print("\nTesting time_ids generation for plm4ts.py...")
    
    B, L, D = 2, 10, 5
    
    # Simulate observed_tp
    observed_tp = torch.randn(B, L, D)
    
    # Generate time_ids as in plm4ts.py
    observed_tp_reshaped = observed_tp.permute(0, 2, 1).reshape(B*D, L)  # (B*D, L)
    time_ids = torch.cat([torch.zeros_like(observed_tp_reshaped[:, :1]), observed_tp_reshaped], dim=1)  # (B*D, L+1)
    
    expected_shape = (B*D, L+1)
    assert time_ids.shape == expected_shape, f"Expected shape {expected_shape}, got {time_ids.shape}"
    
    print("✓ time_ids generation works correctly")
    print(f"  - time_ids shape: {time_ids.shape}")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing TimeRotaryEmbedding Implementation (Core)")
    print("=" * 60)
    
    try:
        test_time_rotary_embedding()
        test_apply_rotary_pos_emb()
        test_time_ids_generation()
        
        print("\n" + "=" * 60)
        print("All core tests passed! ✓")
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
