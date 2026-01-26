"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

This module provides LoRA layers that can be applied to linear layers in pre-trained models.
LoRA reduces the number of trainable parameters by using low-rank decomposition.

Reference: LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class LoRALayer(nn.Module):
    """
    Base LoRA layer that adds low-rank adaptation to existing layers.
    
    LoRA decomposes the weight update ΔW into two low-rank matrices:
    ΔW = B @ A, where A ∈ R^(r×d), B ∈ R^(d×r), and r << d
    
    The final output is: h = W₀x + (B @ A)x, where W₀ is frozen.
    """
    
    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        """
        Args:
            r: Rank of the low-rank decomposition
            lora_alpha: Scaling factor for LoRA (α/r)
            lora_dropout: Dropout probability for LoRA layers
        """
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x


class LinearLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This wraps an existing Linear layer and adds LoRA matrices A and B.
    The original layer weights are frozen, and only A and B are trained.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """
        Args:
            original_layer: The original linear layer to adapt
            r: Rank of LoRA
            lora_alpha: Scaling factor
            lora_dropout: Dropout probability
            merge_weights: If True, merge LoRA weights into original weights during inference
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merge_weights = merge_weights
        self.merged = False
        
        # Get dimensions from original layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Freeze original layer
        original_layer.weight.requires_grad = False
        if original_layer.bias is not None:
            original_layer.bias.requires_grad = False
        
        # LoRA matrices
        # A: (r, in_features) - initialized with random Gaussian
        # B: (out_features, r) - initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize A with random Gaussian distribution
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        # B is kept at zero so LoRA starts with no effect (ΔW = 0)
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Original linear transformation
        result = self.original_layer(x)
        
        if not self.merged:
            # Apply LoRA: x @ A^T @ B^T * scaling
            # This is equivalent to: x @ (B @ A)^T * scaling
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
            result = result + lora_out
        
        return result
    
    def merge_lora_weights(self):
        """
        Merge LoRA weights into the original layer for inference.
        This eliminates the computational overhead of LoRA during inference.
        """
        if not self.merged:
            # Compute ΔW = B @ A * scaling
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            # Add to original weights
            self.original_layer.weight.data += delta_W
            self.merged = True
    
    def unmerge_lora_weights(self):
        """
        Unmerge LoRA weights from the original layer.
        Useful if you want to continue training after merging.
        """
        if self.merged:
            # Compute ΔW = B @ A * scaling
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            # Subtract from original weights
            self.original_layer.weight.data -= delta_W
            self.merged = False
    
    def extra_repr(self) -> str:
        return f'in_features={self.original_layer.in_features}, out_features={self.original_layer.out_features}, r={self.r}, lora_alpha={self.lora_alpha}'


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """
    Apply LoRA to specified modules in a model.
    
    Args:
        model: The model to apply LoRA to
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to common attention modules.
                       Examples: ['q_proj', 'v_proj'], ['query', 'value', 'key']
        r: Rank of LoRA
        lora_alpha: Scaling factor
        lora_dropout: Dropout probability
        
    Returns:
        The modified model with LoRA layers
    """
    # Default target modules for common architectures
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    # Track which modules were modified
    modified_modules = []
    
    def apply_lora_recursive(module: nn.Module, prefix: str = ''):
        """Recursively apply LoRA to matching modules."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this module should get LoRA
            if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
                # Replace with LoRA version
                lora_layer = LinearLoRA(
                    original_layer=child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(module, name, lora_layer)
                modified_modules.append(full_name)
            else:
                # Recurse into child modules
                apply_lora_recursive(child, full_name)
    
    apply_lora_recursive(model)
    
    print(f"Applied LoRA to {len(modified_modules)} modules:")
    for name in modified_modules[:10]:  # Show first 10
        print(f"  - {name}")
    if len(modified_modules) > 10:
        print(f"  ... and {len(modified_modules) - 10} more modules")
    
    return model


def count_lora_parameters(model: nn.Module) -> tuple:
    """
    Count the number of LoRA parameters and total parameters in a model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Tuple of (lora_params, total_params, trainable_params)
    """
    lora_params = 0
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()
    
    return lora_params, total_params, trainable_params


def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Extract only the LoRA parameters from a model's state dict.
    
    This is useful for saving only the LoRA weights, which are much smaller
    than the full model weights.
    
    Args:
        model: The model with LoRA layers
        
    Returns:
        Dictionary containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state_dict[name] = param.data.clone()
    
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, lora_state_dict: dict):
    """
    Load LoRA parameters into a model.
    
    Args:
        model: The model with LoRA layers
        lora_state_dict: Dictionary containing LoRA parameters
    """
    model_state = model.state_dict()
    
    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        else:
            print(f"Warning: LoRA parameter {name} not found in model")
    
    print(f"Loaded {len(lora_state_dict)} LoRA parameters")
