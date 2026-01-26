"""
Test script to verify LoRA functionality in ISTS-PLM models.

This script tests:
1. Model creation with and without LoRA
2. Parameter counting
3. Forward pass
4. Backward pass and gradient computation
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.plm4ts import istsplm_forecast
from models.lora_layers import count_lora_parameters


class Args:
    """Mock arguments for testing"""
    def __init__(self, use_lora=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = 35  # PhysioNet has 35 features
        self.d_model = 896
        self.dropout = 0.1
        self.n_te_plmlayer = 2  # Use fewer layers for faster testing
        self.n_st_plmlayer = 2
        self.te_model = 'qwen'
        self.st_model = 'bert'
        self.de_model = 'bert'
        self.semi_freeze = True
        self.no_decoder_plm = True
        self.input_len = 24
        
        # LoRA settings
        self.use_lora = use_lora
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.lora_target_modules = None  # Use default targets


def test_model_creation():
    """Test model creation with and without LoRA"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    # Test without LoRA
    print("\n[1.1] Creating model WITHOUT LoRA...")
    args_no_lora = Args(use_lora=False)
    try:
        model_no_lora = istsplm_forecast(args_no_lora)
        print("✓ Model created successfully without LoRA")
        
        # Count parameters
        total_params = sum(p.numel() for p in model_no_lora.parameters())
        trainable_params = sum(p.numel() for p in model_no_lora.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
    except Exception as e:
        print(f"✗ Failed to create model without LoRA: {e}")
        return False
    
    # Test with LoRA
    print("\n[1.2] Creating model WITH LoRA...")
    args_lora = Args(use_lora=True)
    try:
        model_lora = istsplm_forecast(args_lora)
        print("✓ Model created successfully with LoRA")
        
        # Count parameters including LoRA
        lora_params, total_params, trainable_params = count_lora_parameters(model_lora)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  LoRA parameters: {lora_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Verify LoRA parameters exist
        lora_found = any('lora_' in name for name, _ in model_lora.named_parameters())
        if lora_found:
            print("✓ LoRA parameters found in model")
        else:
            print("✗ No LoRA parameters found in model")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create model with LoRA: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_forward_pass():
    """Test forward pass with LoRA"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    args = Args(use_lora=True)
    
    try:
        model = istsplm_forecast(args).to(args.device)
        model.eval()
        
        # Create dummy data
        B, L, D = 2, 24, 35  # batch_size, seq_len, num_features
        Lp = 12  # prediction length
        
        observed_tp = torch.rand(B, L, D).to(args.device)
        observed_data = torch.rand(B, L, D).to(args.device)
        observed_mask = torch.ones(B, L, D).to(args.device)
        time_steps_to_predict = torch.rand(B, Lp).to(args.device)
        
        print(f"\n[2.1] Running forward pass...")
        print(f"  Input shape: {observed_data.shape}")
        print(f"  Prediction steps: {Lp}")
        
        with torch.no_grad():
            output = model.forecasting(
                time_steps_to_predict,
                observed_data,
                observed_tp,
                observed_mask
            )
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: (1, {B}, {Lp}, {D})")
        
        # Verify output shape
        expected_shape = (1, B, Lp, D)
        if output.shape == expected_shape:
            print("✓ Output shape matches expected")
        else:
            print(f"✗ Output shape mismatch: {output.shape} vs {expected_shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_backward_pass():
    """Test backward pass and gradient computation with LoRA"""
    print("\n" + "="*60)
    print("TEST 3: Backward Pass and Gradients")
    print("="*60)
    
    args = Args(use_lora=True)
    
    try:
        model = istsplm_forecast(args).to(args.device)
        model.train()
        
        # Create dummy data
        B, L, D = 2, 24, 35
        Lp = 12
        
        observed_tp = torch.rand(B, L, D).to(args.device)
        observed_data = torch.rand(B, L, D).to(args.device)
        observed_mask = torch.ones(B, L, D).to(args.device)
        time_steps_to_predict = torch.rand(B, Lp).to(args.device)
        target = torch.rand(1, B, Lp, D).to(args.device)
        
        print(f"\n[3.1] Running backward pass...")
        
        # Forward pass
        output = model.forecasting(
            time_steps_to_predict,
            observed_data,
            observed_tp,
            observed_mask
        )
        
        # Compute loss
        loss = nn.MSELoss()(output, target)
        print(f"  Loss: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        print("✓ Backward pass successful")
        
        # Check LoRA gradients
        print("\n[3.2] Checking LoRA gradients...")
        lora_params_with_grad = 0
        lora_params_total = 0
        
        for name, param in model.named_parameters():
            if 'lora_' in name:
                lora_params_total += 1
                if param.grad is not None:
                    lora_params_with_grad += 1
                    grad_norm = param.grad.norm().item()
                    print(f"  {name}: grad_norm={grad_norm:.6f}")
        
        print(f"\n  LoRA parameters with gradients: {lora_params_with_grad}/{lora_params_total}")
        
        if lora_params_with_grad > 0:
            print("✓ LoRA parameters have gradients")
        else:
            print("✗ No LoRA parameters have gradients")
            return False
            
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ISTS-PLM LoRA Functionality Tests")
    print("="*60)
    
    # Run tests
    test_results = []
    
    test_results.append(("Model Creation", test_model_creation()))
    test_results.append(("Forward Pass", test_forward_pass()))
    test_results.append(("Backward Pass", test_backward_pass()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in test_results)
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
