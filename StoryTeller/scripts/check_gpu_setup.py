"""
GPU Setup Verification Script

Run this before training to ensure your GPUs are properly configured
and will be utilized during training.
"""

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_pytorch_cuda():
    """Check PyTorch CUDA installation."""
    print("=" * 70)
    print("PYTORCH & CUDA CONFIGURATION")
    print("=" * 70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        return True
    else:
        print("\n⚠️  WARNING: CUDA is not available!")
        print("Training will run on CPU and be EXTREMELY slow.")
        return False


def check_gpus():
    """Check available GPUs and their specifications."""
    if not torch.cuda.is_available():
        return False
    
    print("\n" + "=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)
    
    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of GPUs detected: {num_gpus}")
    
    all_suitable = True
    
    for i in range(num_gpus):
        print(f"\n{'─' * 70}")
        print(f"GPU {i}:")
        print(f"{'─' * 70}")
        
        # Get device properties
        props = torch.cuda.get_device_properties(i)
        gpu_name = torch.cuda.get_device_name(i)
        total_memory = props.total_memory / 1e9
        
        print(f"  Name: {gpu_name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {total_memory:.2f} GB")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")
        
        # Check current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        free = total_memory - reserved
        
        print(f"\n  Memory Status:")
        print(f"    Allocated: {allocated:.2f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"    Reserved:  {reserved:.2f} GB ({reserved/total_memory*100:.1f}%)")
        print(f"    Free:      {free:.2f} GB ({free/total_memory*100:.1f}%)")
        
        # Check if suitable for training
        print(f"\n  Suitability Check:")
        
        # Check memory
        if total_memory >= 40:
            print(f"    ✓ Memory: Excellent ({total_memory:.0f}GB >= 40GB)")
        elif total_memory >= 24:
            print(f"    ✓ Memory: Good ({total_memory:.0f}GB >= 24GB)")
        elif total_memory >= 16:
            print(f"    ⚠ Memory: Marginal ({total_memory:.0f}GB >= 16GB)")
            print(f"      Consider reducing batch size or model size")
            all_suitable = False
        else:
            print(f"    ✗ Memory: Insufficient ({total_memory:.0f}GB < 16GB)")
            print(f"      This GPU cannot train the advanced MoE model")
            all_suitable = False
        
        # Check compute capability
        compute_cap = props.major * 10 + props.minor
        if compute_cap >= 80:  # Ampere or newer (A100, H100)
            print(f"    ✓ Compute: Excellent (Ampere+ architecture)")
            print(f"      TF32 and BF16 supported for faster training")
        elif compute_cap >= 70:  # Volta (V100)
            print(f"    ✓ Compute: Good (Volta architecture)")
        elif compute_cap >= 60:  # Pascal (P100)
            print(f"    ⚠ Compute: Older architecture")
            print(f"      Training will be slower than newer GPUs")
        else:
            print(f"    ✗ Compute: Too old (CC {props.major}.{props.minor})")
            all_suitable = False
        
        # Check if GPU is busy
        if reserved > total_memory * 0.5:
            print(f"    ⚠ GPU is currently busy ({reserved/total_memory*100:.0f}% reserved)")
            print(f"      Consider using a different GPU or waiting")
    
    return all_suitable


def test_gpu_operations():
    """Test basic GPU operations."""
    if not torch.cuda.is_available():
        return False
    
    print("\n" + "=" * 70)
    print("GPU OPERATIONS TEST")
    print("=" * 70)
    
    try:
        # Test tensor operations
        print("\nTesting tensor operations...")
        device = torch.device('cuda:0')
        
        # Create tensors
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        c = torch.matmul(a, b)
        
        print("  ✓ Tensor creation: OK")
        print("  ✓ Matrix multiplication: OK")
        
        # Test mixed precision
        print("\nTesting mixed precision (AMP)...")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            d = torch.matmul(a, b)
        print("  ✓ BFloat16 operations: OK")
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            e = torch.matmul(a, b)
        print("  ✓ Float16 operations: OK")
        
        # Test memory management
        print("\nTesting memory management...")
        torch.cuda.empty_cache()
        print("  ✓ Cache clearing: OK")
        
        print("\n✓ All GPU operations working correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ GPU operations failed: {e}")
        return False


def check_recommended_settings():
    """Check and display recommended settings."""
    print("\n" + "=" * 70)
    print("RECOMMENDED SETTINGS FOR TRAINING")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("\n⚠️  No GPUs available - cannot provide recommendations")
        return
    
    # Get primary GPU info
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1e9
    compute_cap = props.major * 10 + props.minor
    
    print("\nBased on your GPU configuration:")
    print(f"  Primary GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {total_memory:.0f} GB")
    
    # Recommend configuration
    if total_memory >= 80:
        print("\n✓ EXCELLENT GPU - Use moe_advanced.yaml")
        print("  Recommended settings:")
        print("    - batch_size: 16")
        print("    - gradient_accumulation_steps: 2")
        print("    - amp_dtype: bfloat16")
        print("    - Expected training time: 1-2 days")
    elif total_memory >= 40:
        print("\n✓ GREAT GPU - Use moe_advanced.yaml")
        print("  Recommended settings:")
        print("    - batch_size: 8")
        print("    - gradient_accumulation_steps: 4")
        print("    - amp_dtype: bfloat16")
        print("    - Expected training time: 2-3 days")
    elif total_memory >= 24:
        print("\n✓ GOOD GPU - Use moe_model.yaml or moe_advanced.yaml")
        print("  Recommended settings:")
        print("    - batch_size: 4-6")
        print("    - gradient_accumulation_steps: 6-8")
        print("    - amp_dtype: bfloat16 or float16")
        print("    - gradient_checkpointing: true")
        print("    - Expected training time: 3-4 days")
    else:
        print("\n⚠ LIMITED GPU - Use gpt2_small.yaml instead")
        print("  The advanced MoE model may not fit in memory")
        print("  Recommended settings:")
        print("    - Use configs/gpt2_small.yaml")
        print("    - batch_size: 4")
        print("    - gradient_accumulation_steps: 8")
    
    # TF32 recommendation
    if compute_cap >= 80:
        print("\n✓ TF32 Support Available")
        print("  Add to your training script:")
        print("    torch.backends.cuda.matmul.allow_tf32 = True")
        print("    torch.backends.cudnn.allow_tf32 = True")
    
    print("\n" + "=" * 70)


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "GPU SETUP VERIFICATION" + " " * 26 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Run checks
    cuda_ok = check_pytorch_cuda()
    
    if cuda_ok:
        gpus_ok = check_gpus()
        ops_ok = test_gpu_operations()
        check_recommended_settings()
        
        # Final summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        if gpus_ok and ops_ok:
            print("\n✓ Your GPU setup is ready for training!")
            print("✓ All checks passed successfully")
            print("\nNext steps:")
            print("  1. Open train_moe_advanced.ipynb in Jupyter")
            print("  2. Run all cells to start training")
            print("  3. Monitor GPU usage with: nvidia-smi -l 1")
            print("  4. Monitor training in MLflow UI: http://localhost:8080")
        else:
            print("\n⚠ Some issues detected with your GPU setup")
            print("Review the warnings above before starting training")
    else:
        print("\n✗ CUDA is not available")
        print("Please install CUDA and PyTorch with CUDA support")
        print("\nInstallation guide:")
        print("  https://pytorch.org/get-started/locally/")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
