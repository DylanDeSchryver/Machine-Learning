# Advanced MoE Training Guide

This guide will help you train the **improved Mixture of Experts (MoE) model** on your powerful GPU server.

## 🎯 Model Overview

### Advanced MoE Model Specifications

| Feature | Base MoE | **Advanced MoE** | Improvement |
|---------|----------|------------------|-------------|
| Total Parameters | 500M | **800M** | +60% |
| Active Parameters | 100M | **150M** | +50% |
| Hidden Size | 1024 | **1280** | +25% |
| Num Layers | 16 | **24** | +50% |
| Num Experts | 8 | **16** | +100% |
| MoE Frequency | Every 2 layers | **Every layer** | 2x more MoE |
| Activation | GELU | **SwiGLU** | Better performance |
| Normalization | LayerNorm | **RMSNorm** | Faster |

### Key Improvements

1. **More Experts (16 vs 8)**: Better specialization and capacity
2. **All Layers Use MoE**: Maximum sparse computation efficiency
3. **Larger Model**: 800M total parameters for better quality
4. **Modern Architecture**: SwiGLU activation + RMSNorm
5. **Optimized for A100/H100**: Takes advantage of modern GPU features

---

## 🚀 Quick Start

### Step 1: Verify GPU Setup

```bash
# Run the GPU verification script
python scripts/check_gpu_setup.py
```

This will:
- ✅ Detect all available GPUs
- ✅ Check memory and compute capability
- ✅ Test GPU operations
- ✅ Provide recommendations for your hardware

**Expected Output:**
```
✓ Your GPU setup is ready for training!
✓ All checks passed successfully

Primary GPU: NVIDIA A100-SXM4-40GB
Memory: 40 GB
Recommended: Use moe_advanced.yaml with batch_size=8
```

### Step 2: Prepare Data

```bash
# Download datasets (if not already done)
storyteller-download --output_dir data/raw

# Preprocess data
storyteller-preprocess --input_dir data/raw --output_dir data/processed

# Train tokenizer
storyteller-tokenizer --input_file data/processed/train.txt --output_dir data/tokenizers
```

### Step 3: Start MLflow (Optional but Recommended)

```bash
# In a separate terminal
mlflow ui --port 8080
```

Then open http://localhost:8080 in your browser to monitor training.

### Step 4: Train the Model

**Option A: Using Jupyter Notebook (Recommended for GPU Server)**

```bash
# Start Jupyter
jupyter notebook

# Open: train_moe_advanced.ipynb
# Run all cells
```

**Option B: Using Command Line**

```bash
# Train directly
storyteller-train --config configs/moe_advanced.yaml
```

---

## 📊 Training Configuration

### Hardware Requirements

| GPU Model | Memory | Batch Size | Training Time | Status |
|-----------|--------|------------|---------------|--------|
| A100 80GB | 80GB | 16 | 1-2 days | ✅ Excellent |
| A100 40GB | 40GB | 8 | 2-3 days | ✅ Recommended |
| V100 32GB | 32GB | 4 | 4-5 days | ⚠️ Marginal |
| RTX 3090 | 24GB | 2-4 | 5-7 days | ⚠️ Tight fit |

### Configuration File: `configs/moe_advanced.yaml`

Key settings:
```yaml
model:
  hidden_size: 1280
  num_layers: 24
  num_experts: 16
  activation: "swiglu"
  norm_type: "rmsnorm"

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1.5e-4
  num_epochs: 10
  amp_dtype: "bfloat16"
```

### Adjusting for Your Hardware

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size**:
   ```yaml
   batch_size: 4  # or even 2
   ```

2. **Increase gradient accumulation**:
   ```yaml
   gradient_accumulation_steps: 8  # to maintain effective batch size
   ```

3. **Enable gradient checkpointing** (already enabled):
   ```yaml
   gradient_checkpointing: true
   ```

4. **Use float16 instead of bfloat16** (if needed):
   ```yaml
   amp_dtype: "float16"
   ```

---

## 📈 Monitoring Training

### 1. GPU Usage

Monitor GPU utilization in real-time:

```bash
# In a separate terminal
nvidia-smi -l 1
```

**What to look for:**
- GPU Utilization: Should be 90-100%
- Memory Usage: Should be 70-90% of total
- Temperature: Should be < 85°C

### 2. MLflow Dashboard

Open http://localhost:8080 to see:
- Training loss curves
- Validation perplexity
- Learning rate schedule
- Expert utilization statistics
- System metrics (GPU, CPU, memory)

### 3. Jupyter Notebook Output

The notebook will show:
- Real-time training progress
- Loss values per batch
- Evaluation metrics
- Generated story samples

### 4. Expected Metrics

**Good Training Progress:**
```
Epoch 1:
  Train Loss: 4.5 → 3.2
  Val Loss: 3.8
  Val Perplexity: 44.7

Epoch 5:
  Train Loss: 2.1 → 1.9
  Val Loss: 2.3
  Val Perplexity: 9.97

Epoch 10:
  Train Loss: 1.5 → 1.4
  Val Loss: 1.8
  Val Perplexity: 6.05
```

---

## 🎨 Generating Stories

After training completes, generate stories:

### Using the Notebook

The notebook includes a generation cell at the end. Run it to generate sample stories.

### Using Command Line

```bash
# Interactive mode
storyteller-generate \
  --checkpoint checkpoints/moe_advanced/best_model.pt \
  --interactive

# Single prompt
storyteller-generate \
  --checkpoint checkpoints/moe_advanced/best_model.pt \
  --prompt "Once upon a time in a magical forest" \
  --max_length 1024 \
  --temperature 0.85

# Batch generation
storyteller-generate \
  --checkpoint checkpoints/moe_advanced/best_model.pt \
  --prompts_file prompts.txt \
  --output generated_stories.txt
```

### Generation Parameters

Adjust these for different story styles:

```python
# Creative and diverse
temperature = 0.9
top_p = 0.95
repetition_penalty = 1.1

# More focused and coherent
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.2

# Very deterministic
temperature = 0.5
top_p = 0.8
repetition_penalty = 1.3
```

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```yaml
batch_size: 4  # or 2
gradient_accumulation_steps: 8  # to compensate
```

**Solution 2**: Clear GPU cache
```python
import torch
torch.cuda.empty_cache()
```

**Solution 3**: Use gradient checkpointing (already enabled)

### Issue: Training is Slow

**Check 1**: Verify GPU is being used
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.current_device())  # Should show GPU 0
```

**Check 2**: Enable TF32 (for Ampere+ GPUs)
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Check 3**: Increase num_workers
```yaml
num_workers: 8  # or more
```

### Issue: Loss is Not Decreasing

**Check 1**: Verify data is loading correctly
```python
# In notebook, check a batch
batch = next(iter(train_dataloader))
print(batch['input_ids'].shape)
print(tokenizer.decode(batch['input_ids'][0]))
```

**Check 2**: Check learning rate
```python
# Should start small and increase during warmup
print(optimizer.param_groups[0]['lr'])
```

**Check 3**: Monitor expert utilization
```python
# In MLflow, check moe/layer_*_balance metrics
# Should be close to 1.0 (balanced)
```

### Issue: Expert Imbalance

If some experts are underutilized:

**Solution**: Increase load balancing weight
```yaml
load_balancing_loss_weight: 0.02  # increase from 0.01
```

---

## 📝 Checkpoints and Model Saving

### Checkpoint Structure

```
checkpoints/moe_advanced/
├── best_model.pt              # Best validation loss
├── final_model.pt             # Last epoch
├── checkpoint_step_2500.pt    # Periodic checkpoints
├── checkpoint_step_5000.pt
└── ...
```

### Resuming Training

```bash
# Resume from checkpoint
storyteller-train \
  --config configs/moe_advanced.yaml \
  --resume checkpoints/moe_advanced/checkpoint_step_5000.pt
```

### Loading for Inference

```python
import torch
from storyteller.model import StorytellerModel, ModelConfig

# Load checkpoint
checkpoint = torch.load('checkpoints/moe_advanced/best_model.pt')

# Create model
config = ModelConfig(**checkpoint['config'])
model = StorytellerModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 🎓 Understanding the Model

### MoE Architecture

```
Input Tokens
    ↓
Token Embeddings + RoPE
    ↓
┌─────────────────────────────┐
│  Transformer Block 1 (MoE)  │
│  ├─ Multi-Head Attention    │
│  └─ MoE Layer (16 experts)  │
│     ├─ Router (Top-2)       │
│     ├─ Expert 0-15          │
│     └─ Weighted Combine     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│  Transformer Block 2 (MoE)  │
│  ...                        │
└─────────────────────────────┘
    ↓
    ... (24 layers total)
    ↓
Language Modeling Head
    ↓
Output Logits
```

### Expert Specialization

After training, experts may specialize in:
- Different writing styles
- Different topics (fantasy, sci-fi, etc.)
- Different narrative structures
- Different vocabulary domains

You can analyze this using the expert routing statistics in MLflow.

---

## 📊 Expected Results

### Training Metrics

After 10 epochs on high-quality story data:
- **Final Train Loss**: ~1.4-1.6
- **Final Val Loss**: ~1.8-2.0
- **Final Perplexity**: ~6-8
- **Expert Balance**: 0.9-1.1 (good)
- **Routing Entropy**: 2.0-2.5 (diverse)

### Story Quality

Generated stories should exhibit:
- ✅ Coherent narrative structure
- ✅ Consistent characters and settings
- ✅ Creative plot development
- ✅ Proper grammar and punctuation
- ✅ Varied vocabulary
- ✅ Minimal repetition

---

## 🚀 Next Steps

After successful training:

1. **Evaluate Quality**: Generate 100+ stories and manually review
2. **Compare Models**: Train base MoE for comparison
3. **Experiment**: Try different generation parameters
4. **Fine-tune**: Fine-tune on specific story genres
5. **Deploy**: Export model for production use

---

## 📚 Additional Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **PyTorch AMP Guide**: https://pytorch.org/docs/stable/amp.html
- **MoE Paper**: https://arxiv.org/abs/1701.06538
- **Switch Transformers**: https://arxiv.org/abs/2101.03961

---

## ❓ Need Help?

If you encounter issues:

1. Run `python scripts/check_gpu_setup.py` to verify hardware
2. Check MLflow logs for detailed metrics
3. Review the troubleshooting section above
4. Check GPU memory with `nvidia-smi`

---

**Good luck with your training! 🎉**

Your improved MoE model should generate high-quality, creative stories after training.
