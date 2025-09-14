# DistributedSpeed

A distributed training framework for large-scale machine learning models with memory optimization and parallelization features.

## Features

### Core Capabilities
- **ZeRO Optimizer States**: Zero Redundancy Optimizer with Stages 1, 2, and 3
- **Pipeline Parallelism**: Model pipeline parallelization across devices
- **Gradient Compression**: Gradient compression techniques
- **Memory Optimization**: Memory management and optimization utilities
- **Mixed Precision**: FP16/BF16 training support
- **Dynamic Loss Scaling**: Automatic loss scaling for training stability
- **Activation Checkpointing**: Memory-efficient gradient checkpointing

### ZeRO Optimization
- **Stage 1**: Optimizer state partitioning across data parallel processes
- **Stage 2**: Optimizer state + gradient partitioning  
- **Stage 3**: Optimizer state + gradient + parameter partitioning
- **CPU Offloading**: Offload optimizer states and parameters to CPU memory

### Pipeline Features
- **Model Partitioning**: Distribute model layers across devices
- **Gradient Accumulation**: Micro-batch gradient accumulation
- **Pipeline Scheduling**: 1F1B and interleaved pipeline execution
- **Communication Overlap**: Overlap computation and communication operations

### Communication Features
- **NCCL Backend**: NCCL communication support
- **AllReduce Operations**: Gradient synchronization across processes
- **Compression**: Optional gradient compression
- **Communication Optimization**: Bucket fusion and topology-aware communication

## Requirements

```bash
# Core dependencies
torch>=1.13.0
numpy>=1.21.0
psutil>=5.8.0
packaging>=20.0

# Communication
nccl>=2.10.3
mpi4py>=3.1.0

# Optional dependencies
apex>=0.1           # For fused optimizers
transformer-engine  # For enhanced transformer layers
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/distributedspeed.git
cd distributedspeed

# Install in development mode
pip install -e .

# Or install from PyPI
pip install distributedspeed
```

## Quick Start

### Basic Usage

```python
import torch
import distributedspeed

# Initialize your model
model = YourModel()

# Configure DistributedSpeed
config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "Adam",
        "params": {"lr": 1e-4}
    },
    "zero_optimization": {
        "stage": 2,
        "cpu_offload": False
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000
    }
}

# Initialize engine
engine, optimizer, _, _ = distributedspeed.initialize(
    model=model,
    config=config
)

# Training loop
for batch in dataloader:
    loss = engine(batch)
    engine.backward(loss)
    engine.step()
```

### Configuration File

Create a `ds_config.json` file:

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 1000
    }
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true,
    "cpu_offload": false
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },
  "wall_clock_breakdown": false
}
```

## Configuration

### ZeRO Optimization Stages

#### Stage 1: Optimizer State Partitioning
```python
# Partitions optimizer states across data parallel processes
config = {
    "zero_optimization": {
        "stage": 1
    }
}
```

#### Stage 2: Optimizer + Gradient Partitioning  
```python
# Partitions optimizer states and gradients
config = {
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "reduce_scatter": True
    }
}
```

#### Stage 3: Optimizer + Gradient + Parameter Partitioning
```python
# Partitions optimizer states, gradients, and parameters
config = {
    "zero_optimization": {
        "stage": 3,
        "param_persistence_threshold": 1e6,
        "model_persistence_threshold": 1e6
    }
}
```

### Pipeline Parallelism

```python
# Configure pipeline parallelism
config = {
    "pipeline": {
        "stages": 4,
        "partition": "uniform",
        "seed_layers": False,
        "activation_checkpoint_interval": 1
    }
}

# Initialize with pipeline
engine, optimizer, _, lr_scheduler = distributedspeed.initialize(
    model=model,
    config=config,
    pipeline=True
)
```

## Memory Optimization

### CPU Offloading

```json
{
  "zero_optimization": {
    "stage": 3,
    "cpu_offload": true,
    "cpu_offload_params": true,
    "cpu_offload_use_pin_memory": true
  }
}
```

### Activation Checkpointing

```python
# Enable activation checkpointing
import distributedspeed

def checkpoint_handler(module):
    return isinstance(module, torch.nn.TransformerEncoderLayer)

model = distributedspeed.checkpointing.checkpoint(
    model, 
    checkpoint_handler
)
```

### Memory Monitoring

```python
# Monitor memory usage
def print_memory_stats():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Max GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Built-in memory profiling
engine.memory_profile()
```

## Advanced Configuration

### Custom Optimizers

```python
from distributedspeed.zero import ZeroOptimizer

class CustomOptimizer(ZeroOptimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params)
        self.lr = lr
    
    def step(self):
        # Custom optimization logic
        pass

# Register custom optimizer
distributedspeed.register_optimizer("custom", CustomOptimizer)
```

### Communication Compression

```json
{
  "compression": {
    "enabled": true,
    "compression_type": "fp16",
    "quantization_type": "asymmetric_quantization_signed",
    "quantization_bits": 8,
    "error_feedback": true,
    "all_gather_fp16": true
  }
}
```

## Usage

### Command Line Interface

```bash
# Single node training
python -m distributedspeed.launcher \
    --num_gpus 8 \
    train_script.py \
    --distributedspeed_config ds_config.json

# Multi-node training  
python -m distributedspeed.launcher \
    --num_nodes 4 \
    --num_gpus 8 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_script.py \
    --distributedspeed_config ds_config.json

# SLURM integration
sbatch --nodes=4 --gres=gpu:8 \
    distributedspeed_slurm.sh train_script.py
```

### Configuration Generator

```bash
# Generate configuration file
python -m distributedspeed.config_generator \
    --model_size 6B \
    --num_gpus 32 \
    --batch_size 64 \
    --output ds_config.json
```

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/zero/
python -m pytest tests/pipeline/
python -m pytest tests/communication/
```

### Benchmarks

```bash
# Memory benchmark
python benchmarks/memory_benchmark.py --config ds_config.json

# Communication benchmark  
python benchmarks/comm_benchmark.py --backend nccl

# Training benchmark
python benchmarks/training_benchmark.py --model gpt2 --size large
```

## Examples

### GPT Training

```python
# examples/gpt_training.py
import distributedspeed
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

config = {
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 16,
    "zero_optimization": {"stage": 2},
    "fp16": {"enabled": True}
}

engine, optimizer, _, scheduler = distributedspeed.initialize(
    model=model,
    config=config
)

for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = engine(batch)
        loss = outputs.loss
        engine.backward(loss)
        engine.step()
```

### BERT Fine-tuning

```python
# examples/bert_finetuning.py
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

config = {
    "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 2e-5}
    },
    "zero_optimization": {"stage": 1}
}

engine, optimizer, _, scheduler = distributedspeed.initialize(
    model=model,
    config=config
)
```

## Profiling and Debugging

### Profiling

```python
# Enable profiling
config["profiling"] = {
    "enabled": True,
    "profile_step": 5,
    "module_depth": -1,
    "top_modules": 1,
    "output_path": "./profiling_results"
}

# Custom profiling
with distributedspeed.profiling.profile("custom_section"):
    # Your code here
    pass
```

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
engine.set_tensorboard_writer(writer)

# Logs loss curves, learning rate, memory usage, and communication times
```

### Debugging

```bash
# Debug communication
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Debug memory
export CUDA_LAUNCH_BLOCKING=1
python -m torch.utils.bottleneck train_script.py

# Debug ZeRO
export DISTRIBUTEDSPEED_DEBUG=1
```

## Integration

### Hugging Face Transformers

```python
from transformers import Trainer
from distributedspeed import DistributedSpeedCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[DistributedSpeedCallback()]
)
```

### PyTorch Lightning

```python
import pytorch_lightning as pl
from distributedspeed.lightning import DistributedSpeedStrategy

trainer = pl.Trainer(
    strategy=DistributedSpeedStrategy(config="ds_config.json"),
    devices=8,
    accelerator="gpu"
)
```

## Migration

### From PyTorch DDP

```python
# Before (DDP)
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = torch.optim.Adam(model.parameters())

# After (DistributedSpeed)
engine, optimizer, _, _ = distributedspeed.initialize(
    model=model,
    config=ds_config
)
```

### From FairScale

```python
# Before (FairScale)
from fairscale.optim.oss import OSS
optimizer = OSS(params=model.parameters())

# After (DistributedSpeed)
config = {"zero_optimization": {"stage": 1}}
engine, optimizer, _, _ = distributedspeed.initialize(model, config)
```

## Troubleshooting

### Common Issues

**NCCL Initialization Failed**
```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
```

**CUDA Out of Memory**
```json
{
  "zero_optimization": {
    "stage": 3,
    "cpu_offload": true
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true
  }
}
```

**Slow Training Performance**
```json
{
  "communication": {
    "overlap_comm": true,
    "bucket_cap_mb": 25,
    "overlap_reduce_scatter": true
  }
}
```

### Performance Tuning

1. **Batch Size**: Configure `train_micro_batch_size_per_gpu * gradient_accumulation_steps`
2. **Memory**: Enable appropriate ZeRO stage for your model size and available memory
3. **Communication**: Adjust bucket sizes and enable communication overlap
4. **Checkpointing**: Use activation checkpointing to trade compute for memory
5. **Precision**: Consider FP16/BF16 for compatible hardware

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`  
3. Make your changes and add tests
4. Run the test suite: `python -m pytest tests/`
5. Submit a pull request

### Development Setup

```bash
# Development installation
git clone https://github.com/your-org/distributedspeed.git
cd distributedspeed
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Microsoft DeepSpeed team for research and inspiration
- NVIDIA for NCCL and GPU optimization techniques  
- PyTorch team for the core framework
- HuggingFace for transformer implementations
- Open source ML community

## Support

- Documentation: https://distributedspeed.readthedocs.io
- Issues: https://github.com/your-org/distributedspeed/issues
- Discussions: https://github.com/your-org/distributedspeed/discussions
- Email: support@distributedspeed.ai