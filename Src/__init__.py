"""
DistributedSpeed: High-performance distributed training framework for large-scale machine learning models.

This package provides advanced memory optimization and parallelization strategies including:
- ZeRO optimizer states (Stages 1, 2, 3)  
- Pipeline parallelism
- Gradient compression
- Memory optimization
- Mixed precision training
- Activation checkpointing

Usage:
    import distributedspeed
    
    # Initialize engine
    engine, optimizer, _, scheduler = distributedspeed.initialize(
        model=model,
        config=config_dict
    )
    
    # Training loop
    for batch in dataloader:
        loss = engine(batch)
        engine.backward(loss)
        engine.step()

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import warnings
import logging
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path

import torch
import torch.distributed as dist
from packaging import version

# Version information
__version__ = "0.12.6"
__author__ = "DistributedSpeed Contributors"
__email__ = "support@distributedspeed.ai"

# Compatibility checks
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
MIN_TORCH_VERSION = "1.13.0"

if version.parse(torch.__version__) < version.parse(MIN_TORCH_VERSION):
    raise ImportError(f"DistributedSpeed requires PyTorch >= {MIN_TORCH_VERSION}, "
                     f"but found {torch.__version__}")

# Environment setup
def _setup_environment():
    """Setup DistributedSpeed environment variables and logging."""
    
    # Set default environment variables
    env_vars = {
        'DISTRIBUTEDSPEED_DEBUG': '0',
        'DISTRIBUTEDSPEED_LOG_LEVEL': 'INFO', 
        'DISTRIBUTEDSPEED_DISABLE_CUDA_GRAPH': '0',
        'DISTRIBUTEDSPEED_AUTOTUNING': '0',
        'NCCL_ASYNC_ERROR_HANDLING': '1',
        'CUDA_DEVICE_MAX_CONNECTIONS': '1'
    }
    
    for key, default_value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
    
    # Configure logging
    log_level = getattr(logging, os.environ.get('DISTRIBUTEDSPEED_LOG_LEVEL', 'INFO'))
    logging.basicConfig(
        level=log_level,
        format='[DistributedSpeed] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress some warnings in distributed training
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")

# Initialize environment
_setup_environment()

# Import core modules
from .engine import DistributedSpeedEngine
from .config import DistributedSpeedConfig, load_config
from .zero import ZeROOptimizer, ZeROConfig
from .pipeline import PipelineEngine, PipelineConfig
from .communication import CommManager, init_distributed
from .memory import MemoryManager, ActivationCheckpointing

# Import utilities
from .utils import (
    get_world_size,
    get_rank, 
    get_local_rank,
    is_initialized,
    barrier,
    print_rank_0,
    log_dist
)

# Import optimizers and schedulers
from .optimizers import (
    DistributedSpeedOptimizer,
    AdamOptimizer,
    AdamWOptimizer,
    SGDOptimizer
)

from .schedulers import (
    WarmupLRScheduler,
    WarmupCosineScheduler, 
    WarmupLinearScheduler,
    OneCycleScheduler
)

# Import monitoring and profiling
from .monitoring import (
    ThroughputMonitor,
    MemoryMonitor, 
    CommunicationMonitor
)

from .profiling import ProfilerContext, profile

# Global state management
_GLOBAL_STATE = {
    'initialized': False,
    'config': None,
    'engine': None,
    'local_rank': -1,
    'world_size': -1,
    'rank': -1
}

def _validate_config(config: Union[Dict, str, DistributedSpeedConfig]) -> DistributedSpeedConfig:
    """Validate and convert configuration to DistributedSpeedConfig object."""
    
    if isinstance(config, str):
        # Load from file path
        if not os.path.exists(config):
            raise FileNotFoundError(f"Config file not found: {config}")
        config = load_config(config)
    elif isinstance(config, dict):
        config = DistributedSpeedConfig(config)
    elif not isinstance(config, DistributedSpeedConfig):
        raise TypeError(f"Config must be dict, str, or DistributedSpeedConfig, got {type(config)}")
    
    # Validate required fields
    required_fields = ['train_batch_size']
    for field in required_fields:
        if not hasattr(config, field) or getattr(config, field) is None:
            raise ValueError(f"Config missing required field: {field}")
    
    return config

def _setup_distributed():
    """Initialize distributed training environment if not already initialized."""
    
    if not dist.is_available():
        raise RuntimeError("Distributed package not available")
    
    # Initialize distributed backend if not already done
    if not dist.is_initialized():
        # Try to initialize from environment variables
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            master_addr = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '29500')
            
            init_method = f"tcp://{master_addr}:{master_port}"
            
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method=init_method,
                rank=rank,
                world_size=world_size
            )
            
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        else:
            # Single process training
            rank = 0
            world_size = 1
            local_rank = 0
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Update global state
    _GLOBAL_STATE.update({
        'rank': rank,
        'world_size': world_size, 
        'local_rank': local_rank
    })
    
    return rank, world_size, local_rank

def initialize(
    model: torch.nn.Module,
    config: Union[Dict, str, DistributedSpeedConfig],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    training_data: Optional[Any] = None,
    mpu: Optional[Any] = None,
    collate_fn: Optional[callable] = None,
    dist_init_required: bool = True
) -> Tuple[DistributedSpeedEngine, torch.optim.Optimizer, Any, Any]:
    """
    Initialize DistributedSpeed engine for distributed training.
    
    Args:
        model: PyTorch model to train
        config: DistributedSpeed configuration dict, path, or DistributedSpeedConfig object
        optimizer: Optional optimizer (will be created if not provided)
        lr_scheduler: Optional learning rate scheduler  
        training_data: Optional training dataset/dataloader
        mpu: Optional model parallel unit for advanced parallelism
        collate_fn: Optional data collation function
        dist_init_required: Whether to initialize distributed backend
        
    Returns:
        Tuple of (engine, optimizer, dataloader, lr_scheduler)
        
    Example:
        >>> model = torch.nn.Linear(10, 1)
        >>> config = {"train_batch_size": 32, "zero_optimization": {"stage": 2}}
        >>> engine, optimizer, _, scheduler = initialize(model, config)
    """
    
    # Validate configuration
    config = _validate_config(config)
    _GLOBAL_STATE['config'] = config
    
    # Setup distributed environment
    if dist_init_required:
        rank, world_size, local_rank = _setup_distributed()
        log_dist(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        rank = get_rank()
        world_size = get_world_size() 
        local_rank = get_local_rank()
    
    # Move model to appropriate device
    if torch.cuda.is_available() and local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        if rank == 0:
            warnings.warn("CUDA not available, falling back to CPU training")
    
    # Initialize communication manager
    comm_manager = CommManager(config)
    
    # Initialize memory manager
    memory_manager = MemoryManager(config)
    
    # Create engine
    engine = DistributedSpeedEngine(
        model=model,
        config=config,
        comm_manager=comm_manager,
        memory_manager=memory_manager
    )
    
    # Setup optimizer
    if optimizer is None:
        optimizer = engine.create_optimizer()
    else:
        optimizer = engine.wrap_optimizer(optimizer)
    
    # Setup learning rate scheduler
    if lr_scheduler is None and hasattr(config, 'scheduler') and config.scheduler is not None:
        lr_scheduler = engine.create_lr_scheduler(optimizer)
    
    # Setup dataloader if provided
    dataloader = None
    if training_data is not None:
        dataloader = engine.create_dataloader(training_data, collate_fn)
    
    # Mark as initialized
    _GLOBAL_STATE['initialized'] = True
    _GLOBAL_STATE['engine'] = engine
    
    log_dist(f"DistributedSpeed initialization complete: "
            f"ZeRO stage {getattr(config, 'zero_stage', 0)}, "
            f"pipeline stages {getattr(config, 'pipeline_stages', 1)}")
    
    return engine, optimizer, dataloader, lr_scheduler

def is_initialized() -> bool:
    """Check if DistributedSpeed has been initialized."""
    return _GLOBAL_STATE['initialized']

def get_config() -> Optional[DistributedSpeedConfig]:
    """Get the current DistributedSpeed configuration."""
    return _GLOBAL_STATE['config']

def get_engine() -> Optional[DistributedSpeedEngine]:
    """Get the current DistributedSpeed engine."""
    return _GLOBAL_STATE['engine']

# Convenience functions for distributed operations
def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return _GLOBAL_STATE.get('rank', 0)

def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return _GLOBAL_STATE.get('world_size', 1)

def get_local_rank() -> int:
    """Get local rank within node."""
    return _GLOBAL_STATE.get('local_rank', 0)

def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()

def print_rank_0(message: str):
    """Print message only on rank 0."""
    if get_rank() == 0:
        print(message)

def log_dist(message: str, level: str = "INFO"):
    """Log message with distributed information."""
    rank = get_rank()
    logger = logging.getLogger(__name__)
    getattr(logger, level.lower())(f"[Rank {rank}] {message}")

# Cleanup functions
def destroy():
    """Clean up DistributedSpeed and distributed training state."""
    global _GLOBAL_STATE
    
    if _GLOBAL_STATE['engine'] is not None:
        _GLOBAL_STATE['engine'].destroy()
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    _GLOBAL_STATE = {
        'initialized': False,
        'config': None,
        'engine': None,
        'local_rank': -1,
        'world_size': -1,
        'rank': -1
    }

# Context managers for common operations
class timer:
    """Context manager for timing operations."""
    def __init__(self, name: str, log_level: str = "INFO"):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and hasattr(self.start_time, 'record'):
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed = self.start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            import time
            elapsed = time.time() - self.start_time
            
        log_dist(f"{self.name} took {elapsed:.4f}s", self.log_level)

# Memory utilities
def empty_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def reset_peak_memory():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_memory_info():
    """Get current GPU memory information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        return {
            'allocated_gb': allocated / 1e9,
            'cached_gb': cached / 1e9,
            'max_allocated_gb': max_allocated / 1e9,
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - allocated) / 1e9
        }
    return {'allocated_gb': 0, 'cached_gb': 0, 'max_allocated_gb': 0, 'free_gb': 0}

# Configuration shortcuts
def zero_1_config() -> Dict:
    """Get ZeRO Stage 1 configuration preset."""
    return {
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        }
    }

def zero_2_config() -> Dict:
    """Get ZeRO Stage 2 configuration preset."""
    return {
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": False
        }
    }

def zero_3_config() -> Dict:
    """Get ZeRO Stage 3 configuration preset."""
    return {
        "zero_optimization": {
            "stage": 3,
            "param_persistence_threshold": 1e6,
            "model_persistence_threshold": 1e6,
            "max_live_parameters": 1e9,
            "max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": False,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "sub_group_size": 1e9,
            "elastic_checkpoint": True,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "cpu_offload": False
        }
    }

def fp16_config() -> Dict:
    """Get FP16 mixed precision configuration preset."""
    return {
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    }

def bf16_config() -> Dict:
    """Get BF16 mixed precision configuration preset."""
    return {
        "bf16": {
            "enabled": True
        }
    }

# Export all public APIs
__all__ = [
    # Core initialization
    'initialize',
    'is_initialized', 
    'destroy',
    
    # Configuration
    'DistributedSpeedConfig',
    'load_config',
    'get_config',
    'zero_1_config',
    'zero_2_config', 
    'zero_3_config',
    'fp16_config',
    'bf16_config',
    
    # Engine and core components
    'DistributedSpeedEngine',
    'get_engine',
    'ZeROOptimizer',
    'PipelineEngine',
    'CommManager',
    'MemoryManager',
    
    # Distributed utilities
    'get_rank',
    'get_world_size',
    'get_local_rank',
    'barrier',
    'print_rank_0',
    'log_dist',
    'init_distributed',
    
    # Memory utilities
    'empty_cache',
    'reset_peak_memory',
    'get_memory_info',
    'ActivationCheckpointing',
    
    # Monitoring and profiling
    'ThroughputMonitor',
    'MemoryMonitor',
    'CommunicationMonitor',
    'ProfilerContext',
    'profile',
    'timer',
    
    # Optimizers and schedulers
    'DistributedSpeedOptimizer',
    'AdamOptimizer',
    'AdamWOptimizer', 
    'SGDOptimizer',
    'WarmupLRScheduler',
    'WarmupCosineScheduler',
    'WarmupLinearScheduler',
    'OneCycleScheduler',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]