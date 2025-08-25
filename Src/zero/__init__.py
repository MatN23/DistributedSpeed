"""
DistributedSpeed ZeRO (Zero Redundancy Optimizer) Package.

This package implements the ZeRO optimizer family for memory-efficient distributed training.
ZeRO eliminates memory redundancies in data-parallel training by partitioning optimizer states,
gradients, and parameters across data-parallel processes.

ZeRO Stages:
- Stage 1: Optimizer State Partitioning - 4x memory reduction for optimizer states
- Stage 2: Optimizer State + Gradient Partitioning - 8x memory reduction  
- Stage 3: Optimizer State + Gradient + Parameter Partitioning - Linear memory scaling
- Infinity: CPU/NVMe offloading for massive models

Key Features:
- Memory-efficient training of large models
- Automatic gradient and parameter gathering
- Communication optimization and overlap
- CPU offloading support
- Dynamic loss scaling for mixed precision
- Gradient compression and quantization

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from .optimizer import ZeROOptimizer
from .stage1 import ZeROStage1
from .stage2 import ZeROStage2  
from .stage3 import ZeROStage3
from .partition import ParameterPartitioner, GradientPartitioner
from .utils import (
    get_world_size,
    get_rank, 
    flatten_dense_tensors_aligned,
    unflatten_dense_tensors,
    clip_grad_norm_,
    compute_norm
)

logger = logging.getLogger(__name__)


@dataclass
class ZeROConfig:
    """
    Configuration for ZeRO optimizer.
    
    This class defines all configuration options for ZeRO optimization stages,
    including memory management, communication optimization, and CPU offloading.
    """
    
    # Core ZeRO settings
    stage: int = 0
    cpu_offload: bool = False
    cpu_offload_params: bool = False
    cpu_offload_use_pin_memory: bool = True
    nvme_swap_dir: Optional[str] = None
    
    # Communication optimization
    allgather_partitions: bool = True
    allgather_bucket_size: float = 2e8
    overlap_comm: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: float = 2e8
    contiguous_gradients: bool = True
    bucket_cap_mb: float = 25.0
    
    # Stage 3 specific settings
    param_persistence_threshold: float = 1e6
    model_persistence_threshold: float = 1e6
    max_live_parameters: float = 1e9
    max_reuse_distance: float = 1e9
    prefetch_bucket_size: float = 5e8
    param_round_robin: bool = False
    offload_optimizer_config: Optional[Dict[str, Any]] = None
    offload_param_config: Optional[Dict[str, Any]] = None
    
    # Memory management
    sub_group_size: float = 1e9
    elastic_checkpoint: bool = True
    ignore_unused_parameters: bool = True
    partition_grads: bool = True
    round_robin_gradients: bool = False
    
    # Advanced features
    zero_hpz_partition_size: int = 1
    zero_quantized_weights: bool = False
    zero_quantized_gradients: bool = False
    mics_shard_size: int = -1
    mics_hierarchical_params_gather: bool = False
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    loss_scale: float = 1.0
    dynamic_loss_scale: bool = True
    initial_scale_power: int = 16
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: float = 1.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        
        if self.stage not in [0, 1, 2, 3]:
            raise ValueError(f"ZeRO stage must be 0, 1, 2, or 3, got {self.stage}")
        
        if self.stage == 0:
            # Stage 0 is regular data parallelism, disable ZeRO features
            self.cpu_offload = False
            self.cpu_offload_params = False
        
        # Validate bucket sizes
        if self.allgather_bucket_size <= 0:
            raise ValueError("allgather_bucket_size must be positive")
        
        if self.reduce_bucket_size <= 0:
            raise ValueError("reduce_bucket_size must be positive")
        
        # Validate stage 3 parameters
        if self.stage == 3:
            if self.param_persistence_threshold < 0:
                raise ValueError("param_persistence_threshold must be non-negative")
            
            if self.max_live_parameters <= 0:
                raise ValueError("max_live_parameters must be positive")


def create_zero_optimizer(
    optimizer: Optimizer,
    config: Union[ZeROConfig, Dict[str, Any]],
    model_parameters: List[torch.nn.Parameter],
    comm_manager: Optional[Any] = None
) -> ZeROOptimizer:
    """
    Create ZeRO optimizer wrapper.
    
    Args:
        optimizer: Base PyTorch optimizer
        config: ZeRO configuration
        model_parameters: List of model parameters
        comm_manager: Communication manager
        
    Returns:
        ZeRO-wrapped optimizer
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> zero_config = ZeROConfig(stage=2, cpu_offload=True)
        >>> zero_optimizer = create_zero_optimizer(optimizer, zero_config, list(model.parameters()))
    """
    
    if isinstance(config, dict):
        config = ZeROConfig(**config)
    
    return ZeROOptimizer(
        optimizer=optimizer,
        config=config,
        model_parameters=model_parameters,
        comm_manager=comm_manager
    )


def get_zero_stage_class(stage: int):
    """Get ZeRO stage implementation class."""
    
    stage_classes = {
        1: ZeROStage1,
        2: ZeROStage2,
        3: ZeROStage3
    }
    
    if stage not in stage_classes:
        raise ValueError(f"Unsupported ZeRO stage: {stage}")
    
    return stage_classes[stage]


def estimate_zero_memory_savings(
    num_parameters: int,
    world_size: int,
    stage: int = 2,
    optimizer_type: str = "adamw"
) -> Dict[str, float]:
    """
    Estimate memory savings from ZeRO optimization.
    
    Args:
        num_parameters: Total number of model parameters
        world_size: Number of distributed processes
        stage: ZeRO stage (1, 2, or 3)
        optimizer_type: Optimizer type ("adam", "adamw", "sgd")
        
    Returns:
        Dictionary with memory usage estimates in GB
    """
    
    # Parameter memory (4 bytes per FP32 parameter)
    param_memory = num_parameters * 4 / 1e9
    
    # Gradient memory (same as parameters)
    grad_memory = param_memory
    
    # Optimizer state memory (depends on optimizer)
    optimizer_multipliers = {
        "sgd": 0,  # No additional state
        "momentum": 1,  # Momentum buffer
        "adam": 2,  # Momentum + variance
        "adamw": 2  # Momentum + variance
    }
    
    optimizer_multiplier = optimizer_multipliers.get(optimizer_type.lower(), 2)
    optimizer_memory = param_memory * optimizer_multiplier
    
    # Calculate savings based on ZeRO stage
    if stage == 1:
        # Only optimizer states are partitioned
        optimizer_memory_partitioned = optimizer_memory / world_size
        param_memory_partitioned = param_memory  # Not partitioned
        grad_memory_partitioned = grad_memory  # Not partitioned
        
    elif stage == 2:
        # Optimizer states and gradients are partitioned
        optimizer_memory_partitioned = optimizer_memory / world_size
        param_memory_partitioned = param_memory  # Not partitioned
        grad_memory_partitioned = grad_memory / world_size
        
    elif stage == 3:
        # All components are partitioned
        optimizer_memory_partitioned = optimizer_memory / world_size
        param_memory_partitioned = param_memory / world_size
        grad_memory_partitioned = grad_memory / world_size
        
    else:
        # No partitioning (baseline)
        optimizer_memory_partitioned = optimizer_memory
        param_memory_partitioned = param_memory
        grad_memory_partitioned = grad_memory
    
    baseline_memory = param_memory + grad_memory + optimizer_memory
    zero_memory = param_memory_partitioned + grad_memory_partitioned + optimizer_memory_partitioned
    
    return {
        "baseline_memory_gb": baseline_memory,
        "zero_memory_gb": zero_memory,
        "memory_savings_gb": baseline_memory - zero_memory,
        "memory_reduction_factor": baseline_memory / zero_memory,
        "parameters_gb": param_memory_partitioned,
        "gradients_gb": grad_memory_partitioned,
        "optimizer_states_gb": optimizer_memory_partitioned
    }


def validate_zero_config(config: ZeROConfig, world_size: int) -> List[str]:
    """
    Validate ZeRO configuration and return warning messages.
    
    Args:
        config: ZeRO configuration to validate
        world_size: Number of distributed processes
        
    Returns:
        List of validation warning messages
    """
    
    warnings = []
    
    # Check world size compatibility
    if world_size == 1 and config.stage > 0:
        warnings.append("ZeRO optimization has no effect with single process training")
    
    # Check CPU offloading requirements
    if config.cpu_offload and config.stage == 0:
        warnings.append("CPU offloading requires ZeRO stage >= 1")
    
    # Check bucket sizes
    if config.allgather_bucket_size > 1e9:
        warnings.append("Large allgather bucket size may cause memory issues")
    
    if config.reduce_bucket_size > 1e9:
        warnings.append("Large reduce bucket size may cause memory issues")
    
    # Stage 3 specific checks
    if config.stage == 3:
        if config.param_persistence_threshold > config.model_persistence_threshold:
            warnings.append("param_persistence_threshold should not exceed model_persistence_threshold")
        
        if config.max_live_parameters < config.param_persistence_threshold:
            warnings.append("max_live_parameters should be >= param_persistence_threshold")
    
    # Mixed precision checks
    if config.dynamic_loss_scale and config.loss_scale > 1.0:
        warnings.append("Static loss scale specified but dynamic scaling is enabled")
    
    return warnings


# Export public APIs
__all__ = [
    # Main classes
    'ZeROOptimizer',
    'ZeROConfig',
    'ZeROStage1', 
    'ZeROStage2',
    'ZeROStage3',
    
    # Utilities
    'create_zero_optimizer',
    'get_zero_stage_class',
    'estimate_zero_memory_savings',
    'validate_zero_config',
    
    # Partitioning
    'ParameterPartitioner',
    'GradientPartitioner',
    
    # Helper functions
    'flatten_dense_tensors_aligned',
    'unflatten_dense_tensors', 
    'clip_grad_norm_',
    'compute_norm',
    'get_world_size',
    'get_rank'
]