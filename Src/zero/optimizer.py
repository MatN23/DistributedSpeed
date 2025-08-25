"""
DistributedSpeed ZeRO Optimizer Implementation.

This module implements the main ZeROOptimizer class that wraps standard PyTorch optimizers
with ZeRO memory optimization stages. The optimizer handles parameter partitioning,
gradient synchronization, and communication optimization for distributed training.

Key Features:
- Multi-stage ZeRO optimization (Stages 1, 2, 3)
- Automatic parameter and gradient partitioning
- Communication overlap and optimization
- CPU/NVMe offloading support
- Mixed precision training support
- Dynamic loss scaling
- Gradient compression and quantization

The ZeROOptimizer serves as the main entry point for ZeRO functionality and delegates
stage-specific operations to appropriate stage implementations.

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import math
import time
import logging
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

from .stage1 import ZeROStage1
from .stage2 import ZeROStage2
from .stage3 import ZeROStage3
from .utils import (
    get_world_size, get_rank, flatten_dense_tensors_aligned,
    unflatten_dense_tensors, clip_grad_norm_, compute_norm,
    get_global_norm, is_model_parallel_parameter
)
from .partition import ParameterPartitioner, GradientPartitioner

logger = logging.getLogger(__name__)


class ZeROOptimizer:
    """
    ZeRO (Zero Redundancy Optimizer) wrapper for PyTorch optimizers.
    
    This class implements the ZeRO optimizer that partitions optimizer states,
    gradients, and parameters across data-parallel processes to reduce memory
    consumption while maintaining mathematical equivalence to standard optimizers.
    
    The optimizer supports three stages of optimization:
    - Stage 1: Optimizer state partitioning (4x memory reduction)
    - Stage 2: Optimizer state + gradient partitioning (8x memory reduction)  
    - Stage 3: Full partitioning including parameters (linear memory scaling)
    
    Args:
        optimizer: Base PyTorch optimizer to wrap
        config: ZeRO configuration object
        model_parameters: List of model parameters
        comm_manager: Communication manager for distributed operations
        
    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> zero_config = ZeROConfig(stage=2, cpu_offload=True)
        >>> zero_optimizer = ZeROOptimizer(optimizer, zero_config, list(model.parameters()))
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config,  # ZeROConfig type hint causes circular import
        model_parameters: List[nn.Parameter],
        comm_manager: Optional[Any] = None
    ):
        self.config = config
        self.base_optimizer = optimizer
        self.comm_manager = comm_manager
        self.model_parameters = list(model_parameters)
        
        # Distributed setup
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Validation
        self._validate_config()
        
        # Initialize stage-specific implementation
        self.zero_stage = None
        if config.stage > 0:
            self._initialize_zero_stage()
        
        # Gradient scaling for mixed precision
        self.grad_scaler = None
        if config.dynamic_loss_scale:
            self.grad_scaler = GradScaler(
                init_scale=2**config.initial_scale_power,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=config.loss_scale_window
            )
        
        # State tracking
        self.overflow_count = 0
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.current_step = 0
        self.gradient_accumulation_count = 0
        
        # Parameter and gradient management
        self.param_groups = self.base_optimizer.param_groups
        self.parameter_offload = config.cpu_offload_params
        self.optimizer_offload = config.cpu_offload
        
        # Communication optimization
        self.overlap_comm = config.overlap_comm
        self.allgather_bucket_size = int(config.allgather_bucket_size)
        self.reduce_bucket_size = int(config.reduce_bucket_size)
        
        # Memory management
        self.cpu_offload_pin_memory = config.cpu_offload_use_pin_memory
        self.contiguous_gradients = config.contiguous_gradients
        
        # Performance monitoring
        self.timers = defaultdict(float)
        self.communication_data_bytes = 0
        self.parameter_gathering_time = 0.0
        
        # Initialize partitioners if needed
        if config.stage > 0:
            self._setup_partitioning()
        
        logger.info(f"Initialized ZeRO optimizer: stage={config.stage}, "
                   f"world_size={self.world_size}, parameters={len(self.model_parameters)}")
    
    def _validate_config(self):
        """Validate ZeRO configuration."""
        
        if self.config.stage not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid ZeRO stage: {self.config.stage}")
        
        if self.world_size == 1 and self.config.stage > 0:
            warnings.warn("ZeRO optimization has no effect with single process training")
        
        if self.config.cpu_offload and not self.config.stage:
            raise ValueError("CPU offload requires ZeRO stage >= 1")
        
        # Validate gradient accumulation
        if self.config.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
    
    def _initialize_zero_stage(self):
        """Initialize stage-specific ZeRO implementation."""
        
        stage_classes = {
            1: ZeROStage1,
            2: ZeROStage2, 
            3: ZeROStage3
        }
        
        if self.config.stage in stage_classes:
            stage_class = stage_classes[self.config.stage]
            self.zero_stage = stage_class(
                optimizer=self.base_optimizer,
                config=self.config,
                model_parameters=self.model_parameters,
                comm_manager=self.comm_manager
            )
        else:
            raise ValueError(f"Unsupported ZeRO stage: {self.config.stage}")
    
    def _setup_partitioning(self):
        """Setup parameter and gradient partitioning."""
        
        # Parameter partitioner for Stage 3
        if self.config.stage == 3:
            self.parameter_partitioner = ParameterPartitioner(
                parameters=self.model_parameters,
                world_size=self.world_size,
                rank=self.rank,
                config=self.config
            )
        
        # Gradient partitioner for Stage 2+
        if self.config.stage >= 2:
            self.gradient_partitioner = GradientPartitioner(
                parameters=self.model_parameters,
                world_size=self.world_size,
                rank=self.rank,
                config=self.config
            )
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients with ZeRO-aware implementation."""
        
        if self.zero_stage:
            self.zero_stage.zero_grad(set_to_none=set_to_none)
        else:
            self.base_optimizer.zero_grad(set_to_none=set_to_none)
        
        # Reset gradient accumulation counter
        self.gradient_accumulation_count = 0
    
    def step(self, closure=None):
        """
        Perform optimizer step with ZeRO optimization.
        
        Args:
            closure: Optional closure function for loss computation
            
        Returns:
            Loss value if closure provided, otherwise None
        """
        
        start_time = time.time()
        
        # Handle gradient accumulation
        self.gradient_accumulation_count += 1
        
        if self.gradient_accumulation_count < self.gradient_accumulation_steps:
            # Still accumulating gradients, skip optimizer step
            return None
        
        # Reset accumulation counter
        self.gradient_accumulation_count = 0
        
        # Check for gradient overflow in mixed precision training
        if self.grad_scaler is not None:
            # Check if gradients are finite
            found_inf = self._check_overflow()
            if found_inf:
                self.overflow_count += 1
                logger.warning(f"Gradient overflow detected at step {self.current_step}")
                return None
        
        # Perform stage-specific optimization step
        loss = None
        if self.zero_stage:
            loss = self.zero_stage.step(closure)
        else:
            # Standard optimizer step
            if closure is not None:
                loss = closure()
            
            # Scale gradients if using manual scaling
            if self.grad_scaler is None and self.config.loss_scale != 1.0:
                self._scale_gradients(1.0 / self.config.loss_scale)
            
            # Clip gradients
            if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
                self.clip_grad_norm_(self.config.max_grad_norm)
            
            # Optimizer step
            self.base_optimizer.step()
        
        # Update gradient scaler
        if self.grad_scaler is not None:
            self.grad_scaler.update()
        
        # Update step counter
        self.current_step += 1
        
        # Update timing
        self.timers['step_time'] += time.time() - start_time
        
        return loss
    
    def _check_overflow(self) -> bool:
        """Check for gradient overflow in mixed precision training."""
        
        if self.grad_scaler is None:
            return False
        
        # Get optimizer state from scaler
        optimizer_state = self.grad_scaler._per_optimizer_states[id(self.base_optimizer)]
        
        # Check for infinite gradients
        if "found_inf_per_device" in optimizer_state:
            found_inf = optimizer_state["found_inf_per_device"]
            if hasattr(found_inf, 'item'):
                return found_inf.item() > 0
        
        return False
    
    def _scale_gradients(self, scale: float):
        """Scale gradients by given factor."""
        
        for param in self.model_parameters:
            if param.grad is not None:
                param.grad.mul_(scale)
    
    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Perform backward pass with ZeRO optimization.
        
        Args:
            loss: Loss tensor to backpropagate
            retain_graph: Whether to retain computation graph
        """
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        
        # Apply gradient scaling for mixed precision
        if self.grad_scaler is not None:
            scaled_loss = self.grad_scaler.scale(scaled_loss)
        elif self.config.loss_scale != 1.0:
            scaled_loss = scaled_loss * self.config.loss_scale
        
        # Backward pass
        if self.zero_stage and hasattr(self.zero_stage, 'backward'):
            self.zero_stage.backward(scaled_loss, retain_graph=retain_graph)
        else:
            scaled_loss.backward(retain_graph=retain_graph)
    
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        """
        Clip gradient norm with ZeRO-aware implementation.
        
        Args:
            max_norm: Maximum allowed gradient norm
            norm_type: Type of norm to compute (default: 2.0)
            
        Returns:
            Total norm of gradients
        """
        
        if self.zero_stage and hasattr(self.zero_stage, 'clip_grad_norm_'):
            return self.zero_stage.clip_grad_norm_(max_norm, norm_type)
        else:
            # Standard gradient clipping
            parameters = [p for p in self.model_parameters if p.grad is not None]
            return clip_grad_norm_(parameters, max_norm, norm_type)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        
        if len(self.param_groups) > 0:
            return self.param_groups[0]['lr']
        return 0.0
    
    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        
        for param_group in self.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get optimizer state dictionary with ZeRO-specific handling.
        
        Returns:
            State dictionary containing optimizer state
        """
        
        if self.zero_stage and hasattr(self.zero_stage, 'state_dict'):
            return self.zero_stage.state_dict()
        else:
            state_dict = self.base_optimizer.state_dict()
            
            # Add ZeRO-specific state
            state_dict['zero_stage'] = self.config.stage
            state_dict['current_step'] = self.current_step
            state_dict['overflow_count'] = self.overflow_count
            state_dict['gradient_accumulation_count'] = self.gradient_accumulation_count
            
            if self.grad_scaler is not None:
                state_dict['grad_scaler'] = self.grad_scaler.state_dict()
            
            return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load optimizer state dictionary with ZeRO-specific handling.
        
        Args:
            state_dict: State dictionary to load
        """
        
        if self.zero_stage and hasattr(self.zero_stage, 'load_state_dict'):
            self.zero_stage.load_state_dict(state_dict)
        else:
            # Extract ZeRO-specific state
            self.current_step = state_dict.pop('current_step', 0)
            self.overflow_count = state_dict.pop('overflow_count', 0)
            self.gradient_accumulation_count = state_dict.pop('gradient_accumulation_count', 0)
            
            # Load gradient scaler state
            if 'grad_scaler' in state_dict and self.grad_scaler is not None:
                self.grad_scaler.load_state_dict(state_dict.pop('grad_scaler'))
            
            # Remove ZeRO-specific keys
            state_dict.pop('zero_stage', None)
            
            # Load base optimizer state
            self.base_optimizer.load_state_dict(state_dict)
    
    @contextmanager
    def no_sync(self):
        """Context manager to skip gradient synchronization."""
        
        if self.zero_stage and hasattr(self.zero_stage, 'no_sync'):
            with self.zero_stage.no_sync():
                yield
        else:
            # For stage 0, just yield without special handling
            yield
    
    def gather_params(self) -> Dict[str, torch.Tensor]:
        """
        Gather all parameters from partitions (Stage 3 only).
        
        Returns:
            Dictionary mapping parameter names to gathered tensors
        """
        
        if self.zero_stage and hasattr(self.zero_stage, 'gather_params'):
            return self.zero_stage.gather_params()
        else:
            # For non-Stage 3, parameters are already available
            return {f"param_{i}": param for i, param in enumerate(self.model_parameters)}
    
    def partition_params(self, params_dict: Dict[str, torch.Tensor]):
        """
        Partition parameters back to distributed format (Stage 3 only).
        
        Args:
            params_dict: Dictionary of parameter names to tensors
        """
        
        if self.zero_stage and hasattr(self.zero_stage, 'partition_params'):
            self.zero_stage.partition_params(params_dict)
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        
        info = {}
        
        if torch.cuda.is_available():
            info.update({
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'cached_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            })
        
        # Add ZeRO-specific memory info
        if self.zero_stage and hasattr(self.zero_stage, 'get_memory_info'):
            zero_info = self.zero_stage.get_memory_info()
            info.update(zero_info)
        
        return info
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics.
        
        Returns:
            Dictionary with communication metrics
        """
        
        stats = {
            'total_comm_bytes': self.communication_data_bytes,
            'avg_comm_time': self.timers.get('comm_time', 0.0) / max(1, self.current_step),
            'parameter_gathering_time': self.parameter_gathering_time
        }
        
        if self.zero_stage and hasattr(self.zero_stage, 'get_communication_stats'):
            zero_stats = self.zero_stage.get_communication_stats()
            stats.update(zero_stats)
        
        return stats
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """
        Get throughput statistics.
        
        Returns:
            Dictionary with throughput metrics
        """
        
        total_time = self.timers.get('step_time', 1.0)
        
        return {
            'steps_per_second': self.current_step / total_time,
            'avg_step_time': total_time / max(1, self.current_step),
            'overflow_rate': self.overflow_count / max(1, self.current_step)
        }
    
    def reset_stats(self):
        """Reset all statistics."""
        
        self.timers.clear()
        self.communication_data_bytes = 0
        self.parameter_gathering_time = 0.0
        self.overflow_count = 0
        
        if self.zero_stage and hasattr(self.zero_stage, 'reset_stats'):
            self.zero_stage.reset_stats()
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage for current configuration.
        
        Returns:
            Dictionary with memory estimates in GB
        """
        
        num_params = sum(p.numel() for p in self.model_parameters)
        
        # Base parameter memory (4 bytes per FP32 parameter)
        param_memory = num_params * 4 / 1e9
        
        # Gradient memory
        grad_memory = param_memory
        
        # Optimizer state memory (assumes Adam-like optimizer)
        optimizer_memory = param_memory * 2  # momentum + variance
        
        # Apply ZeRO partitioning
        if self.config.stage == 1:
            optimizer_memory /= self.world_size
        elif self.config.stage == 2:
            optimizer_memory /= self.world_size
            grad_memory /= self.world_size
        elif self.config.stage == 3:
            optimizer_memory /= self.world_size
            grad_memory /= self.world_size
            param_memory /= self.world_size
        
        total_memory = param_memory + grad_memory + optimizer_memory
        
        return {
            'parameters_gb': param_memory,
            'gradients_gb': grad_memory,
            'optimizer_states_gb': optimizer_memory,
            'total_gb': total_memory,
            'memory_per_gpu_gb': total_memory
        }
    
    def __repr__(self) -> str:
        """String representation of ZeRO optimizer."""
        
        return (
            f"ZeROOptimizer(stage={self.config.stage}, "
            f"world_size={self.world_size}, "
            f"parameters={len(self.model_parameters)}, "
            f"cpu_offload={self.config.cpu_offload})"
        )
    
    def __getattr__(self, name: str):
        """Delegate attribute access to base optimizer if not found."""
        
        if hasattr(self.base_optimizer, name):
            return getattr(self.base_optimizer, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class ZeROOptimizerBuilder:
    """
    Builder pattern for creating ZeRO optimizers with validation.
    
    This class provides a fluent interface for configuring ZeRO optimizers
    with comprehensive validation and automatic configuration.
    """
    
    def __init__(self):
        self.base_optimizer = None
        self.config = None
        self.model_parameters = None
        self.comm_manager = None
        self.model_size_hint = None
        self.memory_budget_gb = None
    
    def with_optimizer(self, optimizer: Optimizer):
        """Set base optimizer."""
        self.base_optimizer = optimizer
        return self
    
    def with_config(self, config):
        """Set ZeRO configuration."""
        self.config = config
        return self
    
    def with_parameters(self, parameters: List[nn.Parameter]):
        """Set model parameters."""
        self.model_parameters = parameters
        return self
    
    def with_comm_manager(self, comm_manager):
        """Set communication manager."""
        self.comm_manager = comm_manager
        return self
    
    def with_model_size_hint(self, num_parameters: int):
        """Provide model size hint for automatic configuration."""
        self.model_size_hint = num_parameters
        return self
    
    def with_memory_budget(self, memory_gb: float):
        """Set memory budget for automatic configuration."""
        self.memory_budget_gb = memory_gb
        return self
    
    def auto_configure(self):
        """Automatically configure ZeRO based on hints."""
        
        if self.model_size_hint is None or self.memory_budget_gb is None:
            raise ValueError("Model size hint and memory budget required for auto-configuration")
        
        # Import here to avoid circular import
        from . import ZeROConfig
        
        # Estimate memory requirements
        param_memory_gb = self.model_size_hint * 4 / 1e9
        
        # Choose appropriate ZeRO stage based on memory budget
        if param_memory_gb * 8 <= self.memory_budget_gb:
            # Can fit with Stage 1
            stage = 1
            cpu_offload = False
        elif param_memory_gb * 4 <= self.memory_budget_gb:
            # Need Stage 2
            stage = 2
            cpu_offload = False
        elif param_memory_gb * 2 <= self.memory_budget_gb:
            # Need Stage 3
            stage = 3
            cpu_offload = False
        else:
            # Need Stage 3 with CPU offloading
            stage = 3
            cpu_offload = True
        
        self.config = ZeROConfig(
            stage=stage,
            cpu_offload=cpu_offload,
            overlap_comm=True,
            contiguous_gradients=True
        )
        
        return self
    
    def build(self) -> ZeROOptimizer:
        """Build ZeRO optimizer."""
        
        if self.base_optimizer is None:
            raise ValueError("Base optimizer must be provided")
        
        if self.config is None:
            raise ValueError("ZeRO config must be provided")
        
        if self.model_parameters is None:
            raise ValueError("Model parameters must be provided")
        
        return ZeROOptimizer(
            optimizer=self.base_optimizer,
            config=self.config,
            model_parameters=self.model_parameters,
            comm_manager=self.comm_manager
        )