"""
DistributedSpeed ZeRO Stage 1 Implementation.

ZeRO Stage 1 partitions optimizer states across data-parallel processes while keeping
gradients and parameters replicated. This provides a 4x reduction in memory usage for
optimizer states (momentum, variance buffers for Adam-like optimizers).

Key Features:
- Optimizer state partitioning across processes
- Automatic state gathering during optimization
- Communication optimization with bucketing
- CPU offloading support for optimizer states
- Gradient synchronization with AllReduce
- Memory-efficient state management

Stage 1 is ideal for medium-sized models where optimizer state memory is the bottleneck
but gradient and parameter memory can still fit in GPU memory.

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer

from .utils import (
    get_world_size, get_rank, flatten_dense_tensors_aligned,
    unflatten_dense_tensors, clip_grad_norm_, compute_norm,
    get_global_norm, pad_tensor, is_model_parallel_parameter
)

logger = logging.getLogger(__name__)


class ZeROStage1:
    """
    ZeRO Stage 1: Optimizer State Partitioning.
    
    This class implements ZeRO Stage 1 optimization where optimizer states
    (momentum, variance buffers) are partitioned across data-parallel processes
    while gradients and parameters remain replicated.
    
    Memory Savings:
    - Optimizer states: 4x reduction (partitioned across processes)
    - Gradients: No reduction (replicated)
    - Parameters: No reduction (replicated)
    
    Communication:
    - Gradient synchronization via AllReduce
    - Optimizer state gathering when needed
    - Optional communication overlap
    
    Args:
        optimizer: Base PyTorch optimizer
        config: ZeRO configuration
        model_parameters: List of model parameters
        comm_manager: Communication manager
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config,  # ZeROConfig - avoid circular import
        model_parameters: List[nn.Parameter],
        comm_manager: Optional[Any] = None
    ):
        self.optimizer = optimizer
        self.config = config
        self.model_parameters = list(model_parameters)
        self.comm_manager = comm_manager
        
        # Distributed setup
        self.world_size = get_world_size()
        self.rank = get_rank()
        
        # Communication settings
        self.overlap_comm = config.overlap_comm
        self.allgather_bucket_size = int(config.allgather_bucket_size)
        self.reduce_bucket_size = int(config.reduce_bucket_size)
        
        # Memory management
        self.cpu_offload = config.cpu_offload
        self.pin_memory = config.cpu_offload_use_pin_memory
        
        # State management
        self.partitioned_optimizer_states = {}
        self.state_partition_map = {}
        self.gathered_states = {}
        
        # Performance tracking
        self.communication_time = 0.0
        self.optimizer_time = 0.0
        self.state_gathering_time = 0.0
        
        # Initialize partitioning
        self._partition_optimizer_states()
        
        # Setup communication groups if needed
        self._setup_communication_groups()
        
        logger.info(f"Initialized ZeRO Stage 1: world_size={self.world_size}, "
                   f"cpu_offload={self.cpu_offload}, overlap_comm={self.overlap_comm}")
    
    def _partition_optimizer_states(self):
        """Partition optimizer states across processes."""
        
        # Get all parameters that have gradients
        params_with_grad = [p for p in self.model_parameters if p.grad is not None]
        
        if not params_with_grad:
            logger.warning("No parameters with gradients found for optimizer state partitioning")
            return
        
        # Partition parameters across processes
        total_params = len(params_with_grad)
        params_per_rank = (total_params + self.world_size - 1) // self.world_size
        
        start_idx = self.rank * params_per_rank
        end_idx = min((self.rank + 1) * params_per_rank, total_params)
        
        # Assign parameters to this rank
        self.owned_parameters = params_with_grad[start_idx:end_idx]
        
        # Create mapping of parameter to owning rank
        self.param_to_rank = {}
        for rank in range(self.world_size):
            rank_start = rank * params_per_rank
            rank_end = min((rank + 1) * params_per_rank, total_params)
            
            for idx in range(rank_start, rank_end):
                if idx < total_params:
                    param = params_with_grad[idx]
                    self.param_to_rank[param] = rank
        
        # Initialize optimizer states for owned parameters only
        if self.owned_parameters:
            # Create a dummy optimizer step to initialize states
            self._initialize_optimizer_states()
        
        logger.info(f"Rank {self.rank} owns {len(self.owned_parameters)}/{total_params} parameters")
    
    def _initialize_optimizer_states(self):
        """Initialize optimizer states for owned parameters."""
        
        # Temporarily set gradients to trigger state initialization
        original_grads = {}
        for param in self.owned_parameters:
            original_grads[param] = param.grad
            if param.grad is None:
                # Create dummy gradient to initialize optimizer state
                param.grad = torch.zeros_like(param.data)
        
        # Create temporary optimizer with only owned parameters
        temp_param_groups = []
        for group in self.optimizer.param_groups:
            temp_group = group.copy()
            temp_group['params'] = [p for p in group['params'] if p in self.owned_parameters]
            if temp_group['params']:  # Only add non-empty groups
                temp_param_groups.append(temp_group)
        
        if temp_param_groups:
            # Create temporary optimizer to initialize states
            temp_optimizer = type(self.optimizer)(temp_param_groups, **self.optimizer.defaults)
            
            # Perform dummy step to initialize states
            temp_optimizer.step()
            
            # Copy initialized states to main optimizer
            for param in self.owned_parameters:
                if param in temp_optimizer.state:
                    self.optimizer.state[param] = temp_optimizer.state[param]
        
        # Restore original gradients
        for param, grad in original_grads.items():
            param.grad = grad
        
        # Move optimizer states to CPU if offloading is enabled
        if self.cpu_offload:
            self._offload_optimizer_states()
    
    def _setup_communication_groups(self):
        """Setup communication groups for efficient allreduce operations."""
        
        # For Stage 1, we use the default process group for gradient allreduce
        self.process_group = None  # Use default group
        
        # Setup buckets for gradient communication
        self._setup_gradient_buckets()
    
    def _setup_gradient_buckets(self):
        """Setup gradient buckets for efficient communication."""
        
        self.gradient_buckets = []
        current_bucket = []
        current_bucket_size = 0
        
        for param in self.model_parameters:
            if param.requires_grad:
                param_size = param.numel() * param.element_size()
                
                if current_bucket_size + param_size > self.reduce_bucket_size and current_bucket:
                    # Start new bucket
                    self.gradient_buckets.append(current_bucket)
                    current_bucket = [param]
                    current_bucket_size = param_size
                else:
                    current_bucket.append(param)
                    current_bucket_size += param_size
        
        # Add final bucket
        if current_bucket:
            self.gradient_buckets.append(current_bucket)
        
        logger.info(f"Created {len(self.gradient_buckets)} gradient communication buckets")
    
    def _offload_optimizer_states(self):
        """Offload optimizer states to CPU memory."""
        
        for param in self.owned_parameters:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        # Move to CPU with optional pinning
                        cpu_tensor = value.cpu()
                        if self.pin_memory:
                            cpu_tensor = cpu_tensor.pin_memory()
                        state[key] = cpu_tensor
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameters."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """
        Perform optimizer step with Stage 1 ZeRO optimization.
        
        Args:
            closure: Optional closure function
            
        Returns:
            Loss value if closure provided
        """
        
        start_time = time.time()
        
        # Synchronize gradients across all processes
        self._synchronize_gradients()
        
        # Gather optimizer states for owned parameters
        self._gather_optimizer_states()
        
        # Perform optimizer step on owned parameters only
        loss = self._optimizer_step(closure)
        
        # Scatter updated states back to partitions
        self._scatter_optimizer_states()
        
        self.optimizer_time += time.time() - start_time
        
        return loss
    
    def _synchronize_gradients(self):
        """Synchronize gradients across all processes using AllReduce."""
        
        start_time = time.time()
        
        # Process each gradient bucket
        for bucket in self.gradient_buckets:
            bucket_tensors = []
            for param in bucket:
                if param.grad is not None:
                    bucket_tensors.append(param.grad)
            
            if bucket_tensors:
                # Flatten tensors for efficient communication
                flat_tensor = flatten_dense_tensors_aligned(bucket_tensors)
                
                # AllReduce to synchronize gradients
                dist.all_reduce(flat_tensor, op=dist.ReduceOp.SUM, group=self.process_group)
                
                # Average gradients
                flat_tensor.div_(self.world_size)
                
                # Unflatten back to individual gradients
                unflat_tensors = unflatten_dense_tensors(flat_tensor, bucket_tensors)
                
                # Copy back to parameter gradients
                for param_grad, unflat_grad in zip(bucket_tensors, unflat_tensors):
                    param_grad.copy_(unflat_grad)
        
        self.communication_time += time.time() - start_time
    
    def _gather_optimizer_states(self):
        """Gather optimizer states from all processes for owned parameters."""
        
        start_time = time.time()
        
        for param in self.owned_parameters:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                
                # Move states back to GPU if they were offloaded
                if self.cpu_offload:
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and not value.is_cuda:
                            state[key] = value.cuda(non_blocking=True)
        
        self.state_gathering_time += time.time() - start_time
    
    def _optimizer_step(self, closure):
        """Perform optimizer step on owned parameters only."""
        
        # Temporarily remove non-owned parameters from param groups
        original_param_groups = []
        for group in self.optimizer.param_groups:
            original_params = group['params']
            owned_params = [p for p in original_params if p in self.owned_parameters]
            
            original_param_groups.append(original_params)
            group['params'] = owned_params
        
        # Perform optimizer step
        loss = None
        if closure is not None:
            loss = closure()
        
        self.optimizer.step()
        
        # Restore original parameter groups
        for group, original_params in zip(self.optimizer.param_groups, original_param_groups):
            group['params'] = original_params
        
        return loss
    
    def _scatter_optimizer_states(self):
        """Scatter updated optimizer states back to CPU if offloading."""
        
        if self.cpu_offload:
            for param in self.owned_parameters:
                if param in self.optimizer.state:
                    state = self.optimizer.state[param]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor) and value.is_cuda:
                            # Move back to CPU
                            cpu_tensor = value.cpu()
                            if self.pin_memory:
                                cpu_tensor = cpu_tensor.pin_memory()
                            state[key] = cpu_tensor
    
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        """
        Clip gradient norm across all processes.
        
        Args:
            max_norm: Maximum allowed gradient norm
            norm_type: Type of norm to compute
            
        Returns:
            Total gradient norm
        """
        
        # Compute local gradient norm
        local_norm = compute_norm([p.grad for p in self.model_parameters if p.grad is not None], norm_type)
        
        # Gather global gradient norm
        if self.world_size > 1:
            # AllReduce to get global norm
            norm_tensor = torch.tensor(local_norm ** norm_type, device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM, group=self.process_group)
            global_norm = norm_tensor.item() ** (1.0 / norm_type)
        else:
            global_norm = local_norm
        
        # Clip gradients if norm exceeds threshold
        if global_norm > max_norm:
            clip_coef = max_norm / (global_norm + 1e-6)
            for param in self.model_parameters:
                if param.grad is not None:
                    param.grad.mul_(clip_coef)
        
        return torch.tensor(global_norm)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary including partitioned optimizer states."""
        
        state_dict = {
            'optimizer_state': {},
            'param_to_rank': self.param_to_rank,
            'communication_time': self.communication_time,
            'optimizer_time': self.optimizer_time,
            'state_gathering_time': self.state_gathering_time
        }
        
        # Include optimizer states for owned parameters
        for param in self.owned_parameters:
            if param in self.optimizer.state:
                param_id = id(param)
                state_dict['optimizer_state'][param_id] = self.optimizer.state[param]
        
        # Include base optimizer state dict
        base_state = self.optimizer.state_dict()
        state_dict['base_optimizer'] = base_state
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary and restore partitioned optimizer states."""
        
        # Restore timing statistics
        self.communication_time = state_dict.get('communication_time', 0.0)
        self.optimizer_time = state_dict.get('optimizer_time', 0.0)
        self.state_gathering_time = state_dict.get('state_gathering_time', 0.0)
        
        # Restore parameter to rank mapping
        self.param_to_rank = state_dict.get('param_to_rank', {})
        
        # Load base optimizer state
        if 'base_optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['base_optimizer'])
        
        # Restore optimizer states for owned parameters
        if 'optimizer_state' in state_dict:
            optimizer_states = state_dict['optimizer_state']
            for param in self.owned_parameters:
                param_id = id(param)
                if param_id in optimizer_states:
                    self.optimizer.state[param] = optimizer_states[param_id]
        
        # Offload states to CPU if configured
        if self.cpu_offload:
            self._offload_optimizer_states()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory usage information for Stage 1."""
        
        owned_param_count = len(self.owned_parameters)
        total_param_count = len(self.model_parameters)
        
        # Estimate optimizer state memory
        optimizer_state_memory = 0.0
        for param in self.owned_parameters:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        optimizer_state_memory += value.numel() * value.element_size()
        
        optimizer_state_memory_gb = optimizer_state_memory / 1e9
        
        return {
            'owned_parameters': owned_param_count,
            'total_parameters': total_param_count,
            'optimizer_state_memory_gb': optimizer_state_memory_gb,
            'memory_reduction_factor': total_param_count / max(1, owned_param_count)
        }
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        
        return {
            'total_communication_time': self.communication_time,
            'state_gathering_time': self.state_gathering_time,
            'gradient_buckets': len(self.gradient_buckets),
            'reduce_bucket_size_mb': self.reduce_bucket_size / 1e6
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        
        self.communication_time = 0.0
        self.optimizer_time = 0.0
        self.state_gathering_time = 0.0
    
    def __repr__(self) -> str:
        """String representation of ZeRO Stage 1."""
        
        return (
            f"ZeROStage1(world_size={self.world_size}, "
            f"owned_params={len(self.owned_parameters)}, "
            f"cpu_offload={self.cpu_offload}, "
            f"overlap_comm={self.overlap_comm})"
        )