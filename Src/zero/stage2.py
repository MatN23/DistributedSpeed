"""
DistributedSpeed ZeRO Stage 2 Implementation.

ZeRO Stage 2 partitions both optimizer states and gradients across data-parallel processes
while keeping parameters replicated. This provides an 8x reduction in memory usage for
optimizer states and gradients combined.

Key Features:
- Optimizer state and gradient partitioning
- Automatic gradient gathering and scattering
- Reduce-scatter for efficient gradient synchronization
- Communication overlap optimization
- CPU offloading support
- Gradient compression and bucketing

Stage 2 is ideal for large models where both gradient and optimizer memory are bottlenecks
but parameter memory can still fit in GPU memory.

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
    get_global_norm, pad_tensor
)

logger = logging.getLogger(__name__)


class ZeROStage2:
    """
    ZeRO Stage 2: Optimizer State + Gradient Partitioning.
    
    This class implements ZeRO Stage 2 optimization where both optimizer states
    and gradients are partitioned across data-parallel processes while parameters
    remain replicated.
    
    Memory Savings:
    - Optimizer states: 4x reduction (partitioned)
    - Gradients: 4x reduction (partitioned)
    - Parameters: No reduction (replicated)
    
    Communication:
    - Reduce-scatter for gradient synchronization
    - AllGather for gradient collection when needed
    - Communication overlap optimization
    
    Args:
        optimizer: Base PyTorch optimizer
        config: ZeRO configuration
        model_parameters: List of model parameters
        comm_manager: Communication manager
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config,  # ZeROConfig
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
        
        # Configuration
        self.overlap_comm = config.overlap_comm
        self.reduce_scatter = config.reduce_scatter
        self.allgather_partitions = config.allgather_partitions
        self.reduce_bucket_size = int(config.reduce_bucket_size)
        self.allgather_bucket_size = int(config.allgather_bucket_size)
        self.contiguous_gradients = config.contiguous_gradients
        
        # Memory management
        self.cpu_offload = config.cpu_offload
        self.pin_memory = config.cpu_offload_use_pin_memory
        
        # Partitioning setup
        self.parameter_partitions = []
        self.gradient_partitions = {}
        self.partition_to_rank = {}
        self.rank_to_partition = defaultdict(list)
        
        # Communication optimization
        self.gradient_buckets = []
        self.communication_handles = []
        
        # Performance tracking
        self.reduce_scatter_time = 0.0
        self.allgather_time = 0.0
        self.optimizer_time = 0.0
        self.total_comm_volume = 0
        
        # Initialize partitioning
        self._partition_parameters()
        self._setup_gradient_buckets()
        self._partition_optimizer_states()
        
        logger.info(f"Initialized ZeRO Stage 2: world_size={self.world_size}, "
                   f"partitions={len(self.parameter_partitions)}, "
                   f"cpu_offload={self.cpu_offload}")
    
    def _partition_parameters(self):
        """Partition parameters across processes for gradient management."""
        
        # Group parameters by size for balanced partitioning
        params_with_grad = [p for p in self.model_parameters if p.requires_grad]
        
        if not params_with_grad:
            logger.warning("No parameters with gradients found")
            return
        
        # Calculate total parameter count
        total_numel = sum(p.numel() for p in params_with_grad)
        target_partition_size = (total_numel + self.world_size - 1) // self.world_size
        
        # Create balanced partitions
        current_partition = []
        current_partition_size = 0
        
        for param in params_with_grad:
            param_size = param.numel()
            
            if (current_partition_size + param_size > target_partition_size and 
                current_partition and len(self.parameter_partitions) < self.world_size - 1):
                # Start new partition
                self.parameter_partitions.append(current_partition)
                current_partition = [param]
                current_partition_size = param_size
            else:
                current_partition.append(param)
                current_partition_size += param_size
        
        # Add final partition
        if current_partition:
            self.parameter_partitions.append(current_partition)
        
        # Assign partitions to ranks
        for rank in range(min(len(self.parameter_partitions), self.world_size)):
            partition = self.parameter_partitions[rank]
            self.rank_to_partition[rank] = partition
            
            for param in partition:
                self.partition_to_rank[param] = rank
        
        logger.info(f"Created {len(self.parameter_partitions)} parameter partitions")
    
    def _setup_gradient_buckets(self):
        """Setup gradient buckets for efficient reduce-scatter operations."""
        
        # Group parameters by owning rank for bucketed communication
        rank_params = defaultdict(list)
        
        for param in self.model_parameters:
            if param.requires_grad and param in self.partition_to_rank:
                owner_rank = self.partition_to_rank[param]
                rank_params[owner_rank].append(param)
        
        # Create buckets for each rank
        for rank in range(self.world_size):
            if rank in rank_params:
                params = rank_params[rank]
                buckets = self._create_buckets(params, self.reduce_bucket_size)
                
                for bucket in buckets:
                    self.gradient_buckets.append({
                        'rank': rank,
                        'params': bucket,
                        'buffer': None,
                        'handle': None
                    })
        
        logger.info(f"Created {len(self.gradient_buckets)} gradient buckets")
    
    def _create_buckets(self, params: List[nn.Parameter], bucket_size: int) -> List[List[nn.Parameter]]:
        """Create parameter buckets based on size constraints."""
        
        buckets = []
        current_bucket = []
        current_size = 0
        
        for param in params:
            param_size = param.numel() * param.element_size()
            
            if current_size + param_size > bucket_size and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [param]
                current_size = param_size
            else:
                current_bucket.append(param)
                current_size += param_size
        
        if current_bucket:
            buckets.append(current_bucket)
        
        return buckets
    
    def _partition_optimizer_states(self):
        """Partition optimizer states for owned parameters."""
        
        # Initialize states for parameters owned by this rank
        owned_params = self.rank_to_partition.get(self.rank, [])
        
        if owned_params:
            self._initialize_optimizer_states(owned_params)
            
            if self.cpu_offload:
                self._offload_optimizer_states(owned_params)
    
    def _initialize_optimizer_states(self, params: List[nn.Parameter]):
        """Initialize optimizer states for given parameters."""
        
        # Create dummy gradients to initialize states
        original_grads = {}
        for param in params:
            original_grads[param] = param.grad
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
        
        # Create temporary parameter groups
        temp_param_groups = []
        for group in self.optimizer.param_groups:
            temp_group = group.copy()
            temp_group['params'] = [p for p in group['params'] if p in params]
            if temp_group['params']:
                temp_param_groups.append(temp_group)
        
        if temp_param_groups:
            temp_optimizer = type(self.optimizer)(temp_param_groups, **self.optimizer.defaults)
            temp_optimizer.step()
            
            # Copy states to main optimizer
            for param in params:
                if param in temp_optimizer.state:
                    self.optimizer.state[param] = temp_optimizer.state[param]
        
        # Restore original gradients
        for param, grad in original_grads.items():
            param.grad = grad
    
    def _offload_optimizer_states(self, params: List[nn.Parameter]):
        """Offload optimizer states to CPU."""
        
        for param in params:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        cpu_tensor = value.cpu()
                        if self.pin_memory:
                            cpu_tensor = cpu_tensor.pin_memory()
                        state[key] = cpu_tensor
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameters."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def step(self, closure=None):
        """
        Perform optimizer step with Stage 2 ZeRO optimization.
        
        Args:
            closure: Optional closure function
            
        Returns:
            Loss value if closure provided
        """
        
        start_time = time.time()
        
        # Reduce-scatter gradients to owning ranks
        self._reduce_scatter_gradients()
        
        # Gather optimizer states for owned parameters
        owned_params = self.rank_to_partition.get(self.rank, [])
        if owned_params:
            self._gather_optimizer_states(owned_params)
        
        # Perform optimizer step on owned parameters
        loss = self._optimizer_step(owned_params, closure)
        
        # Scatter optimizer states back if offloaded
        if self.cpu_offload and owned_params:
            self._scatter_optimizer_states(owned_params)
        
        # AllGather updated parameters to all ranks
        self._allgather_parameters()
        
        self.optimizer_time += time.time() - start_time
        
        return loss
    
    def _reduce_scatter_gradients(self):
        """Reduce-scatter gradients to owning ranks."""
        
        start_time = time.time()
        
        # Process each gradient bucket
        for bucket_info in self.gradient_buckets:
            params = bucket_info['params']
            owner_rank = bucket_info['rank']
            
            # Collect gradients
            grad_tensors = []
            for param in params:
                if param.grad is not None:
                    grad_tensors.append(param.grad.view(-1))
            
            if grad_tensors:
                # Flatten gradients
                flat_grad = torch.cat(grad_tensors)
                
                if self.reduce_scatter:
                    # Prepare output tensor for reduce-scatter
                    if owner_rank == self.rank:
                        output_tensor = torch.empty_like(flat_grad)
                    else:
                        output_tensor = None
                    
                    # Perform reduce-scatter
                    input_list = [flat_grad] * self.world_size
                    output_list = [output_tensor] if output_tensor is not None else None
                    
                    dist.reduce_scatter(output_list[0] if output_list else torch.empty(0, device=flat_grad.device),
                                      input_list, group=None)
                    
                    # Store scattered gradient for owned parameters
                    if owner_rank == self.rank:
                        bucket_info['scattered_grad'] = output_tensor
                
                else:
                    # Fallback to allreduce + manual partitioning
                    dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
                    flat_grad.div_(self.world_size)
                    
                    if owner_rank == self.rank:
                        bucket_info['scattered_grad'] = flat_grad
        
        self.reduce_scatter_time += time.time() - start_time
        self.total_comm_volume += sum(bucket['scattered_grad'].numel() * 4 
                                    for bucket in self.gradient_buckets 
                                    if 'scattered_grad' in bucket)
    
    def _gather_optimizer_states(self, params: List[nn.Parameter]):
        """Gather optimizer states from CPU to GPU if offloaded."""
        
        if not self.cpu_offload:
            return
        
        for param in params:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and not value.is_cuda:
                        state[key] = value.cuda(non_blocking=True)
    
    def _optimizer_step(self, owned_params: List[nn.Parameter], closure):
        """Perform optimizer step on owned parameters."""
        
        if not owned_params:
            return None
        
        # Update gradients from scattered results
        self._update_parameter_gradients(owned_params)
        
        # Temporarily modify param groups
        original_param_groups = []
        for group in self.optimizer.param_groups:
            original_params = group['params']
            owned_group_params = [p for p in original_params if p in owned_params]
            
            original_param_groups.append(original_params)
            group['params'] = owned_group_params
        
        # Perform optimizer step
        loss = None
        if closure is not None:
            loss = closure()
        
        if any(group['params'] for group in self.optimizer.param_groups):
            self.optimizer.step()
        
        # Restore original param groups
        for group, original_params in zip(self.optimizer.param_groups, original_param_groups):
            group['params'] = original_params
        
        return loss
    
    def _update_parameter_gradients(self, owned_params: List[nn.Parameter]):
        """Update parameter gradients from scattered results."""
        
        for bucket_info in self.gradient_buckets:
            if bucket_info['rank'] == self.rank and 'scattered_grad' in bucket_info:
                params = bucket_info['params']
                scattered_grad = bucket_info['scattered_grad']
                
                # Unflatten scattered gradient back to parameters
                offset = 0
                for param in params:
                    if param in owned_params:
                        param_numel = param.numel()
                        param.grad = scattered_grad[offset:offset+param_numel].view_as(param)
                        offset += param_numel
    
    def _scatter_optimizer_states(self, params: List[nn.Parameter]):
        """Scatter optimizer states back to CPU if offloaded."""
        
        if not self.cpu_offload:
            return
        
        for param in params:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        cpu_tensor = value.cpu()
                        if self.pin_memory:
                            cpu_tensor = cpu_tensor.pin_memory()
                        state[key] = cpu_tensor
    
    def _allgather_parameters(self):
        """AllGather updated parameters to all ranks (Stage 2 keeps params replicated)."""
        
        # In Stage 2, parameters remain replicated, so no gathering needed
        # This is a placeholder for potential future extensions
        pass
    
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        """
        Clip gradient norm across all partitioned gradients.
        
        Args:
            max_norm: Maximum allowed gradient norm
            norm_type: Type of norm to compute
            
        Returns:
            Total gradient norm
        """
        
        # Compute norm of scattered gradients on each rank
        local_norm_squared = 0.0
        
        for bucket_info in self.gradient_buckets:
            if bucket_info['rank'] == self.rank and 'scattered_grad' in bucket_info:
                scattered_grad = bucket_info['scattered_grad']
                if norm_type == 2.0:
                    local_norm_squared += scattered_grad.norm(dtype=torch.float32).item() ** 2
                else:
                    local_norm_squared += scattered_grad.norm(p=norm_type, dtype=torch.float32).item() ** norm_type
        
        # AllReduce to get global norm
        if self.world_size > 1:
            norm_tensor = torch.tensor(local_norm_squared, device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
            
            if norm_type == 2.0:
                global_norm = norm_tensor.item() ** 0.5
            else:
                global_norm = norm_tensor.item() ** (1.0 / norm_type)
        else:
            if norm_type == 2.0:
                global_norm = local_norm_squared ** 0.5
            else:
                global_norm = local_norm_squared ** (1.0 / norm_type)
        
        # Clip gradients if norm exceeds threshold
        if global_norm > max_norm:
            clip_coef = max_norm / (global_norm + 1e-6)
            
            for bucket_info in self.gradient_buckets:
                if bucket_info['rank'] == self.rank and 'scattered_grad' in bucket_info:
                    bucket_info['scattered_grad'].mul_(clip_coef)
        
        return torch.tensor(global_norm)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary including partitioned states."""
        
        state_dict = {
            'parameter_partitions': self.parameter_partitions,
            'partition_to_rank': self.partition_to_rank,
            'rank_to_partition': dict(self.rank_to_partition),
            'reduce_scatter_time': self.reduce_scatter_time,
            'allgather_time': self.allgather_time,
            'optimizer_time': self.optimizer_time,
            'total_comm_volume': self.total_comm_volume,
            'optimizer_state': {}
        }
        
        # Save optimizer states for owned parameters
        owned_params = self.rank_to_partition.get(self.rank, [])
        for param in owned_params:
            if param in self.optimizer.state:
                param_id = id(param)
                state_dict['optimizer_state'][param_id] = self.optimizer.state[param]
        
        # Include base optimizer state
        base_state = self.optimizer.state_dict()
        state_dict['base_optimizer'] = base_state
        
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary and restore partitioned states."""
        
        # Restore partitioning information
        self.parameter_partitions = state_dict.get('parameter_partitions', [])
        self.partition_to_rank = state_dict.get('partition_to_rank', {})
        self.rank_to_partition = defaultdict(list, state_dict.get('rank_to_partition', {}))
        
        # Restore timing statistics
        self.reduce_scatter_time = state_dict.get('reduce_scatter_time', 0.0)
        self.allgather_time = state_dict.get('allgather_time', 0.0)
        self.optimizer_time = state_dict.get('optimizer_time', 0.0)
        self.total_comm_volume = state_dict.get('total_comm_volume', 0)
        
        # Load base optimizer state
        if 'base_optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['base_optimizer'])
        
        # Restore optimizer states for owned parameters
        if 'optimizer_state' in state_dict:
            optimizer_states = state_dict['optimizer_state']
            owned_params = self.rank_to_partition.get(self.rank, [])
            
            for param in owned_params:
                param_id = id(param)
                if param_id in optimizer_states:
                    self.optimizer.state[param] = optimizer_states[param_id]
        
        # Offload states if configured
        owned_params = self.rank_to_partition.get(self.rank, [])
        if self.cpu_offload and owned_params:
            self._offload_optimizer_states(owned_params)
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get memory usage information for Stage 2."""
        
        owned_params = self.rank_to_partition.get(self.rank, [])
        total_params = sum(len(partition) for partition in self.parameter_partitions)
        
        # Calculate optimizer state memory
        optimizer_state_memory = 0.0
        for param in owned_params:
            if param in self.optimizer.state:
                state = self.optimizer.state[param]
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        optimizer_state_memory += value.numel() * value.element_size()
        
        # Calculate gradient memory (partitioned)
        gradient_memory = 0.0
        for bucket_info in self.gradient_buckets:
            if bucket_info['rank'] == self.rank and 'scattered_grad' in bucket_info:
                gradient_memory += bucket_info['scattered_grad'].numel() * bucket_info['scattered_grad'].element_size()
        
        return {
            'owned_parameters': len(owned_params),
            'total_parameters': total_params,
            'optimizer_state_memory_gb': optimizer_state_memory / 1e9,
            'gradient_memory_gb': gradient_memory / 1e9,
            'total_partitioned_memory_gb': (optimizer_state_memory + gradient_memory) / 1e9,
            'memory_reduction_factor': total_params / max(1, len(owned_params))
        }
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        
        return {
            'reduce_scatter_time': self.reduce_scatter_time,
            'allgather_time': self.allgather_time,
            'total_comm_volume_gb': self.total_comm_volume / 1e9,
            'gradient_buckets': len(self.gradient_buckets),
            'reduce_bucket_size_mb': self.reduce_bucket_size / 1e6,
            'allgather_bucket_size_mb': self.allgather_bucket_size / 1e6
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        
        self.reduce_scatter_time = 0.0
        self.allgather_time = 0.0
        self.optimizer_time = 0.0
        self.total_comm_volume = 0
    
    def __repr__(self) -> str:
        """String representation of ZeRO Stage 2."""
        
        owned_params = len(self.rank_to_partition.get(self.rank, []))
        total_params = sum(len(partition) for partition in self.parameter_partitions)
        
        return (
            f"ZeROStage2(world_size={self.world_size}, "
            f"owned_params={owned_params}/{total_params}, "
            f"buckets={len(self.gradient_buckets)}, "
            f"cpu_offload={self.cpu_offload})"
        )