"""
DistributedSpeed ZeRO Parameter and Gradient Partitioning.

This module implements parameter and gradient partitioning strategies for ZeRO optimizer stages.
It handles the distribution of parameters and gradients across multiple processes to minimize
memory usage while maintaining mathematical equivalence to non-partitioned training.

Key Components:
- ParameterPartitioner: Partitions model parameters across processes (Stage 3)
- GradientPartitioner: Partitions gradients across processes (Stage 2+)
- Partition utilities for memory-efficient tensor operations
- Communication-aware partitioning strategies

The partitioning strategies ensure balanced memory usage and efficient communication patterns
for optimal distributed training performance.

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import math
import logging
from typing import Dict, List, Optional, Union, Tuple, Iterator, Any
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .utils import (
    get_world_size, get_rank, flatten_dense_tensors_aligned,
    unflatten_dense_tensors, get_global_norm, is_model_parallel_parameter
)

logger = logging.getLogger(__name__)


@dataclass
class PartitionInfo:
    """Information about parameter/gradient partition."""
    rank: int
    start_idx: int
    end_idx: int
    numel: int
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, ...]
    partition_size: int


class BasePartitioner:
    """
    Base class for parameter and gradient partitioning.
    
    Provides common functionality for partitioning tensors across distributed processes
    including balanced partitioning strategies, communication utilities, and memory
    management helpers.
    """
    
    def __init__(
        self,
        parameters: List[nn.Parameter],
        world_size: int,
        rank: int,
        config
    ):
        self.parameters = list(parameters)
        self.world_size = world_size
        self.rank = rank
        self.config = config
        
        # Device management
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        self.cpu_offload = getattr(config, 'cpu_offload', False)
        self.pin_memory = getattr(config, 'cpu_offload_use_pin_memory', True)
        
        # Partitioning strategy
        self.partition_strategy = getattr(config, 'partition_strategy', 'balanced')
        self.alignment = getattr(config, 'memory_alignment', 64)  # 64-byte alignment
        
        # Communication settings
        self.bucket_size = getattr(config, 'allgather_bucket_size', 2e8)
        self.overlap_comm = getattr(config, 'overlap_comm', True)
        
        # Internal state
        self.partition_info = {}
        self.flattened_tensors = {}
        self.tensor_metadata = {}
        self.communication_handles = {}
        
        # Statistics
        self.total_params = sum(p.numel() for p in self.parameters)
        self.memory_usage = 0
        
        logger.info(f"Initialized {self.__class__.__name__}: "
                   f"world_size={world_size}, rank={rank}, "
                   f"total_params={self.total_params}")
    
    def _calculate_partition_sizes(self, total_elements: int) -> List[int]:
        """
        Calculate partition sizes for balanced distribution.
        
        Args:
            total_elements: Total number of elements to partition
            
        Returns:
            List of partition sizes for each rank
        """
        
        base_size = total_elements // self.world_size
        remainder = total_elements % self.world_size
        
        partition_sizes = []
        for rank in range(self.world_size):
            size = base_size + (1 if rank < remainder else 0)
            # Apply alignment
            if size > 0:
                size = ((size + self.alignment - 1) // self.alignment) * self.alignment
            partition_sizes.append(size)
        
        return partition_sizes
    
    def _get_partition_bounds(self, rank: int, partition_sizes: List[int]) -> Tuple[int, int]:
        """
        Get start and end indices for a specific rank's partition.
        
        Args:
            rank: Process rank
            partition_sizes: List of partition sizes
            
        Returns:
            Tuple of (start_index, end_index)
        """
        
        start_idx = sum(partition_sizes[:rank])
        end_idx = start_idx + partition_sizes[rank]
        
        return start_idx, end_idx
    
    def _align_tensor_size(self, size: int) -> int:
        """Align tensor size to memory alignment boundary."""
        return ((size + self.alignment - 1) // self.alignment) * self.alignment
    
    def _create_partition_tensor(
        self, 
        size: int, 
        dtype: torch.dtype, 
        device: torch.device,
        requires_grad: bool = False,
        pin_memory: bool = False
    ) -> torch.Tensor:
        """
        Create a tensor for holding partition data.
        
        Args:
            size: Tensor size in elements
            dtype: Tensor data type
            device: Target device
            requires_grad: Whether tensor requires gradients
            pin_memory: Whether to pin memory for CPU tensors
            
        Returns:
            Allocated tensor
        """
        
        if device.type == 'cpu' and pin_memory and self.pin_memory:
            tensor = torch.empty(size, dtype=dtype, pin_memory=True)
            if requires_grad:
                tensor.requires_grad_(True)
        else:
            tensor = torch.empty(size, dtype=dtype, device=device)
            if requires_grad:
                tensor.requires_grad_(True)
        
        return tensor


class ParameterPartitioner(BasePartitioner):
    """
    Parameter partitioner for ZeRO Stage 3.
    
    Handles partitioning of model parameters across distributed processes.
    Each process owns a partition of parameters and can gather full parameters
    when needed for computation.
    
    Features:
    - Balanced parameter distribution
    - Efficient all-gather and scatter operations
    - CPU offloading support
    - Memory-aligned partitioning
    - Parameter reconstruction and gathering
    """
    
    def __init__(
        self,
        parameters: List[nn.Parameter],
        world_size: int,
        rank: int,
        config
    ):
        super().__init__(parameters, world_size, rank, config)
        
        # Parameter-specific settings
        self.offload_params = getattr(config, 'cpu_offload_params', False)
        self.prefetch_bucket_size = getattr(config, 'param_prefetch_bucket_size', 2e7)
        
        # Partitioning state
        self.param_to_partition_map = {}
        self.partition_to_params_map = defaultdict(list)
        self.original_shapes = {}
        self.original_devices = {}
        
        # Flattened parameter storage
        self.flattened_params = None
        self.param_partitions = {}
        self.gathered_params = {}
        
        # Communication buffers
        self.gather_handles = {}
        self.scatter_handles = {}
        self.prefetch_handles = {}
        
        # Initialize partitioning
        self._initialize_partitions()
        
        logger.info(f"ParameterPartitioner initialized: "
                   f"partitions={len(self.param_partitions)}, "
                   f"offload={self.offload_params}")
    
    def _initialize_partitions(self):
        """Initialize parameter partitions."""
        
        # Store original parameter information
        for i, param in enumerate(self.parameters):
            self.original_shapes[i] = param.shape
            self.original_devices[i] = param.device
            self.tensor_metadata[i] = {
                'dtype': param.dtype,
                'requires_grad': param.requires_grad,
                'numel': param.numel()
            }
        
        # Flatten all parameters
        self._flatten_parameters()
        
        # Create partitions
        self._create_parameter_partitions()
        
        # Initialize partition ownership
        self._setup_partition_ownership()
    
    def _flatten_parameters(self):
        """Flatten all parameters into a single tensor."""
        
        if not self.parameters:
            self.flattened_params = torch.empty(0)
            return
        
        # Group parameters by dtype and device
        param_groups = defaultdict(list)
        for i, param in enumerate(self.parameters):
            key = (param.dtype, param.device)
            param_groups[key].append((i, param))
        
        # Flatten each group
        self.flattened_tensors = {}
        param_offset = 0
        
        for (dtype, device), param_list in param_groups.items():
            # Extract tensors
            tensors = [param.data for _, param in param_list]
            
            # Flatten and concatenate
            flattened = flatten_dense_tensors_aligned(tensors, self.alignment)
            
            # Store mapping
            for i, (param_idx, param) in enumerate(param_list):
                start_idx = param_offset
                end_idx = start_idx + param.numel()
                
                self.param_to_partition_map[param_idx] = {
                    'group_key': (dtype, device),
                    'start': start_idx,
                    'end': end_idx,
                    'local_start': sum(p.numel() for _, p in param_list[:i]),
                    'local_end': sum(p.numel() for _, p in param_list[:i+1])
                }
                
                param_offset += param.numel()
            
            self.flattened_tensors[(dtype, device)] = flattened
        
        # Create single flattened tensor if possible
        if len(self.flattened_tensors) == 1:
            self.flattened_params = list(self.flattened_tensors.values())[0]
        else:
            # Multiple dtype/device groups - keep separate
            total_elements = sum(t.numel() for t in self.flattened_tensors.values())
            logger.warning(f"Multiple parameter groups detected: {len(self.flattened_tensors)}")
    
    def _create_parameter_partitions(self):
        """Create parameter partitions for distribution."""
        
        for group_key, flattened in self.flattened_tensors.items():
            dtype, device = group_key
            total_elements = flattened.numel()
            
            # Calculate partition sizes
            partition_sizes = self._calculate_partition_sizes(total_elements)
            
            # Create partitions for each rank
            for rank in range(self.world_size):
                start_idx, end_idx = self._get_partition_bounds(rank, partition_sizes)
                partition_size = end_idx - start_idx
                
                if partition_size > 0:
                    # Create partition info
                    partition_info = PartitionInfo(
                        rank=rank,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        numel=partition_size,
                        dtype=dtype,
                        device=device if not self.offload_params else torch.device('cpu'),
                        shape=(partition_size,),
                        partition_size=partition_size
                    )
                    
                    partition_key = (group_key, rank)
                    self.partition_info[partition_key] = partition_info
                    
                    # Create actual partition tensor
                    if rank == self.rank:
                        # Own partition - extract from flattened tensor
                        if start_idx < total_elements:
                            actual_end = min(end_idx, total_elements)
                            partition_data = flattened[start_idx:actual_end].clone()
                        else:
                            partition_data = torch.empty(0, dtype=dtype, device=device)
                        
                        # Move to target device
                        if self.offload_params:
                            partition_data = partition_data.cpu()
                            if self.pin_memory:
                                partition_data = partition_data.pin_memory()
                        
                        self.param_partitions[partition_key] = partition_data
    
    def _setup_partition_ownership(self):
        """Setup which parameters each process owns."""
        
        for param_idx, mapping in self.param_to_partition_map.items():
            group_key = mapping['group_key']
            param_start = mapping['start']
            param_end = mapping['end']
            
            # Find which partition(s) contain this parameter
            for rank in range(self.world_size):
                partition_key = (group_key, rank)
                if partition_key in self.partition_info:
                    partition_info = self.partition_info[partition_key]
                    
                    # Check overlap
                    if (param_start < partition_info.end_idx and 
                        param_end > partition_info.start_idx):
                        
                        self.partition_to_params_map[partition_key].append(param_idx)
    
    def gather_params(
        self, 
        param_indices: Optional[List[int]] = None,
        async_op: bool = False
    ) -> Union[Dict[int, torch.Tensor], torch.distributed.Work]:
        """
        Gather parameters from all partitions.
        
        Args:
            param_indices: Specific parameter indices to gather (None for all)
            async_op: Whether to perform asynchronous operation
            
        Returns:
            Dictionary mapping parameter indices to gathered tensors,
            or Work handle if async_op=True
        """
        
        if param_indices is None:
            param_indices = list(range(len(self.parameters)))
        
        gathered = {}
        work_handles = []
        
        # Group parameters by their tensor groups
        param_groups = defaultdict(list)
        for param_idx in param_indices:
            if param_idx in self.param_to_partition_map:
                mapping = self.param_to_partition_map[param_idx]
                group_key = mapping['group_key']
                param_groups[group_key].append(param_idx)
        
        for group_key, group_param_indices in param_groups.items():
            dtype, device = group_key
            
            if group_key in self.flattened_tensors:
                total_elements = self.flattened_tensors[group_key].numel()
                
                # Create buffer for gathered tensor
                gathered_buffer = torch.empty(
                    total_elements, 
                    dtype=dtype, 
                    device=device
                )
                
                # Gather tensor pieces from all ranks
                partition_tensors = []
                for rank in range(self.world_size):
                    partition_key = (group_key, rank)
                    if partition_key in self.param_partitions:
                        tensor = self.param_partitions[partition_key]
                        # Move back to device if offloaded
                        if tensor.device != device:
                            tensor = tensor.to(device, non_blocking=True)
                        partition_tensors.append(tensor)
                    else:
                        # Empty partition
                        partition_tensors.append(torch.empty(0, dtype=dtype, device=device))
                
                # All-gather operation
                if len(partition_tensors) > 0:
                    if async_op:
                        handle = dist.all_gather(
                            partition_tensors,
                            self.param_partitions.get((group_key, self.rank), 
                                                    torch.empty(0, dtype=dtype, device=device)),
                            async_op=True
                        )
                        work_handles.append(handle)
                    else:
                        dist.all_gather(
                            partition_tensors,
                            self.param_partitions.get((group_key, self.rank),
                                                    torch.empty(0, dtype=dtype, device=device))
                        )
                        
                        # Reconstruct full tensor
                        gathered_buffer = torch.cat(partition_tensors, dim=0)
                        
                        # Extract individual parameters
                        for param_idx in group_param_indices:
                            mapping = self.param_to_partition_map[param_idx]
                            start_idx = mapping['local_start']
                            end_idx = mapping['local_end']
                            
                            param_data = gathered_buffer[start_idx:end_idx]
                            original_shape = self.original_shapes[param_idx]
                            
                            gathered[param_idx] = param_data.view(original_shape)
        
        if async_op:
            # Return work handle for async operations
            return work_handles[0] if len(work_handles) == 1 else work_handles
        
        return gathered
    
    def scatter_params(
        self, 
        params_dict: Dict[int, torch.Tensor],
        async_op: bool = False
    ) -> Optional[torch.distributed.Work]:
        """
        Scatter parameters back to partitions.
        
        Args:
            params_dict: Dictionary of parameter indices to tensors
            async_op: Whether to perform asynchronous operation
            
        Returns:
            Work handle if async_op=True, otherwise None
        """
        
        work_handles = []
        
        # Group parameters by tensor groups
        param_groups = defaultdict(list)
        for param_idx, tensor in params_dict.items():
            if param_idx in self.param_to_partition_map:
                mapping = self.param_to_partition_map[param_idx]
                group_key = mapping['group_key']
                param_groups[group_key].append((param_idx, tensor))
        
        for group_key, group_params in param_groups.items():
            dtype, device = group_key
            
            # Reconstruct flattened tensor
            flattened_parts = []
            for param_idx, tensor in sorted(group_params):
                flattened_parts.append(tensor.flatten())
            
            if flattened_parts:
                flattened_tensor = torch.cat(flattened_parts, dim=0)
                
                # Calculate partitions
                total_elements = flattened_tensor.numel()
                partition_sizes = self._calculate_partition_sizes(total_elements)
                
                # Create partition list for scatter
                partition_list = []
                for rank in range(self.world_size):
                    start_idx, end_idx = self._get_partition_bounds(rank, partition_sizes)
                    if start_idx < total_elements:
                        actual_end = min(end_idx, total_elements)
                        partition = flattened_tensor[start_idx:actual_end].contiguous()
                    else:
                        partition = torch.empty(0, dtype=dtype, device=device)
                    
                    partition_list.append(partition)
                
                # Scatter operation
                output_tensor = torch.empty_like(partition_list[self.rank])
                
                if async_op:
                    handle = dist.scatter(
                        output_tensor,
                        partition_list if self.rank == 0 else None,
                        src=0,
                        async_op=True
                    )
                    work_handles.append(handle)
                else:
                    dist.scatter(
                        output_tensor,
                        partition_list if self.rank == 0 else None,
                        src=0
                    )
                    
                    # Update local partition
                    partition_key = (group_key, self.rank)
                    if self.offload_params:
                        output_tensor = output_tensor.cpu()
                        if self.pin_memory:
                            output_tensor = output_tensor.pin_memory()
                    
                    self.param_partitions[partition_key] = output_tensor
        
        if async_op:
            return work_handles[0] if len(work_handles) == 1 else work_handles
        
        return None
    
    def get_param_partition(self, param_idx: int) -> torch.Tensor:
        """
        Get the local partition for a specific parameter.
        
        Args:
            param_idx: Parameter index
            
        Returns:
            Local partition tensor for the parameter
        """
        
        if param_idx not in self.param_to_partition_map:
            raise ValueError(f"Parameter {param_idx} not found in partition map")
        
        mapping = self.param_to_partition_map[param_idx]
        group_key = mapping['group_key']
        
        # Find which partition contains this parameter
        for rank in range(self.world_size):
            partition_key = (group_key, rank)
            if (partition_key in self.partition_to_params_map and 
                param_idx in self.partition_to_params_map[partition_key]):
                
                if rank == self.rank and partition_key in self.param_partitions:
                    return self.param_partitions[partition_key]
        
        # Parameter not owned by this process
        return torch.empty(0)
    
    def update_param_partition(self, param_idx: int, data: torch.Tensor):
        """
        Update local partition for a specific parameter.
        
        Args:
            param_idx: Parameter index
            data: Updated parameter data
        """
        
        if param_idx not in self.param_to_partition_map:
            raise ValueError(f"Parameter {param_idx} not found in partition map")
        
        mapping = self.param_to_partition_map[param_idx]
        group_key = mapping['group_key']
        
        # Update the partition that owns this parameter
        partition_key = (group_key, self.rank)
        if partition_key in self.param_partitions:
            # Extract relevant portion of data
            local_start = mapping['local_start']
            local_end = mapping['local_end']
            
            param_data = data.flatten()[local_start:local_end]
            
            if self.offload_params:
                param_data = param_data.cpu()
                if self.pin_memory:
                    param_data = param_data.pin_memory()
            
            self.param_partitions[partition_key] = param_data
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        
        gpu_memory = 0
        cpu_memory = 0
        
        for partition in self.param_partitions.values():
            memory_bytes = partition.numel() * partition.element_size()
            if partition.device.type == 'cuda':
                gpu_memory += memory_bytes
            else:
                cpu_memory += memory_bytes
        
        return {
            'gpu_memory_gb': gpu_memory / 1e9,
            'cpu_memory_gb': cpu_memory / 1e9,
            'total_memory_gb': (gpu_memory + cpu_memory) / 1e9
        }


class GradientPartitioner(BasePartitioner):
    """
    Gradient partitioner for ZeRO Stage 2+.
    
    Handles partitioning of gradients across distributed processes.
    Each process accumulates gradients for a partition of parameters
    and performs all-reduce only on owned gradients.
    
    Features:
    - Gradient bucketing and reduction
    - Overlapped communication
    - CPU offloading for gradients
    - Gradient compression support
    - Memory-efficient gradient handling
    """
    
    def __init__(
        self,
        parameters: List[nn.Parameter],
        world_size: int,
        rank: int,
        config
    ):
        super().__init__(parameters, world_size, rank, config)
        
        # Gradient-specific settings
        self.offload_gradients = getattr(config, 'cpu_offload_gradients', False)
        self.gradient_compression = getattr(config, 'gradient_compression', None)
        self.gradient_clipping = getattr(config, 'max_grad_norm', 0.0)
        
        # Bucketing for communication
        self.bucket_size_bytes = int(getattr(config, 'reduce_bucket_size', 2e8))
        self.buckets = []
        self.bucket_to_rank = {}
        
        # Gradient storage
        self.grad_partitions = {}
        self.grad_buffers = {}
        self.reduction_handles = {}
        
        # Synchronization state
        self.gradients_reduced = False
        self.pending_reductions = set()
        
        # Initialize gradient partitioning
        self._initialize_gradient_buckets()
        self._create_gradient_partitions()
        
        logger.info(f"GradientPartitioner initialized: "
                   f"buckets={len(self.buckets)}, "
                   f"offload={self.offload_gradients}")
    
    def _initialize_gradient_buckets(self):
        """Initialize gradient buckets for efficient communication."""
        
        # Group parameters by size for balanced bucketing
        param_info = []
        for i, param in enumerate(self.parameters):
            param_info.append({
                'index': i,
                'numel': param.numel(),
                'dtype': param.dtype,
                'device': param.device,
                'bytes': param.numel() * param.element_size()
            })
        
        # Sort by size for better packing
        param_info.sort(key=lambda x: x['bytes'], reverse=True)
        
        # Create buckets
        current_bucket = []
        current_bucket_bytes = 0
        
        for param in param_info:
            if (current_bucket_bytes + param['bytes'] > self.bucket_size_bytes and 
                current_bucket):
                # Finalize current bucket
                self._finalize_bucket(current_bucket)
                current_bucket = []
                current_bucket_bytes = 0
            
            current_bucket.append(param)
            current_bucket_bytes += param['bytes']
        
        # Finalize last bucket
        if current_bucket:
            self._finalize_bucket(current_bucket)
        
        logger.info(f"Created {len(self.buckets)} gradient buckets")
    
    def _finalize_bucket(self, param_list: List[Dict]):
        """Finalize a gradient bucket."""
        
        bucket_id = len(self.buckets)
        
        # Assign bucket to rank based on round-robin
        assigned_rank = bucket_id % self.world_size
        
        bucket_info = {
            'id': bucket_id,
            'assigned_rank': assigned_rank,
            'parameters': param_list,
            'total_bytes': sum(p['bytes'] for p in param_list),
            'dtype': param_list[0]['dtype'],  # Assume same dtype in bucket
            'device': param_list[0]['device']  # Assume same device in bucket
        }
        
        self.buckets.append(bucket_info)
        self.bucket_to_rank[bucket_id] = assigned_rank
        
        # Track which parameters belong to which buckets
        for param in param_list:
            param_idx = param['index']
            if param_idx not in self.param_to_partition_map:
                self.param_to_partition_map[param_idx] = {}
            self.param_to_partition_map[param_idx]['bucket_id'] = bucket_id
    
    def _create_gradient_partitions(self):
        """Create gradient partitions for owned buckets."""
        
        for bucket in self.buckets:
            bucket_id = bucket['id']
            assigned_rank = bucket['assigned_rank']
            
            if assigned_rank == self.rank:
                # This process owns this bucket
                total_elements = sum(p['numel'] for p in bucket['parameters'])
                dtype = bucket['dtype']
                device = bucket['device']
                
                if self.offload_gradients:
                    device = torch.device('cpu')
                
                # Create partition buffer
                partition_buffer = self._create_partition_tensor(
                    total_elements,
                    dtype,
                    device,
                    requires_grad=False,
                    pin_memory=self.offload_gradients
                )
                
                self.grad_partitions[bucket_id] = partition_buffer
                
                # Create individual gradient views
                offset = 0
                for param_info in bucket['parameters']:
                    param_idx = param_info['index']
                    param_numel = param_info['numel']
                    
                    # Create view into partition buffer
                    grad_view = partition_buffer[offset:offset + param_numel]
                    self.grad_buffers[param_idx] = grad_view
                    
                    offset += param_numel
    
    def accumulate_gradients(self, param_gradients: Dict[int, torch.Tensor]):
        """
        Accumulate gradients into partitioned buffers.
        
        Args:
            param_gradients: Dictionary mapping parameter indices to gradient tensors
        """
        
        for param_idx, grad in param_gradients.items():
            if param_idx in self.param_to_partition_map:
                bucket_id = self.param_to_partition_map[param_idx]['bucket_id']
                assigned_rank = self.bucket_to_rank[bucket_id]
                
                if assigned_rank == self.rank and param_idx in self.grad_buffers:
                    # Accumulate into local buffer
                    grad_buffer = self.grad_buffers[param_idx]
                    flattened_grad = grad.flatten()
                    
                    if self.offload_gradients and grad_buffer.device != flattened_grad.device:
                        flattened_grad = flattened_grad.to(grad_buffer.device, non_blocking=True)
                    
                    grad_buffer.add_(flattened_grad)
    
    def reduce_gradients(self, async_op: bool = True) -> List[torch.distributed.Work]:
        """
        Perform all-reduce on gradient partitions.
        
        Args:
            async_op: Whether to perform asynchronous operations
            
        Returns:
            List of work handles for async operations
        """
        
        if self.gradients_reduced:
            return []
        
        work_handles = []
        
        # All-reduce owned gradient partitions
        for bucket_id, grad_partition in self.grad_partitions.items():
            if bucket_id not in self.pending_reductions:
                # Apply gradient compression if configured
                if self.gradient_compression:
                    grad_partition = self._compress_gradient(grad_partition)
                
                # All-reduce operation
                if async_op:
                    handle = dist.all_reduce(grad_partition, async_op=True)
                    work_handles.append(handle)
                    self.reduction_handles[bucket_id] = handle
                else:
                    dist.all_reduce(grad_partition)
                
                self.pending_reductions.add(bucket_id)
        
        if not async_op:
            self._finalize_gradient_reduction()
        
        return work_handles
    
    def _compress_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply gradient compression if configured."""
        
        if self.gradient_compression == 'fp16':
            return gradient.half()
        elif self.gradient_compression == 'bf16':
            return gradient.bfloat16()
        elif self.gradient_compression == 'quantize_8bit':
            return self._quantize_gradient(gradient, bits=8)
        
        return gradient
    
    def _quantize_gradient(self, gradient: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Quantize gradient to specified bit width."""
        
        if bits == 8:
            # Simple 8-bit quantization
            grad_min = gradient.min()
            grad_max = gradient.max()
            
            if grad_max == grad_min:
                return torch.zeros_like(gradient, dtype=torch.uint8)
            
            scale = (grad_max - grad_min) / 255.0
            quantized = ((gradient - grad_min) / scale).round().clamp(0, 255).byte()
            
            # Store scale and min for dequantization
            # In practice, these would be communicated separately
            return quantized.float() * scale + grad_min
        
        return gradient
    
    def _finalize_gradient_reduction(self):
        """Finalize gradient reduction by scaling and distributing."""
        
        # Scale gradients by world size (for averaging)
        for grad_partition in self.grad_partitions.values():
            grad_partition.div_(self.world_size)
        
        # Mark gradients as reduced
        self.gradients_reduced = True
        self.pending_reductions.clear()
    
    def wait_for_reductions(self):
        """Wait for all asynchronous gradient reductions to complete."""
        
        for bucket_id, handle in self.reduction_handles.items():
            handle.wait()
        
        self.reduction_handles.clear()
        self._finalize_gradient_reduction()
    
    def get_reduced_gradients(self) -> Dict[int, torch.Tensor]:
        """
        Get reduced gradients for all parameters.
        
        Returns:
            Dictionary mapping parameter indices to reduced gradient tensors
        """
        
        if not self.gradients_reduced:
            raise RuntimeError("Gradients not reduced. Call reduce_gradients() first.")
        
        reduced_grads = {}
        
        for param_idx in range(len(self.parameters)):
            if param_idx in self.param_to_partition_map:
                bucket_id = self.param_to_partition_map[param_idx]['bucket_id']
                assigned_rank = self.bucket_to_rank[bucket_id]
                
                if assigned_rank == self.rank and param_idx in self.grad_buffers:
                    # Get gradient from local buffer
                    grad_buffer = self.grad_buffers[param_idx]
                    param_shape = self.parameters[param_idx].shape
                    
                    # Reshape back to original parameter shape
                    reduced_grad = grad_buffer.view(param_shape)
                    
                    # Move back to parameter device if offloaded
                    param_device = self.parameters[param_idx].device
                    if reduced_grad.device != param_device:
                        reduced_grad = reduced_grad.to(param_device, non_blocking=True)
                    
                    reduced_grads[param_idx] = reduced_grad
        
        return reduced_grads
    
    def broadcast_gradients(self, gradients: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Broadcast reduced gradients to all processes.
        
        Args:
            gradients: Reduced gradients from owning processes
            
        Returns:
            Dictionary with gradients for all parameters
        """
        
        all_gradients = {}
        
        # Process each bucket
        for bucket in self.buckets:
            bucket_id = bucket['id']
            assigned_rank = bucket['assigned_rank']
            
            if assigned_rank == self.rank:
                # This process owns the bucket - prepare broadcast data
                bucket_grads = []
                for param_info in bucket['parameters']:
                    param_idx = param_info['index']
                    if param_idx in gradients:
                        bucket_grads.append(gradients[param_idx].flatten())
                    else:
                        # Zero gradient
                        param_numel = param_info['numel']
                        dtype = param_info['dtype']
                        device = param_info['device']
                        bucket_grads.append(torch.zeros(param_numel, dtype=dtype, device=device))
                
                if bucket_grads:
                    broadcast_tensor = torch.cat(bucket_grads, dim=0)
                else:
                    broadcast_tensor = torch.empty(0)
            else:
                # Receiving process - create buffer
                total_elements = sum(p['numel'] for p in bucket['parameters'])
                dtype = bucket['dtype']
                device = bucket['device']
                broadcast_tensor = torch.empty(total_elements, dtype=dtype, device=device)
            
            # Broadcast
            dist.broadcast(broadcast_tensor, src=assigned_rank)
            
            # Unpack gradients for receiving processes
            if assigned_rank != self.rank:
                offset = 0
                for param_info in bucket['parameters']:
                    param_idx = param_info['index']
                    param_numel = param_info['numel']
                    param_shape = self.parameters[param_idx].shape
                    
                    grad_data = broadcast_tensor[offset:offset + param_numel]
                    all_gradients[param_idx] = grad_data.view(param_shape)
                    
                    offset += param_numel
        
        # Combine with local gradients
        all_gradients.update(gradients)
        
        return all_gradients
    
    def zero_gradients(self):
        """Zero all gradient partitions."""
        
        for grad_partition in self.grad_partitions.values():
            grad_partition.zero_()
        
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.zero_()
        
        self.gradients_reduced = False
        self.pending_reductions.clear()
    
    def clip_gradients(self, max_norm: float) -> float:
        """
        Clip gradients with ZeRO-aware norm computation.
        
        Args:
            max_norm: Maximum allowed gradient norm
            
        Returns:
            Total gradient norm before clipping
        """
        
        if not self.gradients_reduced:
            raise RuntimeError("Gradients must be reduced before clipping")
        
        # Compute total norm across all partitions
        total_norm_squared = 0.0
        
        for grad_partition in self.grad_partitions.values():
            partition_norm_squared = grad_partition.norm(dtype=torch.float32) ** 2
            total_norm_squared += partition_norm_squared.item()
        
        # All-reduce to get global norm
        total_norm_tensor = torch.tensor(total_norm_squared, device=self.device)
        dist.all_reduce(total_norm_tensor)
        total_norm = total_norm_tensor.sqrt()
        
        # Apply clipping
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            
            for grad_partition in self.grad_partitions.values():
                grad_partition.mul_(clip_coef)
        
        return total_norm.item()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        
        gpu_memory = 0
        cpu_memory = 0
        
        for grad_partition in self.grad_partitions.values():
            memory_bytes = grad_partition.numel() * grad_partition.element_size()
            if grad_partition.device.type == 'cuda':
                gpu_memory += memory_bytes
            else:
                cpu_memory += memory_bytes
        
        return {
            'gpu_memory_gb': gpu_memory / 1e9,
            'cpu_memory_gb': cpu_memory / 1e9,
            'total_memory_gb': (gpu_memory + cpu_memory) / 1e9
        }


class PartitionManager:
    """
    Manages both parameter and gradient partitioning for ZeRO stages.
    
    Provides a unified interface for handling partitioned parameters and gradients,
    coordinating between parameter gathering and gradient reduction operations.
    """
    
    def __init__(
        self,
        parameters: List[nn.Parameter],
        world_size: int,
        rank: int,
        config
    ):
        self.parameters = parameters
        self.world_size = world_size
        self.rank = rank
        self.config = config
        
        # Initialize partitioners based on ZeRO stage
        self.param_partitioner = None
        self.grad_partitioner = None
        
        if config.stage >= 3:
            self.param_partitioner = ParameterPartitioner(
                parameters, world_size, rank, config
            )
        
        if config.stage >= 2:
            self.grad_partitioner = GradientPartitioner(
                parameters, world_size, rank, config
            )
        
        # Coordination state
        self.parameters_gathered = False
        self.gradients_reduced = False
        
        # Performance tracking
        self.gather_time = 0.0
        self.reduce_time = 0.0
        
        logger.info(f"PartitionManager initialized for ZeRO stage {config.stage}")
    
    def gather_parameters(self, param_indices: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """Gather parameters using parameter partitioner."""
        
        if self.param_partitioner is None:
            # No partitioning - return original parameters
            result = {}
            indices = param_indices or range(len(self.parameters))
            for i in indices:
                result[i] = self.parameters[i]
            return result
        
        start_time = time.time()
        gathered = self.param_partitioner.gather_params(param_indices)
        self.gather_time += time.time() - start_time
        
        self.parameters_gathered = True
        return gathered
    
    def scatter_parameters(self, params_dict: Dict[int, torch.Tensor]):
        """Scatter parameters using parameter partitioner."""
        
        if self.param_partitioner is not None:
            self.param_partitioner.scatter_params(params_dict)
        
        self.parameters_gathered = False
    
    def reduce_gradients(self, gradients: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Reduce gradients using gradient partitioner."""
        
        if self.grad_partitioner is None:
            # No partitioning - return original gradients
            return gradients
        
        start_time = time.time()
        
        # Accumulate gradients
        self.grad_partitioner.accumulate_gradients(gradients)
        
        # Reduce gradients
        work_handles = self.grad_partitioner.reduce_gradients(async_op=False)
        
        # Get reduced gradients
        reduced_grads = self.grad_partitioner.get_reduced_gradients()
        
        # Broadcast to all processes
        all_gradients = self.grad_partitioner.broadcast_gradients(reduced_grads)
        
        self.reduce_time += time.time() - start_time
        self.gradients_reduced = True
        
        return all_gradients
    
    def zero_gradients(self):
        """Zero all gradients."""
        
        if self.grad_partitioner is not None:
            self.grad_partitioner.zero_gradients()
        
        self.gradients_reduced = False
    
    def clip_gradients(self, max_norm: float) -> float:
        """Clip gradients with appropriate partitioning handling."""
        
        if self.grad_partitioner is not None:
            return self.grad_partitioner.clip_gradients(max_norm)
        else:
            # Standard gradient clipping
            parameters = [p for p in self.parameters if p.grad is not None]
            from .utils import clip_grad_norm_
            return clip_grad_norm_(parameters, max_norm)
    
    def get_memory_info(self) -> Dict[str, Dict[str, float]]:
        """Get memory usage information for all partitioners."""
        
        info = {}
        
        if self.param_partitioner is not None:
            info['parameters'] = self.param_partitioner.get_memory_usage()
        
        if self.grad_partitioner is not None:
            info['gradients'] = self.grad_partitioner.get_memory_usage()
        
        return info
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        
        return {
            'total_gather_time': self.gather_time,
            'total_reduce_time': self.reduce_time,
            'avg_gather_time': self.gather_time / max(1, self.parameters_gathered),
            'avg_reduce_time': self.reduce_time / max(1, self.gradients_reduced)
        }
    
    @contextmanager
    def parameter_context(self, param_indices: Optional[List[int]] = None):
        """
        Context manager for temporary parameter gathering.
        
        Usage:
            with partition_manager.parameter_context([0, 1, 2]) as params:
                # Use gathered parameters
                output = model(input, params=params)
        """
        
        gathered_params = self.gather_parameters(param_indices)
        try:
            yield gathered_params
        finally:
            if self.param_partitioner is not None:
                self.scatter_parameters(gathered_params)


class PartitionUtils:
    """Utility functions for partitioning operations."""
    
    @staticmethod
    def calculate_memory_savings(
        num_parameters: int,
        world_size: int,
        stage: int,
        optimizer_type: str = 'adam'
    ) -> Dict[str, float]:
        """
        Calculate memory savings from ZeRO partitioning.
        
        Args:
            num_parameters: Number of model parameters
            world_size: Number of distributed processes
            stage: ZeRO stage (1, 2, or 3)
            optimizer_type: Type of optimizer ('sgd', 'adam', etc.)
            
        Returns:
            Dictionary with memory usage and savings
        """
        
        # Base memory (4 bytes per FP32 parameter)
        param_memory_gb = num_parameters * 4 / 1e9
        
        # Optimizer state memory
        if optimizer_type.lower() == 'adam':
            optimizer_memory_gb = param_memory_gb * 2  # momentum + variance
        elif optimizer_type.lower() == 'sgd':
            optimizer_memory_gb = param_memory_gb  # momentum only
        else:
            optimizer_memory_gb = param_memory_gb * 2  # assume Adam-like
        
        # Gradient memory
        grad_memory_gb = param_memory_gb
        
        # Calculate memory usage for each stage
        baseline_total = param_memory_gb + grad_memory_gb + optimizer_memory_gb
        
        if stage == 1:
            # Only optimizer states partitioned
            stage_memory = param_memory_gb + grad_memory_gb + (optimizer_memory_gb / world_size)
        elif stage == 2:
            # Optimizer states and gradients partitioned
            stage_memory = param_memory_gb + (grad_memory_gb / world_size) + (optimizer_memory_gb / world_size)
        elif stage == 3:
            # Everything partitioned
            stage_memory = (param_memory_gb / world_size) + (grad_memory_gb / world_size) + (optimizer_memory_gb / world_size)
        else:
            stage_memory = baseline_total
        
        memory_savings = baseline_total - stage_memory
        savings_ratio = memory_savings / baseline_total if baseline_total > 0 else 0
        
        return {
            'baseline_memory_gb': baseline_total,
            'stage_memory_gb': stage_memory,
            'memory_savings_gb': memory_savings,
            'savings_ratio': savings_ratio,
            'memory_per_gpu_gb': stage_memory
        }
    
    @staticmethod
    def estimate_communication_overhead(
        num_parameters: int,
        world_size: int,
        stage: int,
        bandwidth_gbps: float = 25.0
    ) -> Dict[str, float]:
        """
        Estimate communication overhead for ZeRO stages.
        
        Args:
            num_parameters: Number of model parameters
            world_size: Number of distributed processes
            stage: ZeRO stage
            bandwidth_gbps: Network bandwidth in GB/s
            
        Returns:
            Dictionary with communication estimates
        """
        
        param_bytes = num_parameters * 4  # FP32
        
        if stage == 1:
            # Only all-reduce for gradients
            comm_bytes_per_step = param_bytes  # gradient all-reduce
        elif stage == 2:
            # Gradient reduce-scatter + all-gather for optimizer states
            comm_bytes_per_step = param_bytes + (param_bytes * 2)  # grads + optimizer states
        elif stage == 3:
            # All-gather for params + reduce-scatter for grads + all-gather for optimizer
            comm_bytes_per_step = param_bytes * 3  # params + grads + optimizer
        else:
            comm_bytes_per_step = param_bytes  # baseline all-reduce
        
        comm_time_seconds = (comm_bytes_per_step / 1e9) / bandwidth_gbps
        
        return {
            'comm_bytes_per_step': comm_bytes_per_step,
            'comm_time_seconds': comm_time_seconds,
            'comm_overhead_ratio': comm_time_seconds / 0.1  # assume 0.1s compute per step
        }
    
    @staticmethod
    def optimal_bucket_size(
        world_size: int,
        bandwidth_gbps: float,
        latency_ms: float = 0.05
    ) -> int:
        """
        Calculate optimal bucket size for communication.
        
        Args:
            world_size: Number of processes
            bandwidth_gbps: Network bandwidth
            latency_ms: Network latency in milliseconds
            
        Returns:
            Optimal bucket size in bytes
        """
        
        # Balance between latency and bandwidth utilization
        # Larger buckets amortize latency but use more memory
        
        latency_penalty_bytes = latency_ms * 1e-3 * bandwidth_gbps * 1e9
        
        # Scale with world size to account for tree reduction
        scaling_factor = math.log2(max(2, world_size))
        
        optimal_size = int(latency_penalty_bytes * scaling_factor)
        
        # Clamp to reasonable range (1MB to 500MB)
        optimal_size = max(1024 * 1024, min(optimal_size, 500 * 1024 * 1024))
        
        return optimal_size
    
    @staticmethod
    def validate_partitioning_config(
        config,
        num_parameters: int,
        world_size: int,
        available_memory_gb: float
    ) -> List[str]:
        """
        Validate partitioning configuration and return warnings.
        
        Returns:
            List of warning messages
        """
        
        warnings = []
        
        # Check memory requirements
        memory_info = PartitionUtils.calculate_memory_savings(
            num_parameters, world_size, config.stage
        )
        
        if memory_info['memory_per_gpu_gb'] > available_memory_gb:
            warnings.append(
                f"Estimated memory usage ({memory_info['memory_per_gpu_gb']:.1f}GB) "
                f"exceeds available memory ({available_memory_gb:.1f}GB). "
                f"Consider enabling CPU offloading or using higher ZeRO stage."
            )
        
        # Check stage compatibility
        if config.stage > 3:
            warnings.append(f"Invalid ZeRO stage: {config.stage}. Must be 0, 1, 2, or 3.")
        
        if config.stage == 0 and world_size > 1:
            warnings.append("ZeRO stage 0 provides no memory savings in distributed training.")
        
        # Check offloading settings
        if hasattr(config, 'cpu_offload') and config.cpu_offload and config.stage == 0:
            warnings.append("CPU offloading requires ZeRO stage >= 1.")
        
        # Check bucket sizes
        if hasattr(config, 'reduce_bucket_size'):
            optimal_bucket = PartitionUtils.optimal_bucket_size(world_size, 25.0)
            actual_bucket = int(config.reduce_bucket_size)
            
            if actual_bucket < optimal_bucket * 0.1 or actual_bucket > optimal_bucket * 10:
                warnings.append(
                    f"Bucket size ({actual_bucket / 1e6:.1f}MB) may not be optimal. "
                    f"Consider using ~{optimal_bucket / 1e6:.1f}MB for better performance."
                )
        
        return warnings