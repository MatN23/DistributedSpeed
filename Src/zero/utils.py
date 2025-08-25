"""
DistributedSpeed ZeRO Utilities.

This module provides utility functions for ZeRO optimization including tensor operations,
gradient manipulation, memory management, and distributed communication helpers.

Key Functions:
- Tensor flattening and unflattening for efficient communication
- Gradient norm computation and clipping
- Memory alignment and padding utilities
- Distributed communication helpers
- Parameter management utilities

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import math
import logging
from typing import List, Optional, Union, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of current process in distributed training."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get the local rank within the node."""
    return int(os.environ.get('LOCAL_RANK', 0))


def flatten_dense_tensors_aligned(tensors: List[torch.Tensor], alignment: int = 128) -> torch.Tensor:
    """
    Flatten a list of dense tensors into a single tensor with memory alignment.
    
    Args:
        tensors: List of tensors to flatten
        alignment: Memory alignment in bytes (default: 128 for GPU efficiency)
        
    Returns:
        Flattened tensor with proper alignment
    """
    
    if not tensors:
        return torch.empty(0)
    
    # Calculate total size with alignment padding
    total_size = 0
    sizes = []
    
    for tensor in tensors:
        size = tensor.numel()
        sizes.append(size)
        
        # Add padding for alignment
        element_size = tensor.element_size()
        aligned_size = math.ceil(size * element_size / alignment) * alignment // element_size
        total_size += aligned_size
    
    # Create output tensor
    device = tensors[0].device
    dtype = tensors[0].dtype
    output = torch.empty(total_size, dtype=dtype, device=device)
    
    # Copy tensors with alignment
    offset = 0
    for i, tensor in enumerate(tensors):
        size = sizes[i]
        output[offset:offset + size].copy_(tensor.view(-1))
        
        # Update offset with alignment
        element_size = tensor.element_size()
        aligned_size = math.ceil(size * element_size / alignment) * alignment // element_size
        offset += aligned_size
    
    # Return only the used portion
    return output[:sum(sizes)]


def unflatten_dense_tensors(flat_tensor: torch.Tensor, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Unflatten a tensor back into a list of tensors with original shapes.
    
    Args:
        flat_tensor: Flattened tensor
        tensors: List of original tensors (for shape reference)
        
    Returns:
        List of tensors with original shapes
    """
    
    if not tensors:
        return []
    
    outputs = []
    offset = 0
    
    for tensor in tensors:
        numel = tensor.numel()
        output = flat_tensor[offset:offset + numel].view(tensor.shape)
        outputs.append(output)
        offset += numel
    
    return outputs


def compute_norm(tensors: List[torch.Tensor], norm_type: float = 2.0) -> float:
    """
    Compute the norm of a list of tensors.
    
    Args:
        tensors: List of tensors
        norm_type: Type of norm to compute (default: 2.0)
        
    Returns:
        Computed norm value
    """
    
    if not tensors:
        return 0.0
    
    if norm_type == float('inf'):
        # Infinity norm
        return max(tensor.abs().max().item() for tensor in tensors)
    else:
        # p-norm
        total_norm = 0.0
        for tensor in tensors:
            if norm_type == 2.0:
                # Optimized for L2 norm
                total_norm += tensor.norm(dtype=torch.float32).item() ** 2
            else:
                total_norm += tensor.norm(p=norm_type, dtype=torch.float32).item() ** norm_type
        
        return total_norm ** (1.0 / norm_type)


def get_global_norm(tensors: List[torch.Tensor], norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute global norm across all processes in distributed training.
    
    Args:
        tensors: List of tensors on current process
        norm_type: Type of norm to compute
        
    Returns:
        Global norm as tensor
    """
    
    local_norm = compute_norm(tensors, norm_type)
    
    if get_world_size() == 1:
        return torch.tensor(local_norm)
    
    # Create tensor for distributed reduction
    device = 'cuda' if torch.cuda.is_available() and tensors and tensors[0].is_cuda else 'cpu'
    
    if norm_type == 2.0:
        norm_tensor = torch.tensor(local_norm ** 2, device=device)
        dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
        return norm_tensor.sqrt()
    else:
        norm_tensor = torch.tensor(local_norm ** norm_type, device=device)
        dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
        return norm_tensor ** (1.0 / norm_type)


def clip_grad_norm_(parameters: List[nn.Parameter], max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """
    Clip gradient norm of parameters.
    
    Args:
        parameters: List of parameters
        max_norm: Maximum allowed gradient norm
        norm_type: Type of norm to compute
        
    Returns:
        Total gradient norm before clipping
    """
    
    # Filter parameters with gradients
    parameters = [p for p in parameters if p.grad is not None]
    
    if not parameters:
        return torch.tensor(0.0)
    
    # Compute gradient norms
    gradients = [p.grad for p in parameters]
    total_norm = compute_norm(gradients, norm_type)
    
    # Clip gradients if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for grad in gradients:
            grad.mul_(clip_coef)
    
    return torch.tensor(total_norm)


def pad_tensor(tensor: torch.Tensor, multiple_of: int) -> torch.Tensor:
    """
    Pad tensor to be a multiple of specified size.
    
    Args:
        tensor: Input tensor
        multiple_of: Size multiple requirement
        
    Returns:
        Padded tensor
    """
    
    current_size = tensor.numel()
    
    if current_size % multiple_of == 0:
        return tensor
    
    # Calculate padding needed
    target_size = math.ceil(current_size / multiple_of) * multiple_of
    padding_size = target_size - current_size
    
    # Create padding
    padding = torch.zeros(padding_size, dtype=tensor.dtype, device=tensor.device)
    
    # Concatenate
    return torch.cat([tensor.view(-1), padding])


def unpad_tensor(tensor: torch.Tensor, original_size: int) -> torch.Tensor:
    """
    Remove padding from tensor.
    
    Args:
        tensor: Padded tensor
        original_size: Original tensor size before padding
        
    Returns:
        Unpadded tensor
    """
    
    return tensor[:original_size]


def is_model_parallel_parameter(param: nn.Parameter) -> bool:
    """
    Check if parameter belongs to model parallel layers.
    
    Args:
        param: Parameter to check
        
    Returns:
        True if parameter is model parallel
    """
    
    # Check for common model parallel parameter attributes
    return hasattr(param, 'model_parallel') or hasattr(param, 'partition_dim')


def get_parameter_dtype(param: nn.Parameter) -> torch.dtype:
    """
    Get the dtype of a parameter, handling various wrapper scenarios.
    
    Args:
        param: Parameter to examine
        
    Returns:
        Parameter dtype
    """
    
    if hasattr(param, 'dtype'):
        return param.dtype
    elif hasattr(param, 'data') and hasattr(param.data, 'dtype'):
        return param.data.dtype
    else:
        return torch.float32  # Default fallback


def align_tensor_size(size: int, alignment: int = 128) -> int:
    """
    Align tensor size to specified boundary.
    
    Args:
        size: Original size
        alignment: Alignment boundary
        
    Returns:
        Aligned size
    """
    
    return math.ceil(size / alignment) * alignment


def create_gradient_bucket(parameters: List[nn.Parameter], bucket_size: int) -> List[List[nn.Parameter]]:
    """
    Create buckets of parameters for efficient gradient communication.
    
    Args:
        parameters: List of parameters to bucket
        bucket_size: Target bucket size in bytes
        
    Returns:
        List of parameter buckets
    """
    
    buckets = []
    current_bucket = []
    current_bucket_size = 0
    
    for param in parameters:
        if param.requires_grad:
            param_size = param.numel() * param.element_size()
            
            if current_bucket_size + param_size > bucket_size and current_bucket:
                # Start new bucket
                buckets.append(current_bucket)
                current_bucket = [param]
                current_bucket_size = param_size
            else:
                current_bucket.append(param)
                current_bucket_size += param_size
    
    # Add final bucket
    if current_bucket:
        buckets.append(current_bucket)
    
    return buckets


def partition_parameters_by_size(
    parameters: List[nn.Parameter],
    world_size: int,
    balance_method: str = 'size'
) -> List[List[nn.Parameter]]:
    """
    Partition parameters across processes with load balancing.
    
    Args:
        parameters: List of parameters to partition
        world_size: Number of processes
        balance_method: Method for balancing ('size' or 'count')
        
    Returns:
        List of parameter partitions for each process
    """
    
    if balance_method == 'count':
        # Simple round-robin partitioning by count
        partitions = [[] for _ in range(world_size)]
        for i, param in enumerate(parameters):
            partitions[i % world_size].append(param)
        return partitions
    
    elif balance_method == 'size':
        # Balance by parameter size
        param_sizes = [(param, param.numel()) for param in parameters]
        param_sizes.sort(key=lambda x: x[1], reverse=True)  # Sort by size, largest first
        
        # Initialize partitions
        partitions = [[] for _ in range(world_size)]
        partition_sizes = [0] * world_size
        
        # Greedy assignment to least loaded partition
        for param, size in param_sizes:
            min_partition = min(range(world_size), key=lambda i: partition_sizes[i])
            partitions[min_partition].append(param)
            partition_sizes[min_partition] += size
        
        return partitions
    
    else:
        raise ValueError(f"Unknown balance method: {balance_method}")


def get_memory_stats() -> dict:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    
    stats = {}
    
    if torch.cuda.is_available():
        stats.update({
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'cached_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
        })
    else:
        stats.update({
            'allocated_gb': 0.0,
            'cached_gb': 0.0,
            'max_allocated_gb': 0.0,
            'free_gb': 0.0
        })
    
    return stats


def estimate_tensor_memory(tensors: List[torch.Tensor]) -> int:
    """
    Estimate memory usage of a list of tensors.
    
    Args:
        tensors: List of tensors
        
    Returns:
        Total memory usage in bytes
    """
    
    total_memory = 0
    for tensor in tensors:
        total_memory += tensor.numel() * tensor.element_size()
    
    return total_memory


def create_contiguous_tensor_from_list(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Create a contiguous tensor from a list of tensors.
    
    Args:
        tensors: List of input tensors
        
    Returns:
        Single contiguous tensor
    """
    
    if not tensors:
        return torch.empty(0)
    
    # Determine output properties
    device = tensors[0].device
    dtype = tensors[0].dtype
    total_size = sum(t.numel() for t in tensors)
    
    # Create output tensor
    output = torch.empty(total_size, dtype=dtype, device=device)
    
    # Copy tensors
    offset = 0
    for tensor in tensors:
        size = tensor.numel()
        output[offset:offset + size].copy_(tensor.view(-1))
        offset += size
    
    return output


def split_contiguous_tensor(tensor: torch.Tensor, sizes: List[int], shapes: List[Tuple]) -> List[torch.Tensor]:
    """
    Split a contiguous tensor back into original tensors.
    
    Args:
        tensor: Contiguous input tensor
        sizes: List of original tensor sizes
        shapes: List of original tensor shapes
        
    Returns:
        List of split tensors
    """
    
    outputs = []
    offset = 0
    
    for size, shape in zip(sizes, shapes):
        output = tensor[offset:offset + size].view(shape)
        outputs.append(output)
        offset += size
    
    return outputs


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safely divide tensors with epsilon for numerical stability.
    
    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Small epsilon value for stability
        
    Returns:
        Result of safe division
    """
    
    return numerator / (denominator + eps)


def reduce_tensor_dict(tensor_dict: dict, op: str = 'mean') -> dict:
    """
    Reduce a dictionary of tensors across all processes.
    
    Args:
        tensor_dict: Dictionary mapping names to tensors
        op: Reduction operation ('mean', 'sum', 'max', 'min')
        
    Returns:
        Dictionary with reduced tensors
    """
    
    if get_world_size() == 1:
        return tensor_dict
    
    reduced_dict = {}
    
    for key, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            # Convert to cuda if available for communication
            if torch.cuda.is_available() and not tensor.is_cuda:
                comm_tensor = tensor.cuda()
            else:
                comm_tensor = tensor.clone()
            
            # Perform reduction
            if op == 'mean':
                dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)
                comm_tensor.div_(get_world_size())
            elif op == 'sum':
                dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)
            elif op == 'max':
                dist.all_reduce(comm_tensor, op=dist.ReduceOp.MAX)
            elif op == 'min':
                dist.all_reduce(comm_tensor, op=dist.ReduceOp.MIN)
            else:
                raise ValueError(f"Unsupported reduction operation: {op}")
            
            # Move back to original device
            if tensor.device != comm_tensor.device:
                comm_tensor = comm_tensor.to(tensor.device)
            
            reduced_dict[key] = comm_tensor
        else:
            # Non-tensor values pass through unchanged
            reduced_dict[key] = tensor
    
    return reduced_dict