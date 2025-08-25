"""
ZeRO (Zero Redundancy Optimizer) implementation for DistributedSpeed.

This module provides the core ZeRO optimizer functionality that eliminates
memory redundancies in distributed training by partitioning optimizer states,
gradients, and parameters across data parallel processes.

ZeRO Stages:
- Stage 1: Partition optimizer states
- Stage 2: Partition optimizer states + gradients  
- Stage 3: Partition optimizer states + gradients + parameters
"""

import math
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.nn import Parameter
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from ..config import ZeROConfig
from ..communication.backend import CommunicationBackend
from ..utils.logging import get_logger
from ..utils.tensor_utils import get_global_norm, clip_tensors_by_global_norm

logger = get_logger(__name__)


class ZeROOptimizer(ABC):
    """
    Abstract base class for ZeRO optimizers.
    
    Defines the common interface and functionality shared across all ZeRO stages.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: ZeROConfig,
        mpu: Optional[Any] = None
    ):
        """
        Initialize ZeRO optimizer base class.
        
        Args:
            optimizer: Base PyTorch optimizer to wrap
            config: ZeRO configuration
            mpu: Model parallel utilities (optional)
        """
        self.optimizer = optimizer
        self.config = config
        self.mpu = mpu
        
        # Distributed training setup
        if not dist.is_initialized():
            raise RuntimeError("Distributed training must be initialized before ZeRO")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        
        # Communication backend
        self.comm_backend = CommunicationBackend()
        
        # ZeRO state
        self.overflow = False
        self.grad_norm = 0.0
        
        # Performance tracking
        self.step_count = 0
        self.communication_time = 0.0
        self.computation_time = 0.0
        
        # Parameter management
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        
        # Initialize ZeRO-specific state
        self._initialize_zero_state()
        
        logger.info(f"ZeRO optimizer initialized on rank {self.rank}")
    
    @abstractmethod
    def _initialize_zero_state(self):
        """Initialize ZeRO-specific state. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def step(self, closure: Optional[callable] = None):
        """Perform optimizer step. Must be implemented by subclasses.""" 
        pass
    
    @abstractmethod
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def check_overflow(self) -> bool:
        """Check for gradient overflow. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_global_norm(self) -> float:
        """Get global gradient norm. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def clip_gradients(self, max_norm: float, norm_type: float = 2.0) -> float:
        """Clip gradients by global norm. Must be implemented by subclasses."""
        pass
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary."""
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'overflow': self.overflow,
            'grad_norm': self.grad_norm,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dictionary."""
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.step_count = state_dict.get('step_count', 0)
        self.overflow = state_dict.get('overflow', False)
        self.grad_norm = state_dict.get('grad_norm', 0.0)
    
    def add_param_group(self, param_group: Dict[str, Any]):
        """Add parameter group to optimizer."""
        self.optimizer.add_param_group(param_group)
        self.param_groups = self.optimizer.param_groups
    
    @property
    def defaults(self):
        """Get optimizer defaults."""
        return self.optimizer.defaults
    
    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        return [group['lr'] for group in self.param_groups]
    
    def set_lr(self, lr: Union[float, List[float]]):
        """Set learning rate for all parameter groups."""
        if isinstance(lr, float):
            lr = [lr] * len(self.param_groups)
        
        for group, learning_rate in zip(self.param_groups, lr):
            group['lr'] = learning_rate
    
    def _partition_parameters(self, parameters: List[Parameter]) -> Dict[int, List[Parameter]]:
        """
        Partition parameters across processes.
        
        Args:
            parameters: List of parameters to partition
            
        Returns:
            Dictionary mapping rank to parameter list
        """
        partition_size = math.ceil(len(parameters) / self.world_size)
        partitions = {}
        
        for rank in range(self.world_size):
            start_idx = rank * partition_size
            end_idx = min(start_idx + partition_size, len(parameters))
            partitions[rank] = parameters[start_idx:end_idx]
        
        return partitions
    
    def _get_fp32_parameters(self, param_group: Dict[str, Any]) -> List[Parameter]:
        """Get FP32 parameters from parameter group."""
        fp32_params = []
        for param in param_group['params']:
            if param.requires_grad:
                if param.dtype != torch.float32:
                    # Create FP32 copy if needed
                    fp32_param = param.detach().clone().float()
                    fp32_params.append(fp32_param)
                else:
                    fp32_params.append(param)
        return fp32_params
    
    def _update_fp16_params(self, fp32_params: List[Parameter], fp16_params: List[Parameter]):
        """Update FP16 parameters from FP32 parameters."""
        for fp32_param, fp16_param in zip(fp32_params, fp16_params):
            fp16_param.data.copy_(fp32_param.data.half())
    
    def _reduce_scatter_tensor(self, tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """
        Reduce-scatter operation for tensor.
        
        Args:
            tensor: Input tensor to reduce-scatter
            group: Process group for communication
            
        Returns:
            Scattered portion of reduced tensor
        """
        # Flatten tensor for reduce-scatter
        flat_tensor = tensor.flatten()
        
        # Ensure tensor size is divisible by world size
        padded_size = math.ceil(flat_tensor.numel() / self.world_size) * self.world_size
        if flat_tensor.numel() < padded_size:
            padded_tensor = torch.zeros(
                padded_size, dtype=flat_tensor.dtype, device=flat_tensor.device
            )
            padded_tensor[:flat_tensor.numel()] = flat_tensor
            flat_tensor = padded_tensor
        
        # Create output tensor
        chunk_size = flat_tensor.numel() // self.world_size
        output_tensor = torch.zeros(
            chunk_size, dtype=flat_tensor.dtype, device=flat_tensor.device
        )
        
        # Perform reduce-scatter
        input_list = list(flat_tensor.chunk(self.world_size))
        dist.reduce_scatter(output_tensor, input_list, group=group)
        
        return output_tensor
    
    def _all_gather_tensor(self, tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        """
        All-gather operation for tensor.
        
        Args:
            tensor: Input tensor to all-gather
            group: Process group for communication
            
        Returns:
            Gathered tensor from all processes
        """
        output_tensors = [
            torch.zeros_like(tensor) for _ in range(self.world_size)
        ]
        dist.all_gather(output_tensors, tensor, group=group)
        return torch.cat(output_tensors, dim=0)
    
    def _communicate_gradients(self, params: List[Parameter]):
        """Communicate gradients across processes."""
        if not self.config.overlap_comm:
            # Synchronous communication
            self._sync_gradients(params)
        else:
            # Asynchronous communication with overlap
            self._async_gradients(params)
    
    def _sync_gradients(self, params: List[Parameter]):
        """Synchronously communicate gradients."""
        start_time = time.time()
        
        # Group gradients by type for efficient communication
        grad_groups = defaultdict(list)
        for param in params:
            if param.grad is not None:
                grad_groups[param.grad.dtype].append(param.grad)
        
        # Communicate each group
        for dtype, grads in grad_groups.items():
            if grads:
                flat_grads = _flatten_dense_tensors(grads)
                dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                flat_grads.div_(self.world_size)
                
                # Unflatten and copy back
                unflat_grads = _unflatten_dense_tensors(flat_grads, grads)
                for grad, unflat_grad in zip(grads, unflat_grads):
                    grad.copy_(unflat_grad)
        
        self.communication_time += time.time() - start_time
    
    def _async_gradients(self, params: List[Parameter]):
        """Asynchronously communicate gradients with computation overlap."""
        # Implementation would use async operations
        # For now, fall back to synchronous
        self._sync_gradients(params)
    
    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient synchronization."""
        # Store original overlap setting
        original_overlap = self.config.overlap_comm
        self.config.overlap_comm = False
        
        try:
            yield
        finally:
            # Restore original setting
            self.config.overlap_comm = original_overlap
    
    def _bucket_tensors(self, tensors: List[torch.Tensor], bucket_size: int) -> List[List[torch.Tensor]]:
        """
        Group tensors into buckets for efficient communication.
        
        Args:
            tensors: List of tensors to bucket
            bucket_size: Maximum bucket size in bytes
            
        Returns:
            List of tensor buckets
        """
        buckets = []
        current_bucket = []
        current_size = 0
        
        for tensor in tensors:
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > bucket_size and current_bucket:
                buckets.append(current_bucket)
                current_bucket = [tensor]
                current_size = tensor_size
            else:
                current_bucket.append(tensor)
                current_size += tensor_size
        
        if current_bucket:
            buckets.append(current_bucket)
        
        return buckets
    
    def _log_memory_usage(self, stage: str = ""):
        """Log current memory usage for debugging."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            
            logger.debug(
                f"[Rank {self.rank}] {stage} Memory - "
                f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
            )
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        stats = {
            'optimizer_memory_gb': 0.0,
            'gradient_memory_gb': 0.0,
            'parameter_memory_gb': 0.0
        }
        
        if not torch.cuda.is_available():
            return stats
        
        # Calculate parameter memory
        param_memory = 0
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    param_memory += param.numel() * param.element_size()
        
        stats['parameter_memory_gb'] = param_memory / (1024**3)
        
        # Calculate gradient memory
        grad_memory = 0
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    grad_memory += param.grad.numel() * param.grad.element_size()
        
        stats['gradient_memory_gb'] = grad_memory / (1024**3)
        
        # Optimizer state memory (approximation)
        optimizer_memory = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    optimizer_memory += value.numel() * value.element_size()
        
        stats['optimizer_memory_gb'] = optimizer_memory / (1024**3)
        
        return stats
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return {
            'step_count': self.step_count,
            'communication_time_s': self.communication_time,
            'computation_time_s': self.computation_time,
            'avg_step_time_s': (self.communication_time + self.computation_time) / max(1, self.step_count),
            'communication_ratio': self.communication_time / max(1, self.communication_time + self.computation_time)
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking statistics."""
        self.step_count = 0
        self.communication_time = 0.0
        self.computation_time = 0.0
    
    def validate_configuration(self):
        """Validate ZeRO configuration for current setup."""
        # Check world size compatibility
        if self.world_size == 1 and self.config.enabled:
            logger.warning(
                "ZeRO optimization enabled with world_size=1. "
                "Consider disabling ZeRO for single-GPU training."
            )
        
        # Check bucket size
        if self.config.allgather_bucket_size <= 0:
            raise ValueError("allgather_bucket_size must be positive")
        
        # Check CPU offload compatibility
        if self.config.cpu_offload and not torch.cuda.is_available():
            logger.warning("CPU offload enabled but CUDA not available")
    
    def __repr__(self) -> str:
        """String representation of ZeRO optimizer."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  stage={self.config.stage},\n"
            f"  world_size={self.world_size},\n"
            f"  rank={self.rank},\n"
            f"  step_count={self.step_count}\n"
            f")"
        )


class ZeROOptimizerUtils:
    """Utility functions for ZeRO optimizers."""
    
    @staticmethod
    def get_partition_info(total_size: int, world_size: int, rank: int) -> Tuple[int, int]:
        """
        Get partition information for current rank.
        
        Args:
            total_size: Total number of elements
            world_size: Number of processes
            rank: Current process rank
            
        Returns:
            Tuple of (start_index, partition_size)
        """
        partition_size = math.ceil(total_size / world_size)
        start_index = rank * partition_size
        actual_size = min(partition_size, total_size - start_index)
        return start_index, actual_size
    
    @staticmethod
    def create_fp32_copy(param: Parameter) -> Parameter:
        """Create FP32 copy of parameter for optimizer."""
        fp32_param = Parameter(param.detach().clone().float())
        fp32_param.requires_grad = param.requires_grad
        return fp32_param
    
    @staticmethod
    def sync_fp32_to_fp16(fp32_param: Parameter, fp16_param: Parameter):
        """Synchronize FP32 parameter updates to FP16 parameter.""" 
        fp16_param.data.copy_(fp32_param.data.half())
    
    @staticmethod
    def partition_tensor(tensor: torch.Tensor, world_size: int) -> List[torch.Tensor]:
        """Partition tensor across world_size processes."""
        if tensor.numel() == 0:
            return [torch.empty(0, dtype=tensor.dtype, device=tensor.device) for _ in range(world_size)]
        
        partition_size = math.ceil(tensor.numel() / world_size)
        partitions = []
        
        flat_tensor = tensor.flatten()
        for rank in range(world_size):
            start_idx = rank * partition_size
            end_idx = min(start_idx + partition_size, flat_tensor.numel())
            
            if start_idx < flat_tensor.numel():
                partition = flat_tensor[start_idx:end_idx].contiguous()
            else:
                partition = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
            
            partitions.append(partition)
        
        return partitions
    
    @staticmethod
    def gather_partitions(partitions: List[torch.Tensor], original_shape: torch.Size) -> torch.Tensor:
        """Gather partitions back into original tensor shape."""
        if not partitions:
            return torch.empty(original_shape)
        
        # Filter out empty partitions
        non_empty_partitions = [p for p in partitions if p.numel() > 0]
        
        if not non_empty_partitions:
            return torch.empty(original_shape, dtype=partitions[0].dtype, device=partitions[0].device)
        
        flat_tensor = torch.cat(non_empty_partitions, dim=0)
        
        # Trim to original size if needed (due to padding)
        total_elements = 1
        for dim in original_shape:
            total_elements *= dim
        
        if flat_tensor.numel() > total_elements:
            flat_tensor = flat_tensor[:total_elements]
        
        return flat_tensor.reshape(original_shape)


# Factory function for creating ZeRO optimizers
def create_zero_optimizer(
    optimizer: Optimizer,
    config: ZeROConfig,
    mpu: Optional[Any] = None
) -> ZeROOptimizer:
    """
    Factory function to create appropriate ZeRO optimizer.
    
    Args:
        optimizer: Base PyTorch optimizer
        config: ZeRO configuration
        mpu: Model parallel utilities
        
    Returns:
        ZeRO optimizer instance
        
    Raises:
        ValueError: If unsupported ZeRO stage is specified
    """
    if not config.enabled:
        return optimizer
    
    if config.stage == 1:
        from .stage1 import ZeROStage1
        return ZeROStage1(optimizer, config, mpu)
    elif config.stage == 2:
        from .stage2 import ZeROStage2
        return ZeROStage2(optimizer, config, mpu)
    elif config.stage == 3:
        from .stage3 import ZeROStage3
        return ZeROStage3(optimizer, config, mpu)
    else:
        raise ValueError(f"Unsupported ZeRO stage: {config.stage}")


class ParameterPartitionManager:
    """
    Manages parameter partitioning for ZeRO optimizers.
    
    This class handles the complex logic of partitioning model parameters
    across processes while maintaining efficient access patterns.
    """
    
    def __init__(self, world_size: int, rank: int):
        """
        Initialize parameter partition manager.
        
        Args:
            world_size: Total number of processes
            rank: Current process rank
        """
        self.world_size = world_size
        self.rank = rank
        self.partitions = {}
        self.partition_info = {}
    
    def create_partitions(
        self,
        parameters: List[Parameter],
        partition_id: str = "default"
    ) -> Dict[int, List[Parameter]]:
        """
        Create parameter partitions across all processes.
        
        Args:
            parameters: List of parameters to partition
            partition_id: Identifier for this partition set
            
        Returns:
            Dictionary mapping rank to parameter list
        """
        total_params = len(parameters)
        params_per_rank = math.ceil(total_params / self.world_size)
        
        partitions = {}
        for rank in range(self.world_size):
            start_idx = rank * params_per_rank
            end_idx = min(start_idx + params_per_rank, total_params)
            partitions[rank] = parameters[start_idx:end_idx]
        
        # Store partition info
        self.partitions[partition_id] = partitions
        self.partition_info[partition_id] = {
            'total_params': total_params,
            'params_per_rank': params_per_rank
        }
        
        return partitions
    
    def get_local_partition(self, partition_id: str = "default") -> List[Parameter]:
        """Get parameters assigned to current rank."""
        return self.partitions.get(partition_id, {}).get(self.rank, [])
    
    def get_partition_for_rank(self, rank: int, partition_id: str = "default") -> List[Parameter]:
        """Get parameters assigned to specific rank."""
        return self.partitions.get(partition_id, {}).get(rank, [])
    
    def is_parameter_local(self, param: Parameter, partition_id: str = "default") -> bool:
        """Check if parameter is assigned to current rank."""
        local_params = self.get_local_partition(partition_id)
        return param in local_params
    
    def get_parameter_owner(self, param: Parameter, partition_id: str = "default") -> Optional[int]:
        """Get the rank that owns a specific parameter."""
        for rank, params in self.partitions.get(partition_id, {}).items():
            if param in params:
                return rank
        return None


class GradientBuffer:
    """
    Efficient gradient buffer for ZeRO communication.
    
    This class manages gradient accumulation and communication patterns
    to minimize memory usage and communication overhead.
    """
    
    def __init__(
        self,
        parameters: List[Parameter],
        bucket_size: int = int(25e6),  # 25MB default
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize gradient buffer.
        
        Args:
            parameters: Parameters to manage gradients for
            bucket_size: Maximum bucket size in bytes
            dtype: Data type for gradient buffer
        """
        self.parameters = parameters
        self.bucket_size = bucket_size
        self.dtype = dtype
        self.device = parameters[0].device if parameters else torch.device('cpu')
        
        # Create buckets
        self.buckets = self._create_buckets()
        self.bucket_buffers = {}
        self.bucket_gradients = {}
        
        # Initialize buffers
        self._initialize_buffers()
    
    def _create_buckets(self) -> List[List[Parameter]]:
        """Create parameter buckets for efficient communication."""
        buckets = []
        current_bucket = []
        current_size = 0
        
        for param in self.parameters:
            if param.requires_grad:
                param_size = param.numel() * torch.finfo(self.dtype).bits // 8
                
                if current_size + param_size > self.bucket_size and current_bucket:
                    buckets.append(current_bucket)
                    current_bucket = [param]
                    current_size = param_size
                else:
                    current_bucket.append(param)
                    current_size += param_size
        
        if current_bucket:
            buckets.append(current_bucket)
        
        return buckets
    
    def _initialize_buffers(self):
        """Initialize gradient buffers for each bucket."""
        for bucket_idx, bucket_params in enumerate(self.buckets):
            total_numel = sum(p.numel() for p in bucket_params)
            
            # Create buffer
            buffer = torch.zeros(
                total_numel, dtype=self.dtype, device=self.device
            )
            self.bucket_buffers[bucket_idx] = buffer
            
            # Map parameters to buffer positions
            param_map = {}
            offset = 0
            for param in bucket_params:
                param_map[param] = (offset, offset + param.numel())
                offset += param.numel()
            
            self.bucket_gradients[bucket_idx] = param_map
    
    def copy_gradients_to_buffer(self, bucket_idx: int):
        """Copy parameter gradients to bucket buffer."""
        buffer = self.bucket_buffers[bucket_idx]
        param_map = self.bucket_gradients[bucket_idx]
        
        for param, (start, end) in param_map.items():
            if param.grad is not None:
                buffer[start:end] = param.grad.flatten().to(self.dtype)
    
    def copy_gradients_from_buffer(self, bucket_idx: int):
        """Copy gradients from bucket buffer back to parameters."""
        buffer = self.bucket_buffers[bucket_idx]
        param_map = self.bucket_gradients[bucket_idx]
        
        for param, (start, end) in param_map.items():
            if param.grad is not None:
                grad_data = buffer[start:end].view(param.grad.shape).to(param.grad.dtype)
                param.grad.copy_(grad_data)
    
    def get_bucket_buffer(self, bucket_idx: int) -> torch.Tensor:
        """Get gradient buffer for specific bucket."""
        return self.bucket_buffers[bucket_idx]
    
    def all_reduce_bucket(self, bucket_idx: int, op: dist.ReduceOp = dist.ReduceOp.SUM):
        """Perform all-reduce on bucket buffer."""
        buffer = self.bucket_buffers[bucket_idx]
        dist.all_reduce(buffer, op=op)
    
    def reduce_scatter_bucket(self, bucket_idx: int, output_tensor: torch.Tensor):
        """Perform reduce-scatter on bucket buffer."""
        buffer = self.bucket_buffers[bucket_idx]
        
        # Create input list for reduce-scatter
        world_size = dist.get_world_size()
        chunk_size = buffer.numel() // world_size
        input_list = [buffer[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]
        
        dist.reduce_scatter(output_tensor, input_list)
    
    def zero_buffers(self):
        """Zero all gradient buffers."""
        for buffer in self.bucket_buffers.values():
            buffer.zero_()


class OptimizerStateManager:
    """
    Manages optimizer state for ZeRO implementations.
    
    Handles partitioning, serialization, and synchronization of optimizer
    state across distributed processes.
    """
    
    def __init__(self, world_size: int, rank: int):
        """
        Initialize optimizer state manager.
        
        Args:
            world_size: Total number of processes
            rank: Current process rank
        """
        self.world_size = world_size
        self.rank = rank
        self.local_state = {}
        self.global_state_map = {}
        self.state_partitions = {}
    
    def partition_optimizer_state(self, optimizer_state: Dict[Parameter, Any]):
        """
        Partition optimizer state across processes.
        
        Args:
            optimizer_state: Full optimizer state dictionary
        """
        parameters = list(optimizer_state.keys())
        partition_size = math.ceil(len(parameters) / self.world_size)
        
        # Determine local parameters
        start_idx = self.rank * partition_size
        end_idx = min(start_idx + partition_size, len(parameters))
        local_params = parameters[start_idx:end_idx]
        
        # Extract local state
        self.local_state = {
            param: optimizer_state[param] for param in local_params
        }
        
        # Create global mapping
        for rank in range(self.world_size):
            rank_start = rank * partition_size
            rank_end = min(rank_start + partition_size, len(parameters))
            rank_params = parameters[rank_start:rank_end]
            self.state_partitions[rank] = rank_params
            
            for param in rank_params:
                self.global_state_map[param] = rank
    
    def get_local_state(self) -> Dict[Parameter, Any]:
        """Get optimizer state for local parameters."""
        return self.local_state
    
    def get_parameter_owner_rank(self, param: Parameter) -> Optional[int]:
        """Get the rank that owns optimizer state for parameter."""
        return self.global_state_map.get(param)
    
    def is_local_parameter(self, param: Parameter) -> bool:
        """Check if parameter state is stored locally."""
        return param in self.local_state
    
    def gather_state_from_rank(self, param: Parameter, source_rank: int) -> Any:
        """
        Gather parameter state from specific rank.
        
        Args:
            param: Parameter to get state for
            source_rank: Rank that owns the state
            
        Returns:
            Parameter optimizer state
        """
        # This would implement actual communication to gather state
        # For now, return None as placeholder
        return None
    
    def broadcast_state_to_all(self, param: Parameter, state: Any):
        """Broadcast parameter state to all ranks.""" 
        # This would implement actual broadcast communication
        # For now, just store locally
        if self.is_local_parameter(param):
            self.local_state[param] = state
    
    def synchronize_state(self):
        """Synchronize optimizer state across all processes."""
        # Implementation would handle state synchronization
        # This is a complex operation that varies by ZeRO stage
        pass
    
    def checkpoint_local_state(self) -> Dict[str, Any]:
        """Create checkpoint of local optimizer state."""
        # Convert parameter keys to indices for serialization
        checkpoint = {}
        for i, (param, state) in enumerate(self.local_state.items()):
            checkpoint[f"param_{i}"] = {
                'param_shape': param.shape,
                'param_dtype': param.dtype,
                'state': state
            }
        return checkpoint
    
    def load_checkpoint_state(self, checkpoint: Dict[str, Any], parameters: List[Parameter]):
        """Load optimizer state from checkpoint."""
        # Restore state from checkpoint
        self.local_state = {}
        param_idx = 0
        
        for param in parameters:
            if param_idx < len(checkpoint):
                checkpoint_key = f"param_{param_idx}"
                if checkpoint_key in checkpoint:
                    state_info = checkpoint[checkpoint_key]
                    if (param.shape == state_info['param_shape'] and
                        param.dtype == state_info['param_dtype']):
                        self.local_state[param] = state_info['state']
                param_idx += 1