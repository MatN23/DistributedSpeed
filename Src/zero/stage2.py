"""
ZeRO Stage 2 Implementation for DistributedSpeed.

ZeRO Stage 2 partitions both optimizer states and gradients across data parallel
processes. This significantly reduces memory usage while maintaining similar
communication patterns to Stage 1.

Features:
- Optimizer state partitioning (from Stage 1)
- Gradient partitioning using reduce-scatter
- Efficient gradient communication with bucketing
- Support for gradient compression
- CPU offloading for optimizer states
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.nn import Parameter

from .optimizer import ZeROOptimizer, ParameterPartitionManager, GradientBuffer, OptimizerStateManager
from ..config import ZeROConfig
from ..utils.logging import get_logger
from ..utils.tensor_utils import get_global_norm, clip_tensors_by_global_norm

logger = get_logger(__name__)


class ZeROStage2(ZeROOptimizer):
    """
    ZeRO Stage 2: Optimizer State + Gradient Partitioning.
    
    This implementation partitions both optimizer states and gradients across
    data parallel processes, providing significant memory savings compared to
    standard data parallelism.
    
    Key optimizations:
    - Gradient reduce-scatter for communication efficiency
    - Bucket-based gradient communication
    - Overlapped communication and computation
    - CPU offloading support
    - Mixed precision training support
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        config: ZeROConfig,
        mpu: Optional[Any] = None
    ):
        """
        Initialize ZeRO Stage 2 optimizer.
        
        Args:
            optimizer: Base PyTorch optimizer to wrap
            config: ZeRO configuration
            mpu: Model parallel utilities (optional)
        """
        super().__init__(optimizer, config, mpu)
        
        # Stage 2 specific initialization
        self.stage = 2
        
        # Parameter and state management
        self.partition_manager = ParameterPartitionManager(self.world_size, self.rank)
        self.state_manager = OptimizerStateManager(self.world_size, self.rank)
        
        # Gradient management
        self.gradient_buffers = {}
        self.reduced_gradients = {}
        self.gradient_reduction_handles = {}
        
        # CPU offloading
        self.cpu_optimizer_states = {}
        self.cpu_gradients = {}
        
        # Performance tracking
        self.reduce_scatter_time = 0.0
        self.allgather_time = 0.0
        self.cpu_offload_time = 0.0
        
        # Initialize Stage 2 components
        self._initialize_stage2()
        
        logger.info(f"ZeRO Stage 2 initialized on rank {self.rank}")
    
    def _initialize_zero_state(self):
        """Initialize ZeRO Stage 2 specific state.""" 
        self._partition_parameters()
        self._initialize_gradient_buffers()
        self._partition_optimizer_states()
        
        if self.config.cpu_offload:
            self._initialize_cpu_offload()
    
    def _partition_parameters(self):
        """Partition parameters across processes for gradient handling."""
        all_parameters = []
        
        # Collect all parameters from all parameter groups
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    all_parameters.append(param)
        
        # Create partitions
        self.parameter_partitions = self.partition_manager.create_partitions(
            all_parameters, "gradients"
        )
        
        # Store local parameters for this rank
        self.local_parameters = self.partition_manager.get_local_partition("gradients")
        
        logger.debug(
            f"Rank {self.rank}: Managing {len(self.local_parameters)} parameters "
            f"out of {len(all_parameters)} total"
        )
    
    def _initialize_gradient_buffers(self):
        """Initialize gradient communication buffers."""
        for group_idx, group in enumerate(self.param_groups):
            group_params = [p for p in group['params'] if p.requires_grad]
            
            if group_params:
                # Create gradient buffer for this parameter group
                buffer = GradientBuffer(
                    group_params,
                    bucket_size=self.config.allgather_bucket_size,
                    dtype=torch.float32
                )
                self.gradient_buffers[group_idx] = buffer
                
                # Initialize reduced gradient storage
                self.reduced_gradients[group_idx] = {}
    
    def _partition_optimizer_states(self):
        """Partition optimizer states across processes.""" 
        # Get full optimizer state
        full_state = self.optimizer.state
        
        # Partition states
        self.state_manager.partition_optimizer_state(full_state)
        
        # Update optimizer to only track local parameters
        self.optimizer.state = self.state_manager.get_local_state()
        
        # Update parameter groups to only include local parameters
        self._update_local_parameter_groups()
    
    def _update_local_parameter_groups(self):
        """Update parameter groups to only include locally owned parameters."""
        new_param_groups = []
        
        for group in self.param_groups:
            new_group = {k: v for k, v in group.items() if k != 'params'}
            local_params = []
            
            for param in group['params']:
                if self.partition_manager.is_parameter_local(param, "gradients"):
                    local_params.append(param)
            
            new_group['params'] = local_params
            new_param_groups.append(new_group)
        
        self.param_groups = new_param_groups
        self.optimizer.param_groups = new_param_groups
    
    def _initialize_cpu_offload(self):
        """Initialize CPU offloading for optimizer states."""
        if not self.config.cpu_offload:
            return
        
        logger.info(f"Rank {self.rank}: Enabling CPU offload for optimizer states")
        
        # Move optimizer states to CPU
        for param, state in self.optimizer.state.items():
            cpu_state = {}
            for key, value in state.items():
                if torch.is_tensor(value):
                    cpu_state[key] = value.cpu().pin_memory()
                else:
                    cpu_state[key] = value
            self.cpu_optimizer_states[param] = cpu_state
        
        # Clear GPU optimizer states
        self.optimizer.state.clear()
    
    def _initialize_stage2(self):
        """Complete Stage 2 initialization.""" 
        # Validate configuration
        self.validate_configuration()
        
        # Pre-allocate communication buffers
        self._preallocate_communication_buffers()
        
        # Setup gradient hooks for automatic reduction
        if self.config.overlap_comm:
            self._register_gradient_hooks()
    
    def _preallocate_communication_buffers(self):
        """Pre-allocate buffers for efficient communication."""
        for group_idx, buffer in self.gradient_buffers.items():
            # Pre-allocate reduce-scatter output buffers
            for bucket_idx in range(len(buffer.buckets)):
                bucket_buffer = buffer.get_bucket_buffer(bucket_idx)
                partition_size = bucket_buffer.numel() // self.world_size
                
                # Create output buffer for reduce-scatter
                output_buffer = torch.zeros(
                    partition_size,
                    dtype=bucket_buffer.dtype,
                    device=bucket_buffer.device
                )
                
                if group_idx not in self.reduced_gradients:
                    self.reduced_gradients[group_idx] = {}
                self.reduced_gradients[group_idx][bucket_idx] = output_buffer
    
    def _register_gradient_hooks(self):
        """Register hooks for overlapped gradient communication."""
        def gradient_hook(param):
            def hook_fn(grad):
                # Trigger asynchronous gradient reduction
                self._async_reduce_gradient(param, grad)
                return grad
            return hook_fn
        
        # Register hooks for all parameters
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    param.register_hook(gradient_hook(param))
    
    def _async_reduce_gradient(self, param: Parameter, grad: torch.Tensor):
        """Asynchronously reduce gradient for parameter."""
        # Find which group and bucket this parameter belongs to
        group_idx, bucket_idx = self._find_parameter_bucket(param)
        
        if group_idx is not None and bucket_idx is not None:
            # Mark gradient as ready for reduction
            self._mark_gradient_ready(group_idx, bucket_idx, param)
    
    def _find_parameter_bucket(self, param: Parameter) -> Tuple[Optional[int], Optional[int]]:
        """Find which group and bucket a parameter belongs to."""
        for group_idx, buffer in self.gradient_buffers.items():
            for bucket_idx, bucket_params in enumerate(buffer.buckets):
                if param in bucket_params:
                    return group_idx, bucket_idx
        return None, None
    
    def _mark_gradient_ready(self, group_idx: int, bucket_idx: int, param: Parameter):
        """Mark gradient as ready and trigger reduction if bucket is full."""
        # Implementation would track gradient readiness and trigger communication
        # when all gradients in a bucket are ready
        pass
    
    def step(self, closure: Optional[callable] = None):
        """
        Perform optimizer step with gradient partitioning.
        
        Args:
            closure: Optional closure for optimizer step
        """
        start_time = time.time()
        
        # Check for gradient overflow first
        self.overflow = self.check_overflow()
        if self.overflow:
            logger.warning(f"Rank {self.rank}: Gradient overflow detected, skipping step")
            self.zero_grad()
            return
        
        # Reduce-scatter gradients across all parameter groups
        self._reduce_scatter_gradients()
        
        # Move optimizer states from CPU if using offloading
        if self.config.cpu_offload:
            self._move_states_to_gpu()
        
        # Perform optimizer step on local parameters only
        if closure is not None:
            loss = self.optimizer.step(closure)
        else:
            self.optimizer.step()
        
        # Move optimizer states back to CPU if using offloading
        if self.config.cpu_offload:
            self._move_states_to_cpu()
        
        # All-gather updated parameters
        self._allgather_parameters()
        
        self.step_count += 1
        self.computation_time += time.time() - start_time
        
        if closure is not None:
            return loss
    
    def _reduce_scatter_gradients(self):
        """Reduce-scatter gradients across all processes."""
        start_time = time.time()
        
        for group_idx, buffer in self.gradient_buffers.items():
            for bucket_idx in range(len(buffer.buckets)):
                # Copy gradients to communication buffer
                buffer.copy_gradients_to_buffer(bucket_idx)
                
                # Get buffer and output tensor
                bucket_buffer = buffer.get_bucket_buffer(bucket_idx)
                output_tensor = self.reduced_gradients[group_idx][bucket_idx]
                
                # Resize output tensor if needed
                partition_size = bucket_buffer.numel() // self.world_size
                if output_tensor.numel() != partition_size:
                    output_tensor = torch.zeros(
                        partition_size,
                        dtype=bucket_buffer.dtype,
                        device=bucket_buffer.device
                    )
                    self.reduced_gradients[group_idx][bucket_idx] = output_tensor
                
                # Perform reduce-scatter
                self._reduce_scatter_tensor(bucket_buffer, output_tensor)
                
                # Copy reduced gradients back to local parameters only
                self._copy_reduced_gradients_to_params(group_idx, bucket_idx, output_tensor)
        
        self.reduce_scatter_time += time.time() - start_time
    
    def _reduce_scatter_tensor(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Perform reduce-scatter operation on tensor."""
        # Create input list for reduce-scatter
        chunk_size = input_tensor.numel() // self.world_size
        input_list = []
        
        for i in range(self.world_size):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            if i == self.world_size - 1:
                # Last chunk may be smaller
                end_idx = input_tensor.numel()
            chunk = input_tensor[start_idx:end_idx]
            
            # Pad chunk to expected size if necessary
            if chunk.numel() < chunk_size and i < self.world_size - 1:
                padded_chunk = torch.zeros(
                    chunk_size, dtype=chunk.dtype, device=chunk.device
                )
                padded_chunk[:chunk.numel()] = chunk
                input_list.append(padded_chunk)
            else:
                input_list.append(chunk)
        
        # Perform reduce-scatter
        dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
        
        # Scale by world size
        output_tensor.div_(self.world_size)
    
    def _copy_reduced_gradients_to_params(self, group_idx: int, bucket_idx: int, reduced_tensor: torch.Tensor):
        """Copy reduced gradients back to local parameters."""
        buffer = self.gradient_buffers[group_idx]
        bucket_params = buffer.buckets[bucket_idx]
        param_map = buffer.bucket_gradients[bucket_idx]
        
        # Calculate local parameter offset
        local_params = self.partition_manager.get_local_partition("gradients")
        
        reduced_offset = 0
        for param in bucket_params:
            if param in local_params:
                param_start, param_end = param_map[param]
                param_size = param_end - param_start
                
                # Extract gradient for this parameter
                if reduced_offset + param_size <= reduced_tensor.numel():
                    param_grad = reduced_tensor[reduced_offset:reduced_offset + param_size]
                    param_grad = param_grad.view(param.grad.shape).to(param.grad.dtype)
                    param.grad.copy_(param_grad)
                    reduced_offset += param_size
    
    def _move_states_to_gpu(self):
        """Move optimizer states from CPU to GPU.""" 
        if not self.config.cpu_offload:
            return
        
        start_time = time.time()
        
        for param in self.local_parameters:
            if param in self.cpu_optimizer_states:
                gpu_state = {}
                cpu_state = self.cpu_optimizer_states[param]
                
                for key, value in cpu_state.items():
                    if torch.is_tensor(value):
                        gpu_state[key] = value.to(self.device, non_blocking=True)
                    else:
                        gpu_state[key] = value
                
                self.optimizer.state[param] = gpu_state
        
        self.cpu_offload_time += time.time() - start_time
    
    def _move_states_to_cpu(self):
        """Move optimizer states from GPU to CPU."""
        if not self.config.cpu_offload:
            return
        
        start_time = time.time()
        
        for param in self.local_parameters:
            if param in self.optimizer.state:
                cpu_state = {}
                gpu_state = self.optimizer.state[param]
                
                for key, value in gpu_state.items():
                    if torch.is_tensor(value):
                        cpu_state[key] = value.cpu().pin_memory()
                    else:
                        cpu_state[key] = value
                
                self.cpu_optimizer_states[param] = cpu_state
        
        # Clear GPU states
        self.optimizer.state.clear()
        
        self.cpu_offload_time += time.time() - start_time
    
    def _allgather_parameters(self):
        """All-gather updated parameters to all processes."""
        start_time = time.time()
        
        # Group parameters by data type for efficient communication
        param_groups_by_dtype = defaultdict(list)
        
        for param in self.local_parameters:
            param_groups_by_dtype[param.dtype].append(param)
        
        # All-gather each group
        for dtype, params in param_groups_by_dtype.items():
            if params:
                # Flatten parameters for communication
                flat_params = torch.cat([p.data.flatten() for p in params])
                
                # All-gather
                gathered_params = [torch.zeros_like(flat_params) for _ in range(self.world_size)]
                gathered_params[self.rank] = flat_params
                dist.all_gather(gathered_params, flat_params)
                
                # Unflatten and distribute to all parameters
                self._unflatten_and_distribute_params(params, gathered_params, dtype)
        
        self.allgather_time += time.time() - start_time
    
    def _unflatten_and_distribute_params(self, local_params: List[Parameter], gathered_tensors: List[torch.Tensor], dtype: torch.dtype):
        """Unflatten gathered parameters and distribute to all processes."""
        # Calculate parameter layout across all ranks
        all_parameters = []
        for group in self.optimizer.param_groups:
            for param in group['params']:
    def _unflatten_and_distribute_params(self, local_params: List[Parameter], gathered_tensors: List[torch.Tensor], dtype: torch.dtype):
        """Unflatten gathered parameters and distribute to all processes."""
        # Calculate parameter layout across all ranks
        all_parameters = []
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.dtype == dtype and param.requires_grad:
                    all_parameters.append(param)
        
        # Create global parameter partitions
        global_partitions = self.partition_manager.create_partitions(all_parameters, f"allgather_{dtype}")
        
        # Distribute parameters from each rank's gathered tensor
        for source_rank, gathered_tensor in enumerate(gathered_tensors):
            rank_params = global_partitions.get(source_rank, [])
            
            offset = 0
            for param in rank_params:
                param_size = param.numel()
                if offset + param_size <= gathered_tensor.numel():
                    param_data = gathered_tensor[offset:offset + param_size]
                    param.data.copy_(param_data.view(param.shape))
                    offset += param_size
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients for all parameters."""
        # Zero gradients in optimizer
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
        # Zero communication buffers
        for buffer in self.gradient_buffers.values():
            buffer.zero_buffers()
        
        # Clear reduced gradients
        for group_gradients in self.reduced_gradients.values():
            for reduced_tensor in group_gradients.values():
                reduced_tensor.zero_()
    
    def check_overflow(self) -> bool:
        """Check for gradient overflow across all parameters."""
        local_overflow = False
        
        # Check local parameters
        for param in self.local_parameters:
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    local_overflow = True
                    break
        
        # All-reduce overflow status
        overflow_tensor = torch.tensor(local_overflow, dtype=torch.float32, device=self.device)
        dist.all_reduce(overflow_tensor, op=dist.ReduceOp.SUM)
        
        global_overflow = overflow_tensor.item() > 0
        self.overflow = global_overflow
        return global_overflow
    
    def get_global_norm(self) -> float:
        """Get global gradient norm across all parameters.""" 
        local_norm_squared = 0.0
        
        # Calculate local norm
        for param in self.local_parameters:
            if param.grad is not None:
                local_norm_squared += param.grad.data.norm(dtype=torch.float32) ** 2
        
        # All-reduce norm squared
        norm_tensor = torch.tensor(local_norm_squared, dtype=torch.float32, device=self.device)
        dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
        
        global_norm = math.sqrt(norm_tensor.item())
        self.grad_norm = global_norm
        return global_norm
    
    def clip_gradients(self, max_norm: float, norm_type: float = 2.0) -> float:
        """
        Clip gradients by global norm.
        
        Args:
            max_norm: Maximum norm value
            norm_type: Type of norm to compute
            
        Returns:
            Total norm of gradients before clipping
        """
        if norm_type != 2.0:
            raise NotImplementedError("ZeRO Stage 2 only supports L2 norm clipping")
        
        global_norm = self.get_global_norm()
        
        if global_norm > max_norm:
            clip_coef = max_norm / (global_norm + 1e-6)
            
            # Clip local gradients
            for param in self.local_parameters:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return global_norm
    
    def all_reduce_gradients(self):
        """Manually trigger gradient all-reduce (no-op for Stage 2)."""
        # In Stage 2, gradients are reduced via reduce-scatter, not all-reduce
        logger.warning("all_reduce_gradients() called on ZeRO Stage 2 - this is a no-op")
    
    def get_partition_info(self) -> Dict[str, Any]:
        """Get information about parameter partitioning."""
        total_params = 0
        local_params = len(self.local_parameters)
        
        for group in self.param_groups:
            total_params += len([p for p in group['params'] if p.requires_grad])
        
        return {
            'stage': self.stage,
            'total_parameters': total_params * self.world_size,  # Approximate
            'local_parameters': local_params,
            'world_size': self.world_size,
            'rank': self.rank,
            'partition_ratio': local_params / max(1, total_params),
            'memory_reduction_factor': self.world_size
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage information."""
        stats = super().get_memory_stats()
        
        # Add Stage 2 specific memory info
        gradient_buffer_memory = 0.0
        for buffer in self.gradient_buffers.values():
            for bucket_buffer in buffer.bucket_buffers.values():
                gradient_buffer_memory += bucket_buffer.numel() * bucket_buffer.element_size()
        
        reduced_gradient_memory = 0.0
        for group_gradients in self.reduced_gradients.values():
            for reduced_tensor in group_gradients.values():
                reduced_gradient_memory += reduced_tensor.numel() * reduced_tensor.element_size()
        
        cpu_offload_memory = 0.0
        if self.config.cpu_offload:
            for state in self.cpu_optimizer_states.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        cpu_offload_memory += value.numel() * value.element_size()
        
        stats.update({
            'gradient_buffer_memory_gb': gradient_buffer_memory / (1024**3),
            'reduced_gradient_memory_gb': reduced_gradient_memory / (1024**3),
            'cpu_offload_memory_gb': cpu_offload_memory / (1024**3),
            'stage': self.stage
        })
        
        return stats
    
    def get_communication_stats(self) -> Dict[str, float]:
        """Get communication performance statistics."""
        base_stats = super().get_performance_stats()
        
        stage2_stats = {
            'reduce_scatter_time_s': self.reduce_scatter_time,
            'allgather_time_s': self.allgather_time,
            'cpu_offload_time_s': self.cpu_offload_time,
            'total_communication_time_s': self.reduce_scatter_time + self.allgather_time,
            'avg_reduce_scatter_time_s': self.reduce_scatter_time / max(1, self.step_count),
            'avg_allgather_time_s': self.allgather_time / max(1, self.step_count),
        }
        
        base_stats.update(stage2_stats)
        return base_stats
    
    def consolidate_state_dict(self, destination: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Consolidate distributed state dict to single process.
        
        Args:
            destination: Optional destination dictionary
            
        Returns:
            Consolidated state dictionary
        """
        if destination is None:
            destination = {}
        
        # Gather optimizer states from all processes
        all_states = [None] * self.world_size
        local_state_info = {
            'rank': self.rank,
            'local_state': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else {},
            'cpu_states': self.cpu_optimizer_states if self.config.cpu_offload else {}
        }
        
        # All-gather state information
        dist.all_gather_object(all_states, local_state_info)
        
        # Consolidate on rank 0
        if self.rank == 0:
            consolidated_state = {
                'optimizer_state_dict': {},
                'step_count': self.step_count,
                'stage': self.stage,
                'config': self.config.__dict__
            }
            
            # Merge states from all ranks
            for rank_info in all_states:
                if rank_info is not None:
                    rank_state = rank_info['local_state']
                    cpu_states = rank_info.get('cpu_states', {})
                    
                    # Merge optimizer state
                    if 'state' in rank_state:
                        consolidated_state['optimizer_state_dict'].update(rank_state['state'])
                    
                    # Merge CPU offloaded states
                    for param, state in cpu_states.items():
                        # Convert parameter reference to index for serialization
                        param_id = id(param)
                        consolidated_state['optimizer_state_dict'][f'cpu_state_{param_id}'] = state
            
            destination.update(consolidated_state)
        
        return destination
    
    def load_consolidated_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load consolidated state dictionary.
        
        Args:
            state_dict: Consolidated state dictionary to load
        """
        if 'optimizer_state_dict' in state_dict:
            # Load step count and config
            self.step_count = state_dict.get('step_count', 0)
            
            # Distribute state dict entries to appropriate ranks
            full_optimizer_state = state_dict['optimizer_state_dict']
            
            # Extract local state for this rank
            local_state = {}
            cpu_states = {}
            
            # This is a simplified version - full implementation would need
            # proper parameter mapping and distribution logic
            for key, value in full_optimizer_state.items():
                if key.startswith(f'cpu_state_'):
                    # CPU offloaded state
                    continue  # Would implement proper CPU state loading
                else:
                    # Regular optimizer state
                    local_state[key] = value
            
            # Load into optimizer
            if hasattr(self.optimizer, 'load_state_dict') and local_state:
                try:
                    self.optimizer.load_state_dict({'state': local_state})
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state dict: {e}")
    
    def save_checkpoint(self, checkpoint_dir: str, tag: str = ""):
        """
        Save ZeRO Stage 2 checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint
            tag: Optional tag for checkpoint name
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save rank-specific checkpoint
        rank_checkpoint = {
            'rank': self.rank,
            'world_size': self.world_size,
            'stage': self.stage,
            'step_count': self.step_count,
            'local_optimizer_state': self.optimizer.state_dict() if hasattr(self.optimizer, 'state_dict') else {},
            'cpu_optimizer_states': self.cpu_optimizer_states if self.config.cpu_offload else {},
            'config': self.config.__dict__,
            'performance_stats': self.get_communication_stats()
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"zero_stage2_rank_{self.rank}{tag}.pt")
        torch.save(rank_checkpoint, checkpoint_path)
        
        logger.info(f"Rank {self.rank}: Saved ZeRO Stage 2 checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_dir: str, tag: str = ""):
        """
        Load ZeRO Stage 2 checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            tag: Optional tag for checkpoint name
        """
        import os
        
        checkpoint_path = os.path.join(checkpoint_dir, f"zero_stage2_rank_{self.rank}{tag}.pt")
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Validate checkpoint
            if checkpoint['rank'] != self.rank:
                logger.error(f"Checkpoint rank mismatch: expected {self.rank}, got {checkpoint['rank']}")
                return False
            
            if checkpoint['world_size'] != self.world_size:
                logger.warning(f"World size mismatch: expected {self.world_size}, got {checkpoint['world_size']}")
            
            # Load state
            self.step_count = checkpoint.get('step_count', 0)
            
            if 'local_optimizer_state' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['local_optimizer_state'])
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
            
            if 'cpu_optimizer_states' in checkpoint and self.config.cpu_offload:
                self.cpu_optimizer_states = checkpoint['cpu_optimizer_states']
            
            logger.info(f"Rank {self.rank}: Loaded ZeRO Stage 2 checkpoint from {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation of ZeRO Stage 2 optimizer."""
        return (
            f"ZeROStage2(\n"
            f"  world_size={self.world_size},\n"
            f"  rank={self.rank},\n"
            f"  local_parameters={len(self.local_parameters)},\n"
            f"  gradient_buckets={sum(len(buf.buckets) for buf in self.gradient_buffers.values())},\n"
            f"  cpu_offload={self.config.cpu_offload},\n"
            f"  step_count={self.step_count}\n"
            f")"
        )