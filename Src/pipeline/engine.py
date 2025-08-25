"""
Pipeline Parallelism Engine for DistributedSpeed.

This module implements advanced pipeline parallelism with support for:
- Multiple scheduling strategies (GPipe, PipeDream-1F1B, Interleaved 1F1B)
- Automatic model partitioning
- Memory-efficient activation checkpointing
- Load balancing across pipeline stages
- Fault tolerance and recovery
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from collections import defaultdict, deque
from enum import Enum
import threading
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import PipelineConfig
from .schedule import PipelineScheduler, GPipeScheduler, PipeDream1F1BScheduler, InterleavedScheduler
from .partition import ModelPartitioner, UniformPartitioner, ProfileBasedPartitioner
from .communication import PipelineCommunicator
from ..utils.logging import get_logger
from ..utils.timer import Timer

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline stage types."""
    FORWARD_ONLY = "forward_only"
    BACKWARD_ONLY = "backward_only"
    FORWARD_BACKWARD = "forward_backward"


class ActivationCheckpointFunction(torch.autograd.Function):
    """Custom activation checkpointing function for pipeline parallelism."""
    
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        """Forward pass with activation checkpointing."""
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.args = args
        
        # Save RNG state if required
        if preserve_rng_state:
            ctx.fwd_cpu_rng_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        
        # Run forward pass without gradients
        with torch.no_grad():
            outputs = run_function(*args)
        
        return outputs
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass with recomputation.""" 
        args = ctx.args
        
        # Restore RNG state for deterministic recomputation
        if ctx.preserve_rng_state:
            rng_devices = []
            if hasattr(ctx, 'fwd_cuda_rng_state'):
                torch.cuda.set_rng_state(ctx.fwd_cuda_rng_state)
            torch.set_rng_state(ctx.fwd_cpu_rng_state)
        
        # Recompute forward pass with gradients
        detached_args = tuple(arg.detach().requires_grad_(arg.requires_grad) for arg in args)
        
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_args)
        
        # Ensure outputs is a tuple
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        
        # Compute gradients
        torch.autograd.backward(outputs, grad_outputs)
        
        grads = tuple(arg.grad if arg.requires_grad else None for arg in detached_args)
        return (None, None) + grads


class PipelineEngine:
    """
    Advanced Pipeline Parallelism Engine.
    
    This engine manages the execution of pipeline parallel training with
    sophisticated scheduling, partitioning, and communication strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PipelineConfig,
        mpu: Optional[Any] = None,
        loss_fn: Optional[Callable] = None
    ):
        """
        Initialize pipeline parallelism engine.
        
        Args:
            model: PyTorch model to partition
            config: Pipeline configuration
            mpu: Model parallel utilities
            loss_fn: Loss function for training
        """
        self.config = config
        self.mpu = mpu
        self.loss_fn = loss_fn
        
        # Initialize distributed environment
        if not dist.is_initialized():
            raise RuntimeError("Distributed training must be initialized")
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        
        # Pipeline topology
        self.num_stages = config.stages
        self.stage_id = self.rank % self.num_stages
        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = self.stage_id == self.num_stages - 1
        
        # Data parallel groups (multiple pipelines)
        self.dp_size = self.world_size // self.num_stages
        self.dp_rank = self.rank // self.num_stages
        
        # Model partitioning
        self.partitioner = self._create_partitioner()
        self.model_part = self._partition_model(model)
        
        # Communication
        self.communicator = PipelineCommunicator(
            rank=self.rank,
            world_size=self.world_size,
            num_stages=self.num_stages,
            stage_id=self.stage_id
        )
        
        # Scheduling
        self.scheduler = self._create_scheduler()
        
        # Pipeline buffers and state
        self.activation_buffer = {}
        self.gradient_buffer = {}
        self.micro_batch_queue = deque()
        
        # Performance tracking
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.communication_time = 0.0
        self.bubble_time = 0.0
        
        # Activation checkpointing
        self.checkpoint_activations = config.activation_checkpoint_interval > 0
        self.checkpoint_interval = config.activation_checkpoint_interval
        
        # Load balancing
        self.load_balancer = LoadBalancer(self.num_stages) if config.load_balance else None
        
        logger.info(
            f"Pipeline engine initialized: rank {self.rank}, stage {self.stage_id}/{self.num_stages}, "
            f"dp_rank {self.dp_rank}/{self.dp_size}"
        )
    
    def _create_partitioner(self) -> ModelPartitioner:
        """Create model partitioner based on configuration."""
        if self.config.partition == "uniform":
            return UniformPartitioner(self.num_stages)
        elif self.config.partition == "profile":
            return ProfileBasedPartitioner(self.num_stages)
        else:
            raise ValueError(f"Unknown partition strategy: {self.config.partition}")
    
    def _partition_model(self, model: nn.Module) -> nn.Module:
        """Partition model for current pipeline stage."""
        # Get partition assignment
        partitions = self.partitioner.partition(model)
        
        if self.stage_id >= len(partitions):
            raise ValueError(f"Stage {self.stage_id} exceeds number of partitions {len(partitions)}")
        
        # Get model part for this stage
        model_part = partitions[self.stage_id]
        
        # Move to device
        model_part = model_part.to(self.device)
        
        # Wrap with DDP if multiple data parallel groups
        if self.dp_size > 1:
            model_part = DDP(
                model_part,
                device_ids=[self.device],
                process_group=self.communicator.get_data_parallel_group()
            )
        
        return model_part
    
    def _create_scheduler(self) -> PipelineScheduler:
        """Create pipeline scheduler based on configuration."""
        if self.config.schedule == "gpipe":
            return GPipeScheduler(
                num_stages=self.num_stages,
                stage_id=self.stage_id,
                num_microbatches=self._calculate_num_microbatches()
            )
        elif self.config.schedule == "1f1b":
            return PipeDream1F1BScheduler(
                num_stages=self.num_stages,
                stage_id=self.stage_id,
                num_microbatches=self._calculate_num_microbatches()
            )
        elif self.config.schedule == "interleaved_1f1b":
            return InterleavedScheduler(
                num_stages=self.num_stages,
                stage_id=self.stage_id,
                num_microbatches=self._calculate_num_microbatches(),
                num_model_chunks=2  # Default to 2 chunks for interleaving
            )
        else:
            raise ValueError(f"Unknown schedule: {self.config.schedule}")
    
    def _calculate_num_microbatches(self) -> int:
        """Calculate number of microbatches based on configuration.""" 
        if self.config.micro_batch_size is not None:
            # Calculate based on micro batch size
            return max(1, self.num_stages * 4)  # Default heuristic
        else:
            # Use default based on number of stages
            return max(4 * self.num_stages, 16)
    
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Execute forward pass through pipeline.
        
        Args:
            inputs: Input tensor batch
            **kwargs: Additional keyword arguments
            
        Returns:
            Output tensor from final stage
        """
        batch_size = inputs.size(0)
        micro_batch_size = self.config.micro_batch_size or (batch_size // self.num_stages)
        
        # Split into microbatches
        microbatches = self._split_batch(inputs, micro_batch_size)
        
        # Execute pipeline schedule
        outputs = self._execute_forward_schedule(microbatches, **kwargs)
        
        # Return outputs from last stage
        if self.is_last_stage:
            return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        else:
            return None
    
    def backward(self, loss: Optional[torch.Tensor] = None, **kwargs):
        """
        Execute backward pass through pipeline.
        
        Args:
            loss: Loss tensor (only used by last stage)
            **kwargs: Additional keyword arguments
        """
        if self.is_last_stage and loss is not None:
            # Split loss into microbatches if needed
            if loss.dim() > 0 and loss.size(0) > 1:
                micro_losses = torch.chunk(loss, loss.size(0), dim=0)
            else:
                micro_losses = [loss]
        else:
            micro_losses = None
        
        # Execute backward schedule
        self._execute_backward_schedule(micro_losses, **kwargs)
    
    def _split_batch(self, batch: torch.Tensor, micro_batch_size: int) -> List[torch.Tensor]:
        """Split batch into microbatches."""
        if batch.size(0) <= micro_batch_size:
            return [batch]
        
        microbatches = []
        for i in range(0, batch.size(0), micro_batch_size):
            end_idx = min(i + micro_batch_size, batch.size(0))
            microbatch = batch[i:end_idx]
            microbatches.append(microbatch)
        
        return microbatches
    
    def _execute_forward_schedule(self, microbatches: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        """Execute forward pass according to pipeline schedule."""
        outputs = []
        
        # Get schedule from scheduler
        schedule = self.scheduler.get_forward_schedule(len(microbatches))
        
        for step in schedule:
            if step['action'] == 'forward':
                micro_batch_id = step['micro_batch_id']
                
                if micro_batch_id < len(microbatches):
                    # Forward pass for this microbatch
                    output = self._forward_micro_batch(
                        microbatches[micro_batch_id],
                        micro_batch_id,
                        **kwargs
                    )
                    
                    if self.is_last_stage:
                        outputs.append(output)
            
            elif step['action'] == 'recv':
                # Receive activation from previous stage
                activation = self.communicator.recv_forward(step['micro_batch_id'])
                self.activation_buffer[step['micro_batch_id']] = activation
            
            elif step['action'] == 'send':
                # Send activation to next stage
                if step['micro_batch_id'] in self.activation_buffer:
                    self.communicator.send_forward(
                        self.activation_buffer[step['micro_batch_id']],
                        step['micro_batch_id']
                    )
        
        return outputs
    
    def _execute_backward_schedule(self, micro_losses: Optional[List[torch.Tensor]], **kwargs):
        """Execute backward pass according to pipeline schedule.""" 
        # Get schedule from scheduler
        num_microbatches = len(micro_losses) if micro_losses else self._calculate_num_microbatches()
        schedule = self.scheduler.get_backward_schedule(num_microbatches)
        
        for step in schedule:
            if step['action'] == 'backward':
                micro_batch_id = step['micro_batch_id']
                
                # Get loss for this microbatch
                if micro_losses and micro_batch_id < len(micro_losses):
                    loss = micro_losses[micro_batch_id]
                else:
                    loss = None
                
                # Backward pass for this microbatch
                self._backward_micro_batch(micro_batch_id, loss, **kwargs)
            
            elif step['action'] == 'recv':
                # Receive gradient from next stage
                gradient = self.communicator.recv_backward(step['micro_batch_id'])
                self.gradient_buffer[step['micro_batch_id']] = gradient
            
            elif step['action'] == 'send':
                # Send gradient to previous stage
                if step['micro_batch_id'] in self.gradient_buffer:
                    self.communicator.send_backward(
                        self.gradient_buffer[step['micro_batch_id']],
                        step['micro_batch_id']
                    )
    
    def _forward_micro_batch(
        self,
        micro_batch: torch.Tensor,
        micro_batch_id: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for single microbatch.
        
        Args:
            micro_batch: Input microbatch
            micro_batch_id: Microbatch identifier
            **kwargs: Additional arguments
            
        Returns:
            Output tensor
        """
    def _backward_micro_batch(
        self,
        micro_batch_id: int,
        loss: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Backward pass for single microbatch.
        
        Args:
            micro_batch_id: Microbatch identifier
            loss: Loss tensor (only for last stage)
            **kwargs: Additional arguments
        """
        start_time = time.time()
        
        # Get stored activation for this microbatch
        if micro_batch_id not in self.activation_buffer:
            logger.warning(f"No activation found for microbatch {micro_batch_id}")
            return
        
        activation = self.activation_buffer[micro_batch_id]
        
        if self.is_last_stage:
            # Last stage: compute loss and start backward pass
            if loss is not None:
                if self.loss_fn is not None:
                    # Apply loss function if provided
                    computed_loss = self.loss_fn(activation, loss)
                else:
                    computed_loss = loss
                
                # Start backward pass
                computed_loss.backward()
        else:
            # Middle/first stages: get gradient from next stage
            if micro_batch_id in self.gradient_buffer:
                grad_output = self.gradient_buffer[micro_batch_id]
            else:
                grad_output = self.communicator.recv_backward(micro_batch_id)
            
            # Backward pass with received gradients
            torch.autograd.backward(activation, grad_output)
        
        # Send gradients to previous stage
        if not self.is_first_stage and activation.grad is not None:
            self.communicator.send_backward(activation.grad, micro_batch_id)
        
        # Clean up activation buffer
        del self.activation_buffer[micro_batch_id]
        if micro_batch_id in self.gradient_buffer:
            del self.gradient_buffer[micro_batch_id]
        
        self.backward_time += time.time() - start_time
    
    def _checkpoint_forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with activation checkpointing."""
        return ActivationCheckpointFunction.apply(
            self._run_forward_pass,
            True,  # preserve_rng_state
            inputs,
            **kwargs
        )
    
    def _run_forward_pass(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run forward pass through model part."""
        return self.model_part(inputs, **kwargs)
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step across pipeline.
        
        Args:
            optimizer: Optimizer instance
        """
        # Synchronize gradients across data parallel group if needed
        if self.dp_size > 1:
            self._sync_data_parallel_gradients()
        
        # Perform optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Clear any remaining buffers
        self.activation_buffer.clear()
        self.gradient_buffer.clear()
    
    def _sync_data_parallel_gradients(self):
        """Synchronize gradients across data parallel replicas."""
        # All-reduce gradients within data parallel group
        dp_group = self.communicator.get_data_parallel_group()
        
        for param in self.model_part.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=dp_group)
                param.grad.div_(self.dp_size)
    
    def train(self):
        """Set pipeline to training mode."""
        self.model_part.train()
    
    def eval(self):
        """Set pipeline to evaluation mode."""
        self.model_part.eval()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get pipeline state dictionary."""
        return {
            'model_part_state_dict': self.model_part.state_dict(),
            'stage_id': self.stage_id,
            'num_stages': self.num_stages,
            'forward_time': self.forward_time,
            'backward_time': self.backward_time,
            'communication_time': self.communication_time,
            'config': self.config.__dict__
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load pipeline state dictionary."""
        if 'model_part_state_dict' in state_dict:
            self.model_part.load_state_dict(state_dict['model_part_state_dict'])
        
        self.forward_time = state_dict.get('forward_time', 0.0)
        self.backward_time = state_dict.get('backward_time', 0.0)
        self.communication_time = state_dict.get('communication_time', 0.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        total_time = self.forward_time + self.backward_time + self.communication_time
        
        return {
            'stage_id': self.stage_id,
            'forward_time_s': self.forward_time,
            'backward_time_s': self.backward_time,
            'communication_time_s': self.communication_time,
            'bubble_time_s': self.bubble_time,
            'total_time_s': total_time,
            'compute_efficiency': (self.forward_time + self.backward_time) / max(total_time, 1e-6),
            'communication_overhead': self.communication_time / max(total_time, 1e-6)
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get pipeline memory usage statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
        else:
            allocated = reserved = 0.0
        
        # Estimate activation buffer memory
        activation_memory = 0.0
        for activation in self.activation_buffer.values():
            if torch.is_tensor(activation):
                activation_memory += activation.numel() * activation.element_size()
        
        return {
            'stage_id': self.stage_id,
            'gpu_memory_allocated_gb': allocated,
            'gpu_memory_reserved_gb': reserved,
            'activation_buffer_gb': activation_memory / (1024**3),
            'num_buffered_activations': len(self.activation_buffer)
        }
    
    def profile_stage_execution(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Profile execution time for current pipeline stage.
        
        Args:
            num_samples: Number of forward passes to profile
            
        Returns:
            Profiling results dictionary
        """
        if not hasattr(self, '_dummy_input'):
            # Create dummy input for profiling
            dummy_size = self.config.micro_batch_size or 4
            if hasattr(self.model_part, 'input_shape'):
                input_shape = (dummy_size,) + self.model_part.input_shape
            else:
                input_shape = (dummy_size, 128)  # Default shape
            
            self._dummy_input = torch.randn(input_shape, device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model_part(self._dummy_input)
        
        # Profile forward pass
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_samples):
            with torch.no_grad():
                output = self.model_part(self._dummy_input)
        
        torch.cuda.synchronize()
        forward_time = (time.time() - start_time) / num_samples
        
        # Profile backward pass
        output = self.model_part(self._dummy_input)
        dummy_loss = output.sum()
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_samples):
            output = self.model_part(self._dummy_input)
            loss = output.sum()
            loss.backward()
            self.model_part.zero_grad()
        
        torch.cuda.synchronize()
        backward_time = (time.time() - start_time) / num_samples
        
        return {
            'stage_id': self.stage_id,
            'forward_time_ms': forward_time * 1000,
            'backward_time_ms': backward_time * 1000,
            'total_time_ms': (forward_time + backward_time) * 1000,
            'throughput_samples_per_sec': num_samples / (forward_time + backward_time)
        }
    
    def balance_load(self, target_time: Optional[float] = None):
        """
        Balance load across pipeline stages.
        
        Args:
            target_time: Target execution time per stage
        """
        if self.load_balancer is None:
            return
        
        # Profile current stage
        profile_results = self.profile_stage_execution()
        
        # Gather profiles from all stages
        all_profiles = [None] * self.num_stages
        dist.all_gather_object(all_profiles, profile_results)
        
        if self.stage_id == 0:
            # Compute load balancing on first stage
            rebalancing_plan = self.load_balancer.compute_rebalancing(all_profiles, target_time)
            
            # Broadcast rebalancing plan
            for stage_id in range(self.num_stages):
                if stage_id != 0:
                    dist.send_object(rebalancing_plan[stage_id], dst=stage_id)
        else:
            # Receive rebalancing plan
            plan = dist.recv_object(src=0)
            
            # Apply rebalancing if needed
            if plan.get('action') == 'repartition':
                self._repartition_model(plan['new_layers'])
    
    def _repartition_model(self, new_layer_assignment: List[int]):
        """
        Repartition model based on new layer assignment.
        
        Args:
            new_layer_assignment: List of layer indices for this stage
        """
        # This would implement dynamic model repartitioning
        # For now, log the intended action
        logger.info(
            f"Stage {self.stage_id}: Repartitioning to layers {new_layer_assignment}"
        )
    
    def checkpoint_activation_memory(self) -> Dict[str, Any]:
        """Create checkpoint of activation memory state."""
        checkpoint = {
            'stage_id': self.stage_id,
            'activation_shapes': {},
            'buffer_sizes': {}
        }
        
        for micro_batch_id, activation in self.activation_buffer.items():
            if torch.is_tensor(activation):
                checkpoint['activation_shapes'][micro_batch_id] = activation.shape
                checkpoint['buffer_sizes'][micro_batch_id] = activation.numel() * activation.element_size()
        
        return checkpoint
    
    def estimate_pipeline_efficiency(self) -> Dict[str, float]:
        """Estimate pipeline efficiency metrics."""
        stats = self.get_performance_stats()
        
        # Theoretical maximum efficiency (no bubbles)
        theoretical_efficiency = 1.0
        
        # Actual efficiency considering bubbles
        compute_time = stats['forward_time_s'] + stats['backward_time_s']
        total_time = stats['total_time_s']
        actual_efficiency = compute_time / max(total_time, 1e-6)
        
        # Pipeline bubble ratio
        bubble_ratio = stats['bubble_time_s'] / max(total_time, 1e-6)
        
        return {
            'theoretical_efficiency': theoretical_efficiency,
            'actual_efficiency': actual_efficiency,
            'efficiency_loss': theoretical_efficiency - actual_efficiency,
            'bubble_ratio': bubble_ratio,
            'communication_ratio': stats['communication_overhead']
        }
    
    def cleanup(self):
        """Clean up pipeline resources.""" 
        # Clear buffers
        self.activation_buffer.clear()
        self.gradient_buffer.clear()
        self.micro_batch_queue.clear()
        
        # Cleanup communicator
        if hasattr(self.communicator, 'cleanup'):
            self.communicator.cleanup()
        
        # Clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Pipeline stage {self.stage_id} cleaned up successfully")
    
    def __repr__(self) -> str:
        """String representation of pipeline engine."""
        return (
            f"PipelineEngine(\n"
            f"  stage_id={self.stage_id}/{self.num_stages},\n"
            f"  dp_rank={self.dp_rank}/{self.dp_size},\n"
            f"  schedule={self.config.schedule},\n"
            f"  partition={self.config.partition},\n"
            f"  checkpoint_activations={self.checkpoint_activations}\n"
            f")"
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


class LoadBalancer:
    """
    Load balancer for pipeline parallelism.
    
    Automatically balances computational load across pipeline stages
    to minimize pipeline bubbles and improve efficiency.
    """
    
    def __init__(self, num_stages: int):
        """
        Initialize load balancer.
        
        Args:
            num_stages: Number of pipeline stages
        """
        self.num_stages = num_stages
        self.stage_profiles = {}
        self.target_time = None
    
    def compute_rebalancing(
        self,
        profiles: List[Dict[str, Any]],
        target_time: Optional[float] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute rebalancing plan for pipeline stages.
        
        Args:
            profiles: Performance profiles from all stages
            target_time: Target execution time per stage
            
        Returns:
            Rebalancing plan for each stage
        """
        if target_time is None:
            # Use average time as target
            total_times = [p['total_time_ms'] for p in profiles if p is not None]
            target_time = sum(total_times) / len(total_times) if total_times else 100.0
        
        rebalancing_plan = {}
        
        for stage_id, profile in enumerate(profiles):
            if profile is None:
                continue
            
            current_time = profile['total_time_ms']
            time_ratio = current_time / target_time
            
            if time_ratio > 1.2:  # Stage is too slow
                plan = {
                    'action': 'reduce_load',
                    'target_reduction': (time_ratio - 1.0) * 0.5,
                    'suggested_layer_reduction': max(1, int((time_ratio - 1.0) * 2))
                }
            elif time_ratio < 0.8:  # Stage is too fast
                plan = {
                    'action': 'increase_load',
                    'target_increase': (1.0 - time_ratio) * 0.5,
                    'suggested_layer_increase': max(1, int((1.0 - time_ratio) * 2))
                }
            else:  # Stage is balanced
                plan = {'action': 'no_change'}
            
            rebalancing_plan[stage_id] = plan
        
        return rebalancing_plan
    
    def get_load_distribution(self, profiles: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze current load distribution across stages.
        
        Args:
            profiles: Performance profiles from all stages
            
        Returns:
            Load distribution statistics
        """
        if not profiles or all(p is None for p in profiles):
            return {'balance_score': 0.0, 'max_imbalance': 1.0}
        
        valid_profiles = [p for p in profiles if p is not None]
        times = [p['total_time_ms'] for p in valid_profiles]
        
        if not times:
            return {'balance_score': 0.0, 'max_imbalance': 1.0}
        
        mean_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # Balance score: 1.0 = perfectly balanced, 0.0 = completely imbalanced
        if max_time > 0:
            balance_score = min_time / max_time
        else:
            balance_score = 1.0
        
        # Maximum imbalance ratio
        max_imbalance = max_time / mean_time if mean_time > 0 else 1.0
        
        return {
            'balance_score': balance_score,
            'max_imbalance': max_imbalance,
            'mean_time_ms': mean_time,
            'max_time_ms': max_time,
            'min_time_ms': min_time,
            'time_variance': sum((t - mean_time) ** 2 for t in times) / len(times)
        }.time()
        
        # Get input (either from previous stage or original input)
        if not self.is_first_stage:
            # Wait for input from previous stage
            if micro_batch_id in self.activation_buffer:
                inputs = self.activation_buffer[micro_batch_id]
            else:
                inputs = self.communicator.recv_forward(micro_batch_id)
        else:
            inputs = micro_batch
        
        # Apply activation checkpointing if enabled
        if self.checkpoint_activations and micro_batch_id % self.checkpoint_interval == 0:
            outputs = self._checkpoint_forward(inputs, **kwargs)
        else:
            outputs = self.model_part(inputs, **kwargs)
        
        # Send to next stage or return
        if not self.is_last_stage:
            self.communicator.send_forward(outputs, micro_batch_id)
            # Store activation for backward pass
            self.activation_buffer[micro_batch_id] = outputs.detach().requires_grad_(True)
        
        self.forward_time += time.time() - start_time
        return outputs
    
    def _backward_micro_batch(
        self,
        micro_batch_id: int,
        loss: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Backward pass for single microbatch.
        
        Args:
            micro_batch_id: Microbatch identifier
            loss: Loss tensor (only for last stage)
            **kwargs: Additional arguments
        """
        start_time = time