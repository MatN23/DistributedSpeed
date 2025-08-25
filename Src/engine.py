"""
DistributedSpeed Engine - Main distributed training engine.

This module provides the core DistributedSpeedEngine class that orchestrates
distributed training with advanced optimizations including ZeRO, pipeline parallelism,
gradient compression, and memory optimization.

The engine handles:
- Model wrapping and distributed setup
- Optimizer creation and wrapping 
- Forward/backward pass coordination
- Gradient synchronization and optimization
- Checkpointing and state management
- Memory and communication optimization

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import math
import time
import logging
import warnings
from typing import Dict, Any, Optional, Union, List, Tuple, Iterator
from contextlib import contextmanager
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from .config import DistributedSpeedConfig
from .zero import ZeROOptimizer, ZeROConfig
from .pipeline import PipelineEngine, PipelineConfig  
from .communication import CommManager, AllReduceCoalesced
from .memory import MemoryManager, ActivationCheckpointing
from .optimizers import create_optimizer
from .schedulers import create_scheduler
from .monitoring import ThroughputMonitor, MemoryMonitor
from .utils import get_rank, get_world_size, get_local_rank, print_rank_0, log_dist

logger = logging.getLogger(__name__)


class DistributedSpeedEngine:
    """
    Main DistributedSpeed training engine.
    
    This class orchestrates distributed training with advanced optimizations including:
    - ZeRO optimizer state partitioning (Stages 1, 2, 3)
    - Pipeline parallelism for large models  
    - Gradient compression and communication optimization
    - Mixed precision training (FP16/BF16)
    - Activation checkpointing for memory efficiency
    - Dynamic loss scaling and gradient clipping
    - Comprehensive monitoring and profiling
    
    Args:
        model: PyTorch model to train
        config: DistributedSpeed configuration
        comm_manager: Communication manager for distributed operations
        memory_manager: Memory manager for optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedSpeedConfig,
        comm_manager: Optional[CommManager] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        self.config = config
        self.model = model
        self.training = True
        self.global_step = 0
        self.global_samples = 0
        self.skipped_steps = 0
        
        # Distributed setup
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.local_rank = get_local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Configuration validation
        self._validate_config()
        
        # Initialize managers
        self.comm_manager = comm_manager or CommManager(config)
        self.memory_manager = memory_manager or MemoryManager(config)
        
        # Training configuration
        self.train_batch_size = config.train_batch_size
        self.train_micro_batch_size_per_gpu = getattr(config, 'train_micro_batch_size_per_gpu', None)
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        
        # Calculate micro batch size if not provided
        if self.train_micro_batch_size_per_gpu is None:
            self.train_micro_batch_size_per_gpu = max(1, self.train_batch_size // (self.world_size * self.gradient_accumulation_steps))
        
        # Mixed precision setup
        self.fp16_enabled = getattr(config, 'fp16', {}).get('enabled', False)
        self.bf16_enabled = getattr(config, 'bf16', {}).get('enabled', False)
        self.amp_enabled = self.fp16_enabled or self.bf16_enabled
        
        if self.fp16_enabled and self.bf16_enabled:
            raise ValueError("Cannot enable both FP16 and BF16 simultaneously")
        
        # Gradient scaling for FP16
        self.grad_scaler = None
        if self.fp16_enabled:
            initial_scale = getattr(config, 'fp16', {}).get('initial_scale_power', 16)
            self.grad_scaler = GradScaler(
                init_scale=2**initial_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=getattr(config, 'fp16', {}).get('loss_scale_window', 1000)
            )
        
        # Gradient clipping
        self.max_grad_norm = getattr(config, 'gradient_clipping', 1.0)
        
        # ZeRO optimization setup
        self.zero_optimization = getattr(config, 'zero_optimization', {})
        self.zero_stage = self.zero_optimization.get('stage', 0)
        
        # Pipeline parallelism setup  
        self.pipeline_parallelism = getattr(config, 'pipeline_parallelism_size', 1) > 1
        if self.pipeline_parallelism:
            self.pipeline_engine = PipelineEngine(model, config)
            self.model = self.pipeline_engine.model
        
        # Model preparation
        self._prepare_model()
        
        # Optimizer and scheduler (will be set during initialization)
        self.optimizer = None
        self.lr_scheduler = None
        
        # Activation checkpointing
        if getattr(config, 'activation_checkpointing', {}).get('enabled', False):
            self.activation_checkpointing = ActivationCheckpointing(config)
            self.model = self.activation_checkpointing.apply(self.model)
        
        # Monitoring setup
        self.throughput_monitor = ThroughputMonitor()
        self.memory_monitor = MemoryMonitor()
        
        # State tracking
        self.step_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optim_time = 0.0
        
        # Loss tracking
        self.loss_scale = 1.0
        self.overflow_count = 0
        
        # Profiling
        self.profile_step = getattr(config, 'profiling', {}).get('profile_step', -1)
        self.profiler = None
        
        log_dist(f"DistributedSpeed engine initialized: "
                f"world_size={self.world_size}, zero_stage={self.zero_stage}, "
                f"pipeline_stages={getattr(config, 'pipeline_parallelism_size', 1)}, "
                f"fp16={self.fp16_enabled}, bf16={self.bf16_enabled}")
    
    def _validate_config(self):
        """Validate configuration settings."""
        
        if self.config.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        
        if hasattr(self.config, 'gradient_accumulation_steps'):
            if self.config.gradient_accumulation_steps <= 0:
                raise ValueError("gradient_accumulation_steps must be positive")
        
        # Validate ZeRO configuration
        zero_config = getattr(self.config, 'zero_optimization', {})
        zero_stage = zero_config.get('stage', 0)
        if zero_stage not in [0, 1, 2, 3]:
            raise ValueError(f"ZeRO stage must be 0, 1, 2, or 3, got {zero_stage}")
        
        # Validate mixed precision settings
        fp16_config = getattr(self.config, 'fp16', {})
        if fp16_config.get('enabled', False):
            if not torch.cuda.is_available():
                raise ValueError("FP16 training requires CUDA")
            
        # Validate pipeline parallelism
        pipeline_size = getattr(self.config, 'pipeline_parallelism_size', 1)
        if pipeline_size > 1:
            if self.world_size % pipeline_size != 0:
                raise ValueError(f"World size {self.world_size} must be divisible by pipeline size {pipeline_size}")
    
    def _prepare_model(self):
        """Prepare model for distributed training."""
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Apply model compilation if requested
        if getattr(self.config, 'compile', False):
            if hasattr(torch, 'compile'):
                log_dist("Compiling model with torch.compile")
                compile_kwargs = getattr(self.config, 'compile_kwargs', {})
                self.model = torch.compile(self.model, **compile_kwargs)
            else:
                warnings.warn("torch.compile not available, skipping model compilation")
        
        # Convert to appropriate precision
        if self.bf16_enabled:
            self.model = self.model.to(torch.bfloat16)
        elif self.fp16_enabled and not self.zero_stage:
            # For ZeRO, precision conversion is handled by the optimizer
            self.model = self.model.to(torch.float16)
        
        # Setup gradient checkpointing if enabled
        gradient_checkpointing = getattr(self.config, 'gradient_checkpointing', False)
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            log_dist("Enabled gradient checkpointing")
    
    def create_optimizer(self, parameters: Optional[Iterator[nn.Parameter]] = None) -> torch.optim.Optimizer:
        """Create and wrap optimizer with ZeRO if needed."""
        
        if parameters is None:
            parameters = self.model.parameters()
        
        # Create base optimizer
        optimizer_config = getattr(self.config, 'optimizer', {})
        optimizer = create_optimizer(parameters, optimizer_config)
        
        # Wrap with ZeRO if enabled
        if self.zero_stage > 0:
            zero_config = ZeROConfig(self.zero_optimization)
            optimizer = ZeROOptimizer(
                optimizer=optimizer,
                config=zero_config,
                model_parameters=list(self.model.parameters()),
                comm_manager=self.comm_manager
            )
            log_dist(f"Created ZeRO Stage {self.zero_stage} optimizer")
        
        self.optimizer = optimizer
        return optimizer
    
    def wrap_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """Wrap existing optimizer with ZeRO if needed."""
        
        if self.zero_stage > 0:
            zero_config = ZeROConfig(self.zero_optimization)
            optimizer = ZeROOptimizer(
                optimizer=optimizer,
                config=zero_config,
                model_parameters=list(self.model.parameters()),
                comm_manager=self.comm_manager
            )
            log_dist(f"Wrapped optimizer with ZeRO Stage {self.zero_stage}")
        
        self.optimizer = optimizer
        return optimizer
    
    def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create learning rate scheduler."""
        
        scheduler_config = getattr(self.config, 'scheduler', None)
        if scheduler_config is None:
            return None
        
        self.lr_scheduler = create_scheduler(optimizer, scheduler_config)
        log_dist(f"Created LR scheduler: {scheduler_config.get('type', 'unknown')}")
        return self.lr_scheduler
    
    def create_dataloader(
        self,
        dataset,
        collate_fn: Optional[callable] = None,
        shuffle: bool = True
    ) -> DataLoader:
        """Create distributed dataloader."""
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
        
        # Calculate effective batch size per GPU
        batch_size = self.train_micro_batch_size_per_gpu
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            pin_memory=getattr(self.config, 'dataloader_pin_memory', True),
            drop_last=getattr(self.config, 'dataloader_drop_last', True)
        )
        
        log_dist(f"Created dataloader: batch_size={batch_size}, num_workers={dataloader.num_workers}")
        return dataloader
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        
        start_time = time.time()
        
        # Set training mode
        self.model.train(self.training)
        
        # Use automatic mixed precision if enabled
        precision_context = self._get_precision_context()
        
        with precision_context:
            if self.pipeline_parallelism:
                # Pipeline forward pass
                outputs = self.pipeline_engine.forward(*args, **kwargs)
            else:
                # Standard forward pass
                outputs = self.model(*args, **kwargs)
        
        self.forward_time = time.time() - start_time
        
        # Update throughput monitoring
        batch_size = self._get_batch_size(args, kwargs)
        self.throughput_monitor.update_forward(batch_size, self.forward_time)
        
        return outputs
    
    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """Backward pass with gradient scaling and accumulation."""
        
        start_time = time.time()
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        
        # Apply gradient scaling for mixed precision
        if self.grad_scaler is not None:
            scaled_loss = self.grad_scaler.scale(scaled_loss)
        
        # Backward pass
        if self.pipeline_parallelism:
            self.pipeline_engine.backward(scaled_loss, retain_graph=retain_graph)
        else:
            scaled_loss.backward(retain_graph=retain_graph)
        
        self.backward_time = time.time() - start_time
        
        # Update throughput monitoring
        batch_size = self._get_current_batch_size()
        self.throughput_monitor.update_backward(batch_size, self.backward_time)
    
    def step(self) -> bool:
        """Optimizer step with gradient synchronization and clipping."""
        
        start_time = time.time()
        
        # Check if we should skip this step due to gradient accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps != 0:
            self.global_step += 1
            return True
        
        # Gradient synchronization for DDP-style training
        if not self.zero_stage and self.world_size > 1:
            self._synchronize_gradients()
        
        # Gradient clipping and scaling
        should_step = self._prepare_gradients()
        
        if should_step:
            # Optimizer step
            if self.grad_scaler is not None:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            
            # Learning rate scheduler step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Clear gradients
            self.optimizer.zero_grad()
            
        else:
            # Skip step due to overflow or other issues
            self.skipped_steps += 1
            log_dist(f"Skipped step {self.global_step} due to gradient overflow", "WARNING")
        
        self.global_step += 1
        self.optim_time = time.time() - start_time
        
        # Update monitoring
        self._update_monitoring()
        
        # Profiling
        if self.profile_step > 0 and self.global_step == self.profile_step:
            self._start_profiling()
        
        return should_step
    
    def _get_precision_context(self):
        """Get precision context for forward pass."""
        
        if self.amp_enabled and torch.cuda.is_available():
            dtype = torch.float16 if self.fp16_enabled else torch.bfloat16
            return autocast(device_type='cuda', dtype=dtype)
        else:
            return contextmanager(lambda: (yield))()
    
    def _get_batch_size(self, args: tuple, kwargs: dict) -> int:
        """Extract batch size from forward arguments."""
        
        # Try to get batch size from first argument (common pattern)
        if args and hasattr(args[0], 'shape'):
            return args[0].shape[0]
        
        # Try common keyword arguments
        for key in ['input_ids', 'inputs', 'x', 'data']:
            if key in kwargs and hasattr(kwargs[key], 'shape'):
                return kwargs[key].shape[0]
        
        # Default to configured batch size
        return self.train_micro_batch_size_per_gpu
    
    def _get_current_batch_size(self) -> int:
        """Get current effective batch size."""
        return self.train_micro_batch_size_per_gpu
    
    def _synchronize_gradients(self):
        """Synchronize gradients across processes for DDP-style training."""
        
        if self.zero_stage > 0:
            # ZeRO handles gradient synchronization automatically
            return
        
        # Manual gradient synchronization
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
    
    def _prepare_gradients(self) -> bool:
        """Prepare gradients for optimizer step."""
        
        # Check for gradient overflow in mixed precision training
        if self.grad_scaler is not None:
            # Check if gradients are finite
            optimizer_state = self.grad_scaler._per_optimizer_states[id(self.optimizer)]
            if optimizer_state.get("stage") == "READY":
                found_inf = optimizer_state["found_inf_per_device"]
                if found_inf.item():
                    self.overflow_count += 1
                    return False
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            if self.grad_scaler is not None:
                # Unscale gradients before clipping
                self.grad_scaler.unscale_(self.optimizer)
            
            # Clip gradients
            if hasattr(self.optimizer, 'clip_grad_norm_'):
                # ZeRO optimizer has its own clipping method
                grad_norm = self.optimizer.clip_grad_norm_(self.max_grad_norm)
            else:
                # Standard gradient clipping
                parameters = [p for p in self.model.parameters() if p.grad is not None]
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            
            # Check for NaN gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                log_dist(f"Found NaN/Inf gradient norm: {grad_norm}", "WARNING")
                return False
        
        return True
    
    def _update_monitoring(self):
        """Update monitoring statistics."""
        
        self.step_time = self.forward_time + self.backward_time + self.optim_time
        
        # Memory monitoring
        if torch.cuda.is_available():
            memory_stats = {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'cached': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
            self.memory_monitor.update(memory_stats)
        
        # Log progress periodically
        if self.global_step % getattr(self.config, 'logging_steps', 100) == 0:
            self._log_progress()
    
    def _log_progress(self):
        """Log training progress."""
        
        current_lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.config.optimizer.get('params', {}).get('lr', 0)
        
        tokens_per_sec = self.throughput_monitor.get_throughput()
        memory_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        log_msg = (
            f"Step {self.global_step} | "
            f"LR: {current_lr:.2e} | "
            f"Tokens/s: {tokens_per_sec:.0f} | "
            f"GPU Memory: {memory_gb:.1f}GB | "
            f"Step Time: {self.step_time:.3f}s"
        )
        
        if self.overflow_count > 0:
            log_msg += f" | Overflows: {self.overflow_count}"
        
        print_rank_0(log_msg)
    
    def _start_profiling(self):
        """Start profiling at specified step."""
        
        if torch.cuda.is_available():
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()
            log_dist("Started profiling")
    
    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
        self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        self.training = True
        self.model.train()
        return self
    
    def save_checkpoint(self, checkpoint_path: str, tag: Optional[str] = None):
        """Save training checkpoint."""
        
        if self.rank != 0:
            return
        
        # Create checkpoint directory
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'global_step': self.global_step,
            'global_samples': self.global_samples,
            'model_state_dict': self._get_model_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'config': self.config,
            'loss_scale': self.loss_scale,
            'overflow_count': self.overflow_count,
            'skipped_steps': self.skipped_steps
        }
        
        # Add gradient scaler state for FP16
        if self.grad_scaler is not None:
            checkpoint['grad_scaler_state_dict'] = self.grad_scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        log_dist(f"Saved checkpoint to {checkpoint_path}")
        
        # Save tag file if provided
        if tag:
            tag_path = os.path.join(os.path.dirname(checkpoint_path), f"latest_{tag}")
            with open(tag_path, 'w') as f:
                f.write(os.path.basename(checkpoint_path))
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True, load_lr_scheduler: bool = True):
        """Load training checkpoint."""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_state_dict = checkpoint.get('model_state_dict', {})
        if model_state_dict:
            missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                log_dist(f"Missing keys in model state dict: {missing_keys}", "WARNING")
            if unexpected_keys:
                log_dist(f"Unexpected keys in model state dict: {unexpected_keys}", "WARNING")
        
        # Load optimizer state
        if load_optimizer and self.optimizer is not None:
            optimizer_state_dict = checkpoint.get('optimizer_state_dict')
            if optimizer_state_dict:
                self.optimizer.load_state_dict(optimizer_state_dict)
        
        # Load learning rate scheduler state
        if load_lr_scheduler and self.lr_scheduler is not None:
            lr_scheduler_state_dict = checkpoint.get('lr_scheduler_state_dict')
            if lr_scheduler_state_dict:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
        
        # Load gradient scaler state
        if self.grad_scaler is not None:
            grad_scaler_state_dict = checkpoint.get('grad_scaler_state_dict')
            if grad_scaler_state_dict:
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)
        
        # Restore training state
        self.global_step = checkpoint.get('global_step', 0)
        self.global_samples = checkpoint.get('global_samples', 0)
        self.loss_scale = checkpoint.get('loss_scale', 1.0)
        self.overflow_count = checkpoint.get('overflow_count', 0)
        self.skipped_steps = checkpoint.get('skipped_steps', 0)
        
        log_dist(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")
    
    def _get_model_state_dict(self) -> Dict[str, Any]:
        """Get model state dict, handling different wrapper types."""
        
        if hasattr(self.model, 'module'):
            # Handle DDP or other wrappers
            return self.model.module.state_dict()
        elif hasattr(self.model, '_orig_mod'):
            # Handle torch.compile wrapper
            return self.model._orig_mod.state_dict()
        else:
            return self.model.state_dict()
    
    def get_model(self) -> nn.Module:
        """Get the underlying model, unwrapping any distributed wrappers."""
        
        if hasattr(self.model, 'module'):
            return self.model.module
        elif hasattr(self.model, '_orig_mod'):
            return self.model._orig_mod
        else:
            return self.model
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        elif self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        else:
            return 0.0
    
    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups."""
        
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def get_global_step(self) -> int:
        """Get current global step."""
        return self.global_step
    
    def get_global_samples(self) -> int:
        """Get total number of samples processed."""
        return self.global_samples
    
    def get_loss_scale(self) -> float:
        """Get current loss scale for mixed precision training."""
        
        if self.grad_scaler is not None:
            return self.grad_scaler.get_scale()
        return self.loss_scale
    
    @contextmanager
    def no_sync(self):
        """Context manager to disable gradient synchronization."""
        
        if hasattr(self.model, 'no_sync'):
            with self.model.no_sync():
                yield
        else:
            yield
    
    @contextmanager  
    def profiling(self, enabled: bool = True):
        """Context manager for profiling specific sections."""
        
        if enabled and torch.cuda.is_available():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True
            ) as prof:
                yield prof
        else:
            yield None
    
    def memory_profile(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        
        stats = {}
        
        if torch.cuda.is_available():
            stats.update({
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'cached_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            })
            
            # Get per-device stats
            for i in range(torch.cuda.device_count()):
                device_stats = {
                    f'device_{i}_allocated_gb': torch.cuda.memory_allocated(i) / 1e9,
                    f'device_{i}_cached_gb': torch.cuda.memory_reserved(i) / 1e9
                }
                stats.update(device_stats)
        
        return stats
    
    def throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        return self.throughput_monitor.get_stats()
    
    def communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.comm_manager.get_stats()
    
    def reset_stats(self):
        """Reset all monitoring statistics."""
        
        self.throughput_monitor.reset()
        self.memory_monitor.reset()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def validate_batch_size(self):
        """Validate batch size configuration."""
        
        effective_batch_size = (
            self.train_micro_batch_size_per_gpu * 
            self.world_size * 
            self.gradient_accumulation_steps
        )
        
        if effective_batch_size != self.train_batch_size:
            log_dist(
                f"Batch size mismatch: configured={self.train_batch_size}, "
                f"effective={effective_batch_size} "
                f"(micro_batch={self.train_micro_batch_size_per_gpu}, "
                f"world_size={self.world_size}, "
                f"grad_accum={self.gradient_accumulation_steps})",
                "WARNING"
            )
    
    def estimate_memory_usage(self, sequence_length: int = 2048) -> Dict[str, float]:
        """Estimate memory usage for given sequence length."""
        
        # Get model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        
        # Parameter memory (in GB)
        param_memory = num_params * 4 / 1e9  # 4 bytes per FP32 parameter
        
        if self.fp16_enabled or self.bf16_enabled:
            param_memory /= 2  # Half precision
        
        # Gradient memory (same as parameters)
        grad_memory = param_memory
        
        # Optimizer state memory (depends on optimizer)
        optimizer_multiplier = 2  # Adam/AdamW: momentum + variance
        optimizer_memory = param_memory * optimizer_multiplier
        
        # Activation memory (rough estimate)
        batch_size = self.train_micro_batch_size_per_gpu
        hidden_size = getattr(self.config, 'hidden_size', 1024)
        num_layers = getattr(self.config, 'num_layers', 24)
        
        activation_memory = (
            batch_size * sequence_length * hidden_size * num_layers * 4 / 1e9
        )
        
        # Apply ZeRO savings
        if self.zero_stage == 1:
            optimizer_memory /= self.world_size
        elif self.zero_stage == 2:
            grad_memory /= self.world_size
            optimizer_memory /= self.world_size
        elif self.zero_stage == 3:
            param_memory /= self.world_size
            grad_memory /= self.world_size
            optimizer_memory /= self.world_size
        
        return {
            'parameters_gb': param_memory,
            'gradients_gb': grad_memory,
            'optimizer_states_gb': optimizer_memory,
            'activations_gb': activation_memory,
            'total_gb': param_memory + grad_memory + optimizer_memory + activation_memory
        }
    
    def __call__(self, *args, **kwargs):
        """Make engine callable for forward pass."""
        return self.forward(*args, **kwargs)
    
    def __repr__(self) -> str:
        """String representation of the engine."""
        
        return (
            f"DistributedSpeedEngine(\n"
            f"  world_size={self.world_size},\n"
            f"  zero_stage={self.zero_stage},\n"
            f"  pipeline_stages={getattr(self.config, 'pipeline_parallelism_size', 1)},\n"
            f"  fp16_enabled={self.fp16_enabled},\n"
            f"  bf16_enabled={self.bf16_enabled},\n"
            f"  train_batch_size={self.train_batch_size},\n"
            f"  micro_batch_size={self.train_micro_batch_size_per_gpu},\n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps}\n"
            f")"
        )
    
    def destroy(self):
        """Clean up resources and finalize training."""
        
        # Stop profiler if running
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Destroy communication resources
        if self.comm_manager:
            self.comm_manager.destroy()
        
        log_dist("DistributedSpeed engine destroyed")


class EngineBuilder:
    """
    Builder pattern for creating DistributedSpeed engines with validation.
    
    This class provides a fluent interface for configuring and building
    DistributedSpeed engines with comprehensive validation and error checking.
    """
    
    def __init__(self):
        self.model = None
        self.config = None
        self.optimizer = None
        self.lr_scheduler = None
        self.training_data = None
        self.validation_data = None
        self.comm_manager = None
        self.memory_manager = None
        self.dist_init_required = True
    
    def with_model(self, model: nn.Module):
        """Set the model to train."""
        self.model = model
        return self
    
    def with_config(self, config: Union[Dict, str, DistributedSpeedConfig]):
        """Set the training configuration."""
        self.config = config
        return self
    
    def with_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set a custom optimizer."""
        self.optimizer = optimizer
        return self
    
    def with_lr_scheduler(self, lr_scheduler: torch.optim.lr_scheduler._LRScheduler):
        """Set a custom learning rate scheduler."""
        self.lr_scheduler = lr_scheduler
        return self
    
    def with_training_data(self, training_data):
        """Set training dataset or dataloader."""
        self.training_data = training_data
        return self
    
    def with_validation_data(self, validation_data):
        """Set validation dataset or dataloader."""
        self.validation_data = validation_data
        return self
    
    def with_comm_manager(self, comm_manager: CommManager):
        """Set custom communication manager."""
        self.comm_manager = comm_manager
        return self
    
    def with_memory_manager(self, memory_manager: MemoryManager):
        """Set custom memory manager."""
        self.memory_manager = memory_manager
        return self
    
    def skip_dist_init(self):
        """Skip distributed initialization (for single GPU training)."""
        self.dist_init_required = False
        return self
    
    def build(self) -> Tuple[DistributedSpeedEngine, torch.optim.Optimizer, Any, Any]:
        """Build and return the configured engine."""
        
        if self.model is None:
            raise ValueError("Model must be provided")
        
        if self.config is None:
            raise ValueError("Config must be provided")
        
        # Import here to avoid circular imports
        from . import initialize
        
        return initialize(
            model=self.model,
            config=self.config,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            training_data=self.training_data,
            mpu=None,
            collate_fn=None,
            dist_init_required=self.dist_init_required
        )