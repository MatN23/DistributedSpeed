"""
DistributedSpeed Configuration Management.

This module provides comprehensive configuration management for DistributedSpeed,
including validation, schema definitions, presets, and utilities for loading
and managing training configurations.

Key Features:
- Schema validation with detailed error messages
- Configuration presets for common use cases
- Environment variable interpolation
- JSON/YAML configuration file support
- Runtime configuration updates
- Configuration inheritance and merging

Copyright (c) 2024 DistributedSpeed Contributors
Licensed under the Apache License, Version 2.0
"""

import os
import json
import yaml
import copy
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "AdamW"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
    })


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "WarmupCosineScheduler"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "warmup_steps": 1000,
        "total_steps": 100000,
        "min_lr": 1e-6
    })


@dataclass
class ZeROConfig:
    """ZeRO optimization configuration."""
    stage: int = 0
    cpu_offload: bool = False
    cpu_offload_params: bool = False
    cpu_offload_use_pin_memory: bool = True
    
    # Stage 1/2/3 common parameters
    allgather_partitions: bool = True
    allgather_bucket_size: float = 2e8
    overlap_comm: bool = True
    reduce_scatter: bool = True
    reduce_bucket_size: float = 2e8
    contiguous_gradients: bool = True
    
    # Stage 3 specific parameters
    param_persistence_threshold: float = 1e6
    model_persistence_threshold: float = 1e6
    max_live_parameters: float = 1e9
    max_reuse_distance: float = 1e9
    prefetch_bucket_size: float = 5e8
    param_round_robin: bool = False
    offload_optimizer_config: Optional[Dict[str, Any]] = None
    offload_param_config: Optional[Dict[str, Any]] = None
    
    # Advanced parameters
    sub_group_size: float = 1e9
    elastic_checkpoint: bool = True
    ignore_unused_parameters: bool = True
    partition_grads: bool = True
    round_robin_gradients: bool = False
    zero_hpz_partition_size: int = 1
    zero_quantized_weights: bool = False
    zero_quantized_gradients: bool = False


@dataclass 
class FP16Config:
    """FP16 mixed precision configuration."""
    enabled: bool = False
    loss_scale: float = 0.0  # 0 means dynamic scaling
    initial_scale_power: int = 16
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: float = 1.0
    auto_cast: bool = True


@dataclass
class BF16Config:
    """BF16 mixed precision configuration.""" 
    enabled: bool = False
    auto_cast: bool = True


@dataclass
class ActivationCheckpointingConfig:
    """Activation checkpointing configuration."""
    enabled: bool = False
    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = False
    number_checkpoints: Optional[int] = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False
    partition_method: str = "uniform"  # uniform, type_based, block_based
    checkpoint_wrapper: str = "torch"  # torch, fairscale


@dataclass
class PipelineConfig:
    """Pipeline parallelism configuration."""
    stages: int = 1
    partition: str = "uniform"  # uniform, parameters, type
    seed_layers: bool = False
    activation_checkpoint_interval: int = 1
    pipe_partitioned: bool = False
    pipe_replicated: bool = False
    loss_fn: Optional[str] = None


@dataclass
class CompressionConfig:
    """Gradient compression configuration."""
    enabled: bool = False
    compression_type: str = "fp16"  # fp16, int8, int4
    quantization_type: str = "asymmetric_quantization_signed"
    quantization_bits: int = 8
    error_feedback: bool = True
    all_gather_fp16: bool = True
    schedule_quantization: bool = False
    quantization_period: int = 1000
    compression_technique: str = "residual"  # residual, natural


@dataclass 
class CommunicationConfig:
    """Communication optimization configuration."""
    overlap_comm: bool = True
    bucket_cap_mb: float = 25.0
    overlap_reduce_scatter: bool = True
    overlap_all_gather: bool = True
    hierarchical_allreduce: bool = False
    sparse_gradients: bool = False
    compression: CompressionConfig = field(default_factory=CompressionConfig)


@dataclass
class ProfilingConfig:
    """Profiling and monitoring configuration."""
    enabled: bool = False
    profile_step: int = 5
    module_depth: int = -1
    top_modules: int = 1
    output_path: str = "./profiling_results"
    detailed_profiling: bool = False
    memory_profiling: bool = True
    communication_profiling: bool = True
    tensorboard_trace_handler: bool = True


@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    keep_last_n_checkpoints: int = 5
    save_optimizer_state: bool = True
    save_lr_scheduler_state: bool = True
    async_save: bool = False
    use_compression: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    console_logging: bool = True
    distributed_logging: bool = True
    tensorboard_logging: bool = False
    wandb_logging: bool = False
    logging_steps: int = 100
    eval_logging_steps: int = 500


class DistributedSpeedConfig:
    """
    Main DistributedSpeed configuration class.
    
    This class manages all configuration aspects of DistributedSpeed training,
    including validation, defaults, and runtime updates.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary or None for defaults
        """
        
        # Set defaults
        self._set_defaults()
        
        # Load from dictionary if provided
        if config_dict:
            self._load_from_dict(config_dict)
        
        # Validate configuration
        self._validate()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _set_defaults(self):
        """Set default configuration values."""
        
        # Core training parameters
        self.train_batch_size = 32
        self.train_micro_batch_size_per_gpu = None  # Auto-calculated
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0
        self.gradient_checkpointing = False
        
        # Model parameters  
        self.hidden_size = 1024
        self.num_layers = 24
        self.num_attention_heads = 16
        self.seq_length = 2048
        self.vocab_size = 50304
        
        # Optimization
        self.optimizer = OptimizerConfig()
        self.scheduler = SchedulerConfig()
        self.zero_optimization = ZeROConfig()
        
        # Mixed precision
        self.fp16 = FP16Config()
        self.bf16 = BF16Config()
        
        # Advanced features
        self.activation_checkpointing = ActivationCheckpointingConfig()
        self.pipeline_parallelism = PipelineConfig()
        self.communication = CommunicationConfig()
        
        # Infrastructure
        self.dataloader = DataLoaderConfig()
        self.checkpointing = CheckpointingConfig()
        self.logging = LoggingConfig()
        self.profiling = ProfilingConfig()
        
        # Compilation and optimization
        self.compile = False
        self.compile_kwargs = {}
        
        # Memory optimization
        self.memory_efficient_attention = False
        self.cpu_adam = False
        self.nvme_swap_dir = None
        
        # Training behavior
        self.seed = 1234
        self.data_path = []
        self.eval_interval = 1000
        self.eval_iters = 100
        self.log_interval = 100
        self.save_interval = 1000
        
        # Advanced distributed settings
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.sequence_parallel = False
        self.expert_model_parallel_size = 1
    
    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary with nested structure support."""
        
        def _deep_update(base_dict, update_dict):
            """Deep update dictionary with nested support."""
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    _deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        # Convert config objects to dicts for updating
        self_dict = {
            'optimizer': asdict(self.optimizer),
            'scheduler': asdict(self.scheduler),
            'zero_optimization': asdict(self.zero_optimization),
            'fp16': asdict(self.fp16),
            'bf16': asdict(self.bf16),
            'activation_checkpointing': asdict(self.activation_checkpointing),
            'pipeline_parallelism': asdict(self.pipeline_parallelism),
            'communication': asdict(self.communication),
            'dataloader': asdict(self.dataloader),
            'checkpointing': asdict(self.checkpointing),
            'logging': asdict(self.logging),
            'profiling': asdict(self.profiling)
        }
        
        # Deep update with provided config
        _deep_update(self_dict, config_dict)
        
        # Update object attributes
        for key, value in config_dict.items():
            if key in ['optimizer', 'scheduler', 'zero_optimization', 'fp16', 'bf16',
                      'activation_checkpointing', 'pipeline_parallelism', 'communication',
                      'dataloader', 'checkpointing', 'logging', 'profiling']:
                # Handle nested config objects
                continue
            else:
                setattr(self, key, value)
        
        # Reconstruct config objects
        self.optimizer = OptimizerConfig(**self_dict['optimizer'])
        self.scheduler = SchedulerConfig(**self_dict['scheduler'])
        self.zero_optimization = ZeROConfig(**self_dict['zero_optimization'])
        self.fp16 = FP16Config(**self_dict['fp16'])
        self.bf16 = BF16Config(**self_dict['bf16'])
        self.activation_checkpointing = ActivationCheckpointingConfig(**self_dict['activation_checkpointing'])
        self.pipeline_parallelism = PipelineConfig(**self_dict['pipeline_parallelism'])
        self.communication = CommunicationConfig(**self_dict['communication'])
        self.dataloader = DataLoaderConfig(**self_dict['dataloader'])
        self.checkpointing = CheckpointingConfig(**self_dict['checkpointing'])
        self.logging = LoggingConfig(**self_dict['logging'])
        self.profiling = ProfilingConfig(**self_dict['profiling'])
    
    def _validate(self):
        """Validate configuration parameters."""
        
        errors = []
        
        # Validate core parameters
        if self.train_batch_size <= 0:
            errors.append("train_batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            errors.append("gradient_accumulation_steps must be positive")
        
        # Validate ZeRO configuration
        if self.zero_optimization.stage not in [0, 1, 2, 3]:
            errors.append("ZeRO stage must be 0, 1, 2, or 3")
        
        # Validate mixed precision
        if self.fp16.enabled and self.bf16.enabled:
            errors.append("Cannot enable both FP16 and BF16 simultaneously")
        
        # Validate optimizer configuration
        if not hasattr(self.optimizer, 'type') or not self.optimizer.type:
            errors.append("Optimizer type must be specified")
        
        # Validate data paths
        if hasattr(self, 'data_path') and self.data_path:
            for path in self.data_path:
                if not os.path.exists(path):
                    logger.warning(f"Data path does not exist: {path}")
        
        # Validate pipeline configuration
        if self.pipeline_parallelism.stages > 1:
            if self.zero_optimization.stage == 3:
                logger.warning("Pipeline parallelism with ZeRO Stage 3 may have limited benefits")
        
        # Validate activation checkpointing
        if self.activation_checkpointing.enabled and self.gradient_checkpointing:
            logger.warning("Both activation checkpointing and gradient checkpointing are enabled")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        
        env_mappings = {
            'DISTRIBUTEDSPEED_TRAIN_BATCH_SIZE': ('train_batch_size', int),
            'DISTRIBUTEDSPEED_MICRO_BATCH_SIZE': ('train_micro_batch_size_per_gpu', int),
            'DISTRIBUTEDSPEED_GRAD_ACCUM_STEPS': ('gradient_accumulation_steps', int),
            'DISTRIBUTEDSPEED_LEARNING_RATE': ('optimizer.params.lr', float),
            'DISTRIBUTEDSPEED_ZERO_STAGE': ('zero_optimization.stage', int),
            'DISTRIBUTEDSPEED_FP16_ENABLED': ('fp16.enabled', lambda x: x.lower() == 'true'),
            'DISTRIBUTEDSPEED_BF16_ENABLED': ('bf16.enabled', lambda x: x.lower() == 'true'),
            'DISTRIBUTEDSPEED_COMPILE': ('compile', lambda x: x.lower() == 'true'),
            'DISTRIBUTEDSPEED_SEED': ('seed', int)
        }
        
        for env_var, (attr_path, converter) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    self._set_nested_attr(attr_path, converted_value)
                    logger.info(f"Applied environment override: {env_var}={converted_value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to apply environment override {env_var}={env_value}: {e}")
    
    def _set_nested_attr(self, attr_path: str, value: Any):
        """Set nested attribute using dot notation."""
        
        attrs = attr_path.split('.')
        obj = self
        
        # Navigate to the parent object
        for attr in attrs[:-1]:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                logger.warning(f"Attribute path {attr_path} not found")
                return
        
        # Set the final attribute
        final_attr = attrs[-1]
        if hasattr(obj, final_attr):
            setattr(obj, final_attr, value)
        else:
            logger.warning(f"Final attribute {final_attr} not found in {attr_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        
        result = {}
        
        # Simple attributes
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                
                # Handle dataclass objects
                if hasattr(attr_value, '__dataclass_fields__'):
                    result[attr_name] = asdict(attr_value)
                else:
                    result[attr_name] = attr_value
        
        return result
    
    def save(self, filepath: str):
        """Save configuration to file."""
        
        config_dict = self.to_dict()
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        
        temp_dict = self.to_dict()
        temp_dict.update(updates)
        
        # Create new instance with updated values
        updated_config = DistributedSpeedConfig(temp_dict)
        
        # Copy all attributes
        for attr_name in dir(updated_config):
            if not attr_name.startswith('_') and not callable(getattr(updated_config, attr_name)):
                setattr(self, attr_name, getattr(updated_config, attr_name))
    
    def get_effective_batch_size(self, world_size: int = 1) -> int:
        """Calculate effective batch size."""
        
        micro_batch_size = self.train_micro_batch_size_per_gpu
        if micro_batch_size is None:
            micro_batch_size = max(1, self.train_batch_size // (world_size * self.gradient_accumulation_steps))
        
        return micro_batch_size * world_size * self.gradient_accumulation_steps
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        
        return {
            'hidden_size': getattr(self, 'hidden_size', 1024),
            'num_layers': getattr(self, 'num_layers', 24),
            'num_attention_heads': getattr(self, 'num_attention_heads', 16),
            'seq_length': getattr(self, 'seq_length', 2048),
            'vocab_size': getattr(self, 'vocab_size', 50304)
        }
    
    def clone(self) -> 'DistributedSpeedConfig':
        """Create a deep copy of the configuration."""
        return DistributedSpeedConfig(self.to_dict())
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        
        key_params = [
            f"batch_size={self.train_batch_size}",
            f"zero_stage={self.zero_optimization.stage}",
            f"fp16={self.fp16.enabled}",
            f"bf16={self.bf16.enabled}",
            f"pipeline_stages={self.pipeline_parallelism.stages}"
        ]
        
        return f"DistributedSpeedConfig({', '.join(key_params)})"


def load_config(config_path: str) -> DistributedSpeedConfig:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DistributedSpeedConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            # Try JSON first, then YAML
            content = f.read()
            try:
                config_dict = json.loads(content)
            except json.JSONDecodeError:
                try:
                    config_dict = yaml.safe_load(content)
                except yaml.YAMLError as e:
                    raise ValueError(f"Failed to parse configuration file: {e}")
    
    if not isinstance(config_dict, dict):
        raise ValueError("Configuration file must contain a dictionary")
    
    return DistributedSpeedConfig(config_dict)


def get_config_template() -> Dict[str, Any]:
    """Get a complete configuration template with all options."""
    
    return {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": False,
        
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        
        "scheduler": {
            "type": "WarmupCosineScheduler",
            "params": {
                "warmup_steps": 1000,
                "total_steps": 100000,
                "min_lr": 1e-6
            }
        },
        
        "zero_optimization": {
            "stage": 2,
            "cpu_offload": False,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1.0
        },
        
        "bf16": {
            "enabled": False
        },
        
        "activation_checkpointing": {
            "enabled": False,
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "synchronize_checkpoint_boundary": False
        },
        
        "pipeline_parallelism": {
            "stages": 1,
            "partition": "uniform",
            "seed_layers": False,
            "activation_checkpoint_interval": 1
        },
        
        "communication": {
            "overlap_comm": True,
            "bucket_cap_mb": 25.0,
            "overlap_reduce_scatter": True,
            "overlap_all_gather": True,
            "compression": {
                "enabled": False,
                "compression_type": "fp16",
                "quantization_bits": 8,
                "error_feedback": True
            }
        },
        
        "dataloader": {
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": True,
            "persistent_workers": False,
            "prefetch_factor": 2
        },
        
        "checkpointing": {
            "save_dir": "./checkpoints",
            "save_interval": 1000,
            "keep_last_n_checkpoints": 5,
            "save_optimizer_state": True,
            "save_lr_scheduler_state": True
        },
        
        "logging": {
            "log_level": "INFO",
            "console_logging": True,
            "tensorboard_logging": False,
            "wandb_logging": False,
            "logging_steps": 100
        },
        
        "profiling": {
            "enabled": False,
            "profile_step": 5,
            "output_path": "./profiling_results",
            "detailed_profiling": False
        },
        
        "compile": False,
        "seed": 1234
    }


# Configuration presets for common use cases
PRESET_CONFIGS = {
    "debug": {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
        "fp16": {"enabled": False},
        "logging": {"logging_steps": 1}
    },
    
    "small_model": {
        "train_batch_size": 128,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {"stage": 1},
        "fp16": {"enabled": True},
        "gradient_checkpointing": True
    },
    
    "medium_model": {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "zero_optimization": {"stage": 2, "cpu_offload": False},
        "fp16": {"enabled": True},
        "gradient_checkpointing": True,
        "activation_checkpointing": {"enabled": True}
    },
    
    "large_model": {
        "train_batch_size": 512,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 8,
        "zero_optimization": {
            "stage": 3,
            "cpu_offload": True,
            "param_persistence_threshold": 1e6
        },
        "bf16": {"enabled": True},
        "gradient_checkpointing": True,
        "activation_checkpointing": {
            "enabled": True,
            "cpu_checkpointing": True
        }
    },
    
    "huge_model": {
        "train_batch_size": 1024,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 16,
        "zero_optimization": {
            "stage": 3,
            "cpu_offload": True,
            "cpu_offload_params": True,
            "param_persistence_threshold": 1e5,
            "max_live_parameters": 1e9
        },
        "bf16": {"enabled": True},
        "gradient_checkpointing": True,
        "activation_checkpointing": {
            "enabled": True,
            "cpu_checkpointing": True,
            "partition_activations": True
        },
        "nvme_swap_dir": "/tmp/nvme_swap"
    }
}


def get_preset_config(preset_name: str, **overrides) -> DistributedSpeedConfig:
    """
    Get a preset configuration with optional overrides.
    
    Args:
        preset_name: Name of preset configuration
        **overrides: Configuration overrides
        
    Returns:
        DistributedSpeedConfig object
        
    Raises:
        ValueError: If preset name is not found
    """
    
    if preset_name not in PRESET_CONFIGS:
        available_presets = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available_presets}")
    
    # Start with template and apply preset
    config_dict = get_config_template()
    preset_dict = PRESET_CONFIGS[preset_name]
    
    # Deep merge preset configuration
    def deep_merge(base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(config_dict, preset_dict)
    
    # Apply overrides
    if overrides:
        deep_merge(config_dict, overrides)
    
    return DistributedSpeedConfig(config_dict)


def validate_config_compatibility(config: DistributedSpeedConfig, world_size: int) -> List[str]:
    """
    Validate configuration compatibility with given world size.
    
    Args:
        config: Configuration to validate
        world_size: Number of distributed processes
        
    Returns:
        List of warning messages
    """
    
    warnings = []
    
    # Check batch size divisibility
    effective_batch_size = config.get_effective_batch_size(world_size)
    if effective_batch_size != config.train_batch_size:
        warnings.append(
            f"Effective batch size ({effective_batch_size}) differs from configured "
            f"batch size ({config.train_batch_size})"
        )
    
    # Check pipeline parallelism
    if config.pipeline_parallelism.stages > 1:
        if world_size % config.pipeline_parallelism.stages != 0:
            warnings.append(
                f"World size ({world_size}) should be divisible by pipeline stages "
                f"({config.pipeline_parallelism.stages})"
            )
    
    # Check ZeRO configuration
    if config.zero_optimization.stage == 3 and config.train_micro_batch_size_per_gpu == 1:
        warnings.append(
            "ZeRO Stage 3 with micro batch size 1 may have reduced efficiency"
        )
    
    # Check memory optimization
    if config.fp16.enabled and not config.gradient_checkpointing:
        warnings.append(
            "Consider enabling gradient checkpointing with FP16 for better memory efficiency"
        )
    
    return warnings


def auto_configure(
    model_size: str,
    world_size: int,
    memory_per_gpu: float,
    target_batch_size: int = 256
) -> DistributedSpeedConfig:
    """
    Automatically configure DistributedSpeed based on model size and hardware.
    
    Args:
        model_size: Model size category ('small', 'medium', 'large', 'huge')
        world_size: Number of GPUs available
        memory_per_gpu: GPU memory in GB
        target_batch_size: Target global batch size
        
    Returns:
        Optimized DistributedSpeedConfig
        
    Raises:
        ValueError: If model size is not supported
    """
    
    size_mapping = {
        'small': 'small_model',
        'medium': 'medium_model', 
        'large': 'large_model',
        'huge': 'huge_model'
    }
    
    if model_size not in size_mapping:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    # Start with preset
    preset_name = size_mapping[model_size]
    config = get_preset_config(preset_name)
    
    # Adjust for target batch size
    micro_batch_size = max(1, target_batch_size // (world_size * 8))  # Assume 8 grad accum steps
    grad_accum_steps = max(1, target_batch_size // (world_size * micro_batch_size))
    
    # Memory-based adjustments
    if memory_per_gpu < 16:
        # Low memory - use ZeRO Stage 3 and reduce batch size
        config.zero_optimization.stage = 3
        config.zero_optimization.cpu_offload = True
        micro_batch_size = min(micro_batch_size, 2)
        config.gradient_checkpointing = True
        config.activation_checkpointing.enabled = True
        config.activation_checkpointing.cpu_checkpointing = True
    elif memory_per_gpu < 32:
        # Medium memory - use ZeRO Stage 2
        config.zero_optimization.stage = 2
        micro_batch_size = min(micro_batch_size, 4)
        config.gradient_checkpointing = True
    else:
        # High memory - can use larger batches
        if model_size in ['small', 'medium']:
            config.zero_optimization.stage = 1
        micro_batch_size = min(micro_batch_size, 8)
    
    # Update batch size configuration
    config.train_batch_size = target_batch_size
    config.train_micro_batch_size_per_gpu = micro_batch_size
    config.gradient_accumulation_steps = grad_accum_steps
    
    return config