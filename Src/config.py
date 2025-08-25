"""
Configuration management for DistributedSpeed.

This module provides comprehensive configuration classes for all aspects of
distributed training including ZeRO optimization, pipeline parallelism,
memory management, and training parameters.
"""

import os
import json
import yaml
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import warnings
from enum import Enum

class ZeROStage(Enum):
    """ZeRO optimization stages."""
    DISABLED = 0
    OPTIMIZER_STATE_PARTITIONING = 1
    GRADIENT_PARTITIONING = 2 
    PARAMETER_PARTITIONING = 3

class ActivationCheckpointing(Enum):
    """Activation checkpointing strategies."""
    DISABLED = "disabled"
    UNIFORM = "uniform"
    BLOCK = "block"
    CUSTOM = "custom"

class SchedulerType(Enum):
    """Pipeline scheduling strategies."""
    GPIPE = "gpipe"
    PIPEDREAM_1F1B = "1f1b"
    PIPEDREAM_FLUSH = "pipedream_flush"
    INTERLEAVED_1F1B = "interleaved_1f1b"

@dataclass
class ZeROConfig:
    """
    ZeRO optimization configuration.
    
    ZeRO (Zero Redundancy Optimizer) reduces memory usage by partitioning
    optimizer states, gradients, and parameters across data parallel processes.
    """
    
    # Core ZeRO settings
    enabled: bool = True
    stage: int = 2
    
    # Communication optimization
    reduce_scatter: bool = True
    allgather_partitions: bool = True
    allgather_bucket_size: int = int(2e8)
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    sub_group_size: int = int(1e9)
    
    # CPU offloading (Stage 2 and 3)
    cpu_offload: bool = False
    cpu_offload_params: bool = False
    cpu_offload_use_pin_memory: bool = True
    
    # Advanced memory optimization
    prefetch_bucket_size: int = int(5e7)
    param_persistence_threshold: int = int(1e6)
    max_reuse_distance: int = 1000
    max_live_parameters: int = int(1e9)
    
    # Stage 3 specific settings
    gather_16bit_weights_on_model_save: bool = True
    stage3_prefetch_bucket_size: int = int(5e7)
    stage3_param_persistence_threshold: int = int(1e6)
    stage3_max_reuse_distance: int = 1000
    stage3_max_live_parameters: int = int(1e9)
    stage3_gather_fp16_weights_on_model_save: bool = False
    
    # Memory monitoring
    memory_efficient_linear: bool = True
    round_robin_gradients: bool = False
    
    def __post_init__(self):
        """Validate ZeRO configuration."""
        if self.stage not in [0, 1, 2, 3]:
            raise ValueError(f"ZeRO stage must be 0, 1, 2, or 3, got {self.stage}")
        
        if self.stage == 0:
            self.enabled = False
        
        if self.cpu_offload and self.stage < 2:
            warnings.warn(
                "CPU offload is only supported for ZeRO stage 2 and 3. "
                "Disabling CPU offload.",
                UserWarning
            )
            self.cpu_offload = False

@dataclass
class ActivationCheckpointingConfig:
    """Activation checkpointing configuration for memory optimization."""
    
    enabled: bool = False
    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = False
    number_checkpoints: Optional[int] = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False
    
    # Selective checkpointing
    use_reentrant: bool = True
    checkpoint_method: str = ActivationCheckpointing.UNIFORM.value
    
    def __post_init__(self):
        """Validate activation checkpointing configuration."""
        valid_methods = [e.value for e in ActivationCheckpointing]
        if self.checkpoint_method not in valid_methods:
            raise ValueError(
                f"checkpoint_method must be one of {valid_methods}, "
                f"got {self.checkpoint_method}"
            )

@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    
    type: str = "AdamW"
    
    # Learning rate
    lr: float = 1e-4
    
    # Adam/AdamW parameters
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False
    
    # SGD parameters
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    
    # Advanced optimization
    bias_correction: bool = True
    adam_w_mode: bool = True
    
    # Custom parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate optimizer configuration."""
        supported_optimizers = ["AdamW", "Adam", "SGD", "RMSprop", "Adagrad"]
        if self.type not in supported_optimizers:
            warnings.warn(
                f"Optimizer {self.type} may not be fully supported. "
                f"Supported optimizers: {supported_optimizers}",
                UserWarning
            )

@dataclass 
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    
    type: str = "WarmupLR"
    
    # Warmup parameters
    warmup_min_lr: float = 0.0
    warmup_max_lr: float = 1e-4
    warmup_num_steps: int = 1000
    warmup_type: str = "linear"
    
    # Cosine annealing
    total_num_steps: int = 100000
    cycle_min_lr: float = 0.0
    cycle_max_lr: float = 1e-4
    decay_lr_rate: float = -1
    cycle_first_step_size: int = 1000
    cycle_first_stair_count: int = 500
    cycle_second_step_size: int = 1000
    cycle_second_stair_count: int = 500
    
    # Step LR
    step_size: int = 1000
    gamma: float = 0.1
    
    # Exponential LR
    exponential_gamma: float = 0.95
    
    # Custom parameters
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfig:
    """Pipeline parallelism configuration."""
    
    enabled: bool = False
    stages: int = 1
    partition: str = "uniform"
    seed_layers: bool = False
    activation_checkpoint_interval: int = 0
    
    # Scheduling
    schedule: str = SchedulerType.PIPEDREAM_1F1B.value
    micro_batch_size: Optional[int] = None
    
    # Memory optimization
    memory_efficient: bool = True
    
    # Load balancing
    load_balance: bool = True
    
    def __post_init__(self):
        """Validate pipeline configuration."""
        if self.enabled and self.stages < 2:
            raise ValueError("Pipeline parallelism requires at least 2 stages")
        
        valid_schedules = [e.value for e in SchedulerType]
        if self.schedule not in valid_schedules:
            raise ValueError(
                f"Pipeline schedule must be one of {valid_schedules}, "
                f"got {self.schedule}"
            )

@dataclass
class TrainingConfig:
    """General training configuration."""
    
    # Batch sizes
    train_batch_size: int = 32
    train_micro_batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 1
    
    # Training behavior
    steps_per_print: int = 100
    wall_clock_breakdown: bool = False
    dump_state: bool = False
    
    # Precision
    fp16: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    })
    
    bf16: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "auto_cast": False
    })
    
    # Gradient clipping
    gradient_clipping: float = 1.0
    
    def __post_init__(self):
        """Validate and compute derived training parameters."""
        if self.gradient_accumulation_steps <= 0:
            self.gradient_accumulation_steps = max(
                1, self.train_batch_size // self.train_micro_batch_size_per_gpu
            )
        
        # Ensure batch size consistency
        effective_batch_size = (
            self.train_micro_batch_size_per_gpu * 
            self.gradient_accumulation_steps
        )
        
        if self.train_batch_size != effective_batch_size:
            warnings.warn(
                f"train_batch_size ({self.train_batch_size}) != "
                f"train_micro_batch_size_per_gpu ({self.train_micro_batch_size_per_gpu}) "
                f"* gradient_accumulation_steps ({self.gradient_accumulation_steps}) "
                f"= {effective_batch_size}. Using effective batch size.",
                UserWarning
            )
            self.train_batch_size = effective_batch_size

@dataclass 
class CompressionConfig:
    """Gradient compression configuration."""
    
    enabled: bool = False
    compression_type: str = "none"  # "none", "fp16", "1bit", "topk"
    
    # 1-bit compression
    onebit_start_step: int = 1000
    onebit_threshold: float = 0.01
    
    # Top-K compression  
    topk_ratio: float = 0.01
    topk_min_threshold: float = 1e-6
    
    # Error feedback
    error_feedback: bool = True
    
    def __post_init__(self):
        """Validate compression configuration."""
        valid_types = ["none", "fp16", "1bit", "topk"]
        if self.compression_type not in valid_types:
            raise ValueError(
                f"compression_type must be one of {valid_types}, "
                f"got {self.compression_type}"
            )

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    
    enabled: bool = True
    
    # Logging
    tensorboard: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "output_path": "./logs/tensorboard",
        "job_name": "distributedspeed_experiment"
    })
    
    wandb: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "project": "distributedspeed",
        "team": None,
        "group": None,
        "tags": []
    })
    
    # Profiling
    profile: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "trace_ready": 1000,
        "profile_steps": [100, 200, 300],
        "output_path": "./logs/profiling"
    })
    
    # Memory monitoring
    memory_breakdown: bool = False
    memory_snapshot: bool = False

@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    
    # Saving
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    keep_last_n_checkpoints: int = 5
    
    # Loading
    load_path: Optional[str] = None
    auto_resume: bool = True
    
    # Checkpoint content
    save_zero_checkpoint: bool = True
    use_node_local_storage: bool = False
    
    # Async checkpointing
    async_save: bool = True

@dataclass
class DistributedConfig:
    """
    Main configuration class for DistributedSpeed.
    
    This class aggregates all configuration options for distributed training,
    providing a single interface for configuration management.
    """
    
    # Core configurations
    zero_optimization: ZeROConfig = field(default_factory=ZeROConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig) 
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Advanced features
    activation_checkpointing: ActivationCheckpointingConfig = field(
        default_factory=ActivationCheckpointingConfig
    )
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Communication
    communication_data_type: str = "fp16"
    prescale_gradients: bool = False
    gradient_predivide_factor: float = 1.0
    
    # Performance
    sparse_gradients: bool = False
    cpu_optimizer: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DistributedConfig':
        """Create configuration from dictionary."""
        # Deep copy to avoid modifying original
        config_dict = copy.deepcopy(config_dict)
        
        # Extract nested configurations
        zero_config = config_dict.pop('zero_optimization', {})
        optimizer_config = config_dict.pop('optimizer', {})
        scheduler_config = config_dict.pop('scheduler', {})  
        training_config = config_dict.pop('training', {})
        pipeline_config = config_dict.pop('pipeline', {})
        activation_config = config_dict.pop('activation_checkpointing', {})
        compression_config = config_dict.pop('compression', {})
        monitoring_config = config_dict.pop('monitoring', {})
        checkpoint_config = config_dict.pop('checkpoint', {})
        
        return cls(
            zero_optimization=ZeROConfig(**zero_config),
            optimizer=OptimizerConfig(**optimizer_config),
            scheduler=SchedulerConfig(**scheduler_config),
            training=TrainingConfig(**training_config), 
            pipeline=PipelineConfig(**pipeline_config),
            activation_checkpointing=ActivationCheckpointingConfig(**activation_config),
            compression=CompressionConfig(**compression_config),
            monitoring=MonitoringConfig(**monitoring_config),
            checkpoint=CheckpointConfig(**checkpoint_config),
            **config_dict
        )
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DistributedConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'DistributedConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save_json(self, json_path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def save_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to YAML file.""" 
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def validate(self):
        """Validate the complete configuration."""
        # ZeRO and Pipeline compatibility
        if self.zero_optimization.stage == 3 and self.pipeline.enabled:
            warnings.warn(
                "ZeRO Stage 3 with Pipeline Parallelism may have compatibility issues. "
                "Consider using ZeRO Stage 2 with Pipeline Parallelism.",
                UserWarning
            )
        
        # Mixed precision validation
        if self.training.fp16["enabled"] and self.training.bf16["enabled"]:
            raise ValueError("Cannot enable both FP16 and BF16 simultaneously")
        
        # Batch size validation
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        total_batch_size = self.training.train_batch_size * world_size
        
        if total_batch_size > 10000:
            warnings.warn(
                f"Very large total batch size: {total_batch_size}. "
                "This may affect training stability.",
                UserWarning
            )
        
        # Memory optimization recommendations
        if (self.zero_optimization.stage >= 2 and 
            not self.activation_checkpointing.enabled):
            warnings.warn(
                "Consider enabling activation checkpointing with ZeRO Stage 2+ "
                "for additional memory savings.",
                UserWarning
            )
    
    def get_effective_batch_size(self, world_size: int = 1) -> int:
        """Calculate the effective global batch size."""
        return (
            self.training.train_micro_batch_size_per_gpu * 
            self.training.gradient_accumulation_steps * 
            world_size
        )
    
    def get_memory_estimates(self) -> Dict[str, float]:
        """Estimate memory usage based on configuration."""
        estimates = {
            "model_memory_gb": 0.0,
            "optimizer_memory_gb": 0.0, 
            "gradient_memory_gb": 0.0,
            "activation_memory_gb": 0.0,
            "total_memory_gb": 0.0
        }
        
        # This would be populated with actual estimates based on model size
        # For now, return placeholder estimates
        return estimates
    
    def optimize_for_hardware(self, gpu_memory_gb: float, num_gpus: int):
        """Automatically optimize configuration for available hardware."""
        # Adjust batch sizes based on available memory
        if gpu_memory_gb <= 16:
            # Small GPU optimization
            self.training.train_micro_batch_size_per_gpu = min(
                self.training.train_micro_batch_size_per_gpu, 2
            )
            self.zero_optimization.stage = max(self.zero_optimization.stage, 2)
            self.activation_checkpointing.enabled = True
            
        elif gpu_memory_gb <= 40:
            # Medium GPU optimization  
            self.training.train_micro_batch_size_per_gpu = min(
                self.training.train_micro_batch_size_per_gpu, 4
            )
            if not self.zero_optimization.enabled:
                self.zero_optimization.stage = 1
                
        # Enable pipeline parallelism for multi-GPU setups
        if num_gpus >= 4 and not self.pipeline.enabled:
            self.pipeline.enabled = True
            self.pipeline.stages = min(num_gpus // 2, 8)

# Predefined configuration presets
def get_debug_config() -> DistributedConfig:
    """Get configuration optimized for debugging and fast iteration."""
    return DistributedConfig(
        zero_optimization=ZeROConfig(
            enabled=True,
            stage=1,
            overlap_comm=False
        ),
        training=TrainingConfig(
            train_batch_size=8,
            train_micro_batch_size_per_gpu=2,
            gradient_accumulation_steps=1,
            steps_per_print=10,
            wall_clock_breakdown=True
        ),
        monitoring=MonitoringConfig(
            enabled=True,
            memory_breakdown=True,
            profile={"enabled": True, "profile_steps": [10, 20, 30]}
        )
    )

def get_small_config() -> DistributedConfig:
    """Get configuration for small models (< 1B parameters)."""
    return DistributedConfig(
        zero_optimization=ZeROConfig(
            enabled=True,
            stage=1,
            overlap_comm=True
        ),
        training=TrainingConfig(
            train_batch_size=32,
            train_micro_batch_size_per_gpu=4,
            fp16={"enabled": True}
        ),
        optimizer=OptimizerConfig(
            type="AdamW",
            lr=1e-4,
            weight_decay=0.01
        )
    )

def get_medium_config() -> DistributedConfig:
    """Get configuration for medium models (1B-10B parameters)."""
    return DistributedConfig(
        zero_optimization=ZeROConfig(
            enabled=True,
            stage=2,
            overlap_comm=True,
            allgather_bucket_size=int(2e8)
        ),
        training=TrainingConfig(
            train_batch_size=64,
            train_micro_batch_size_per_gpu=2,
            gradient_accumulation_steps=4,
            fp16={"enabled": True}
        ),
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=True,
            partition_activations=True
        ),
        optimizer=OptimizerConfig(
            type="AdamW", 
            lr=5e-5,
            weight_decay=0.01
        )
    )

def get_large_config() -> DistributedConfig:
    """Get configuration for large models (10B+ parameters).""" 
    return DistributedConfig(
        zero_optimization=ZeROConfig(
            enabled=True,
            stage=3,
            overlap_comm=True,
            cpu_offload=True,
            stage3_prefetch_bucket_size=int(5e7)
        ),
        training=TrainingConfig(
            train_batch_size=128,
            train_micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=16,
            bf16={"enabled": True}
        ),
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=True,
            partition_activations=True,
            cpu_checkpointing=True
        ),
        pipeline=PipelineConfig(
            enabled=True,
            stages=4,
            schedule="1f1b"
        ),
        optimizer=OptimizerConfig(
            type="AdamW",
            lr=1e-5, 
            weight_decay=0.01
        )
    )

def get_inference_config() -> DistributedConfig:
    """Get configuration optimized for inference."""
    return DistributedConfig(
        zero_optimization=ZeROConfig(
            enabled=True,
            stage=3,
            overlap_comm=False,
            cpu_offload=True
        ),
        training=TrainingConfig(
            train_batch_size=1,
            train_micro_batch_size_per_gpu=1,
            bf16={"enabled": True}
        ),
        activation_checkpointing=ActivationCheckpointingConfig(
            enabled=False
        )
    )

# Configuration registry
CONFIG_PRESETS = {
    "debug": get_debug_config,
    "small": get_small_config, 
    "medium": get_medium_config,
    "large": get_large_config,
    "inference": get_inference_config
}

def get_config(preset_name: str) -> DistributedConfig:
    """Get a predefined configuration preset."""
    if preset_name not in CONFIG_PRESETS:
        available = list(CONFIG_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available presets: {available}"
        )
    
    return CONFIG_PRESETS[preset_name]()

def list_presets() -> List[str]:
    """List all available configuration presets.""" 
    return list(CONFIG_PRESETS.keys())

# Environment-based configuration
def get_config_from_env() -> DistributedConfig:
    """Create configuration from environment variables."""
    config = DistributedConfig()
    
    # Training configuration from environment
    if "DS_TRAIN_BATCH_SIZE" in os.environ:
        config.training.train_batch_size = int(os.environ["DS_TRAIN_BATCH_SIZE"])
    
    if "DS_MICRO_BATCH_SIZE" in os.environ:
        config.training.train_micro_batch_size_per_gpu = int(
            os.environ["DS_MICRO_BATCH_SIZE"]
        )
    
    if "DS_GRADIENT_ACCUMULATION" in os.environ:
        config.training.gradient_accumulation_steps = int(
            os.environ["DS_GRADIENT_ACCUMULATION"]
        )
    
    # ZeRO configuration from environment
    if "DS_ZERO_STAGE" in os.environ:
        config.zero_optimization.stage = int(os.environ["DS_ZERO_STAGE"])
        config.zero_optimization.enabled = config.zero_optimization.stage > 0
    
    if "DS_CPU_OFFLOAD" in os.environ:
        config.zero_optimization.cpu_offload = (
            os.environ["DS_CPU_OFFLOAD"].lower() == "true"
        )
    
    # Precision from environment
    if "DS_FP16" in os.environ:
        config.training.fp16["enabled"] = (
            os.environ["DS_FP16"].lower() == "true"
        )
    
    if "DS_BF16" in os.environ:
        config.training.bf16["enabled"] = (
            os.environ["DS_BF16"].lower() == "true"
        )
    
    # Pipeline configuration from environment
    if "DS_PIPELINE_STAGES" in os.environ:
        stages = int(os.environ["DS_PIPELINE_STAGES"])
        config.pipeline.enabled = stages > 1
        config.pipeline.stages = stages
    
    return config