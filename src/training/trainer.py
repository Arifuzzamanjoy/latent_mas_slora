"""
LoRA Trainer - Custom training pipeline for adapter fine-tuning

Features:
- Gradient accumulation for memory efficiency
- Mixed precision training
- Checkpoint management
- Training metrics and logging
"""

import os
import json
import time
import torch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TrainingDataset, DataCollator


@dataclass
class TrainingConfig:
    """Configuration for LoRA training"""
    # LoRA parameters
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Training parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Optimizer
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "linear", "constant"
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # "bf16", "fp16", "no"
    
    # Data
    max_length: int = 2048
    
    # Saving
    output_dir: str = "./lora_output"
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingResult:
    """Result from training run"""
    adapter_path: str
    final_loss: float
    training_steps: int
    total_time_seconds: float
    best_eval_loss: Optional[float] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    config: Optional[TrainingConfig] = None


class LoRATrainer:
    """
    Custom LoRA training pipeline.
    
    Optimized for:
    - Memory-efficient training on 24-48GB VRAM
    - Domain-specific adapter creation
    - Integration with existing LatentMAS agents
    
    Example:
        trainer = LoRATrainer(model, tokenizer)
        result = trainer.train(
            train_dataset,
            config=TrainingConfig(num_epochs=3, learning_rate=2e-4)
        )
        # Adapter saved to result.adapter_path
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.base_model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self._peft_model = None
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
    
    def _setup_lora(self, config: TrainingConfig) -> None:
        """Setup LoRA adapter for training"""
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        
        # Prepare for training if quantized
        if hasattr(self.base_model, 'is_quantized') and self.base_model.is_quantized:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
        )
        
        # Wrap model
        self._peft_model = get_peft_model(self.base_model, lora_config)
        
        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            self._peft_model.enable_input_require_grads()
            if hasattr(self._peft_model, 'gradient_checkpointing_enable'):
                self._peft_model.gradient_checkpointing_enable()
        
        # Print trainable params
        trainable = sum(p.numel() for p in self._peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._peft_model.parameters())
        print(f"[Training] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def _setup_optimizer(self, config: TrainingConfig, num_training_steps: int) -> None:
        """Setup optimizer and scheduler"""
        from torch.optim import AdamW, Adam, SGD
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, LinearLR, ConstantLR, SequentialLR
        )
        
        # Filter trainable parameters
        params = [p for p in self._peft_model.parameters() if p.requires_grad]
        
        # Optimizer
        if config.optimizer == "adamw":
            self._optimizer = AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.999),
            )
        elif config.optimizer == "adam":
            self._optimizer = Adam(params, lr=config.learning_rate)
        elif config.optimizer == "sgd":
            self._optimizer = SGD(params, lr=config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
        # Scheduler
        warmup_steps = int(num_training_steps * config.warmup_ratio)
        
        if config.scheduler == "cosine":
            warmup = LinearLR(
                self._optimizer,
                start_factor=0.1,
                total_iters=warmup_steps,
            )
            decay = CosineAnnealingLR(
                self._optimizer,
                T_max=num_training_steps - warmup_steps,
            )
            self._scheduler = SequentialLR(
                self._optimizer,
                schedulers=[warmup, decay],
                milestones=[warmup_steps],
            )
        elif config.scheduler == "linear":
            warmup = LinearLR(
                self._optimizer,
                start_factor=0.1,
                total_iters=warmup_steps,
            )
            decay = LinearLR(
                self._optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps - warmup_steps,
            )
            self._scheduler = SequentialLR(
                self._optimizer,
                schedulers=[warmup, decay],
                milestones=[warmup_steps],
            )
        else:
            self._scheduler = ConstantLR(self._optimizer, factor=1.0)
        
        # Mixed precision scaler
        if config.mixed_precision == "fp16":
            self._scaler = torch.cuda.amp.GradScaler()
    
    def train(
        self,
        train_dataset: TrainingDataset,
        config: Optional[TrainingConfig] = None,
        eval_dataset: Optional[TrainingDataset] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingResult:
        """
        Train LoRA adapter.
        
        Args:
            train_dataset: Training dataset
            config: Training configuration
            eval_dataset: Optional evaluation dataset
            callbacks: Optional callback functions
            
        Returns:
            TrainingResult with adapter path and metrics
        """
        config = config or TrainingConfig()
        callbacks = callbacks or []
        
        # Setup
        os.makedirs(config.output_dir, exist_ok=True)
        self._setup_lora(config)
        
        # DataLoader
        collator = DataCollator(self.tokenizer, max_length=config.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collator,
            )
        
        # Calculate steps
        num_update_steps = len(train_loader) // config.gradient_accumulation_steps
        num_training_steps = num_update_steps * config.num_epochs
        
        self._setup_optimizer(config, num_training_steps)
        
        # Mixed precision context
        if config.mixed_precision == "bf16":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif config.mixed_precision == "fp16":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = torch.autocast(device_type="cuda", enabled=False)
        
        # Training loop
        print(f"[Training] Starting training for {config.num_epochs} epochs")
        print(f"[Training] Total steps: {num_training_steps}")
        
        self._peft_model.train()
        global_step = 0
        best_eval_loss = float('inf')
        training_history = []
        start_time = time.time()
        
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            
            for step, batch in enumerate(progress):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with autocast_ctx:
                    outputs = self._peft_model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
                
                # Backward pass
                if config.mixed_precision == "fp16" and self._scaler:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item() * config.gradient_accumulation_steps
                epoch_steps += 1
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if config.max_grad_norm > 0:
                        if self._scaler:
                            self._scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._peft_model.parameters(),
                            config.max_grad_norm,
                        )
                    
                    # Optimizer step
                    if self._scaler:
                        self._scaler.step(self._optimizer)
                        self._scaler.update()
                    else:
                        self._optimizer.step()
                    
                    self._scheduler.step()
                    self._optimizer.zero_grad()
                    global_step += 1
                    
                    # Logging
                    if global_step % config.logging_steps == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self._scheduler.get_last_lr()[0]
                        progress.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        })
                        
                        training_history.append({
                            "step": global_step,
                            "loss": avg_loss,
                            "learning_rate": lr,
                            "epoch": epoch + 1,
                        })
                    
                    # Evaluation
                    if eval_loader and global_step % config.eval_steps == 0:
                        eval_loss = self._evaluate(eval_loader, autocast_ctx)
                        print(f"\n[Training] Step {global_step}: eval_loss = {eval_loss:.4f}")
                        
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            self._save_checkpoint(config.output_dir, "best")
                        
                        self._peft_model.train()
                    
                    # Save checkpoint
                    if global_step % config.save_steps == 0:
                        self._save_checkpoint(config.output_dir, f"step_{global_step}")
                    
                    # Callbacks
                    for callback in callbacks:
                        callback(global_step, epoch_loss / epoch_steps)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"[Training] Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        final_path = self._save_checkpoint(config.output_dir, "final")
        
        # Save config
        config_path = Path(config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        total_time = time.time() - start_time
        
        return TrainingResult(
            adapter_path=final_path,
            final_loss=avg_epoch_loss,
            training_steps=global_step,
            total_time_seconds=total_time,
            best_eval_loss=best_eval_loss if eval_loader else None,
            training_history=training_history,
            config=config,
        )
    
    @torch.no_grad()
    def _evaluate(self, eval_loader: DataLoader, autocast_ctx) -> float:
        """Run evaluation"""
        self._peft_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast_ctx:
                outputs = self._peft_model(**batch)
                total_loss += outputs.loss.item()
            
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, output_dir: str, name: str) -> str:
        """Save adapter checkpoint"""
        save_path = Path(output_dir) / name
        self._peft_model.save_pretrained(str(save_path))
        print(f"[Training] Saved checkpoint to {save_path}")
        return str(save_path)
    
    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint"""
        from peft import PeftModel
        
        self._peft_model = PeftModel.from_pretrained(
            self.base_model,
            checkpoint_path,
        )
        print(f"[Training] Resumed from {checkpoint_path}")


def quick_train(
    model,
    tokenizer,
    train_data: str,
    output_dir: str = "./lora_output",
    num_epochs: int = 3,
    **kwargs,
) -> TrainingResult:
    """
    Quick training helper function.
    
    Args:
        model: Base model
        tokenizer: Tokenizer
        train_data: Path to training data (JSON/JSONL)
        output_dir: Output directory
        num_epochs: Number of epochs
        **kwargs: Additional TrainingConfig parameters
        
    Returns:
        TrainingResult
    """
    config = TrainingConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        **kwargs,
    )
    
    dataset = TrainingDataset.from_json(train_data, tokenizer, max_length=config.max_length)
    
    trainer = LoRATrainer(model, tokenizer)
    return trainer.train(dataset, config)
