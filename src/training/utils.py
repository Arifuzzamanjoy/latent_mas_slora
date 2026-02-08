"""
Training Utilities

Helper functions for:
- Dataset preparation
- LoRA evaluation
- Model export
"""

import json
import torch
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


def prepare_dataset(
    data_source: Union[str, Path, List[Dict]],
    output_path: Optional[str] = None,
    format: str = "alpaca",
    max_samples: Optional[int] = None,
) -> str:
    """
    Prepare and validate a dataset for training.
    
    Args:
        data_source: Input data (file path or list of dicts)
        output_path: Output path for processed data
        format: Output format ("alpaca", "chatml", "simple")
        max_samples: Maximum number of samples to include
        
    Returns:
        Path to prepared dataset
    """
    # Load data
    if isinstance(data_source, (str, Path)):
        with open(data_source, 'r', encoding='utf-8') as f:
            if str(data_source).endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
    else:
        data = data_source
    
    # Limit samples
    if max_samples:
        data = data[:max_samples]
    
    # Validate and normalize
    processed = []
    for i, item in enumerate(data):
        # Normalize keys
        normalized = {
            "instruction": item.get("instruction", item.get("prompt", item.get("question", ""))),
            "input": item.get("input", item.get("context", "")),
            "output": item.get("output", item.get("response", item.get("answer", ""))),
        }
        
        # Validate
        if not normalized["instruction"]:
            print(f"[Warning] Skipping item {i}: missing instruction")
            continue
        if not normalized["output"]:
            print(f"[Warning] Skipping item {i}: missing output")
            continue
        
        processed.append(normalized)
    
    print(f"[Prepare] Processed {len(processed)} valid examples from {len(data)} total")
    
    # Save
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        print(f"[Prepare] Saved to {output_path}")
        return output_path
    
    return processed


@torch.no_grad()
def evaluate_lora(
    model,
    tokenizer,
    eval_data: Union[str, Path, List[Dict]],
    adapter_name: Optional[str] = None,
    max_samples: int = 100,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Evaluate a LoRA adapter on test data.
    
    Args:
        model: Model with LoRA adapter
        tokenizer: Tokenizer
        eval_data: Evaluation data
        adapter_name: Specific adapter to evaluate
        max_samples: Maximum samples to evaluate
        max_new_tokens: Max tokens for generation
        
    Returns:
        Evaluation metrics
    """
    # Load data
    if isinstance(eval_data, (str, Path)):
        with open(eval_data, 'r', encoding='utf-8') as f:
            if str(eval_data).endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
    else:
        data = eval_data
    
    data = data[:max_samples]
    
    # Set adapter if specified
    if adapter_name and hasattr(model, 'set_adapter'):
        model.set_adapter(adapter_name)
    
    model.eval()
    device = next(model.parameters()).device
    
    results = {
        "total_samples": len(data),
        "exact_matches": 0,
        "partial_matches": 0,
        "generations": [],
    }
    
    for item in data:
        instruction = item.get("instruction", item.get("prompt", ""))
        expected = item.get("output", item.get("response", ""))
        input_text = item.get("input", "")
        
        # Build prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Compare
        generated_clean = generated.strip().lower()
        expected_clean = expected.strip().lower()
        
        if generated_clean == expected_clean:
            results["exact_matches"] += 1
        elif expected_clean in generated_clean or generated_clean in expected_clean:
            results["partial_matches"] += 1
        
        results["generations"].append({
            "instruction": instruction[:100],
            "expected": expected[:200],
            "generated": generated[:200],
        })
    
    # Compute metrics
    results["exact_match_rate"] = results["exact_matches"] / len(data)
    results["partial_match_rate"] = results["partial_matches"] / len(data)
    results["combined_rate"] = (results["exact_matches"] + results["partial_matches"]) / len(data)
    
    print(f"[Evaluate] Results:")
    print(f"  Exact matches: {results['exact_matches']}/{len(data)} ({results['exact_match_rate']:.2%})")
    print(f"  Partial matches: {results['partial_matches']}/{len(data)} ({results['partial_match_rate']:.2%})")
    
    return results


def export_adapter(
    model,
    output_path: Union[str, Path],
    adapter_name: Optional[str] = None,
    merge_weights: bool = False,
) -> str:
    """
    Export LoRA adapter weights.
    
    Args:
        model: Model with LoRA adapter
        output_path: Output path
        adapter_name: Specific adapter to export
        merge_weights: Whether to merge into base model
        
    Returns:
        Path to exported adapter
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if adapter_name and hasattr(model, 'set_adapter'):
        model.set_adapter(adapter_name)
    
    if merge_weights:
        # Merge LoRA into base model
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_path))
        print(f"[Export] Merged model saved to {output_path}")
    else:
        # Save adapter only
        model.save_pretrained(str(output_path))
        print(f"[Export] Adapter saved to {output_path}")
    
    return str(output_path)


def merge_adapters(
    model,
    adapter_names: List[str],
    weights: Optional[List[float]] = None,
    output_adapter_name: str = "merged",
) -> None:
    """
    Merge multiple LoRA adapters.
    
    Args:
        model: Model with multiple adapters
        adapter_names: Names of adapters to merge
        weights: Weights for each adapter (default: equal)
        output_adapter_name: Name for merged adapter
    """
    if not hasattr(model, 'add_weighted_adapter'):
        print("[Warning] Model doesn't support adapter merging")
        return
    
    weights = weights or [1.0 / len(adapter_names)] * len(adapter_names)
    
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name=output_adapter_name,
        combination_type="linear",
    )
    
    print(f"[Merge] Created merged adapter '{output_adapter_name}' from {adapter_names}")


def create_training_config_from_agent(agent_config, **overrides) -> Dict[str, Any]:
    """
    Create training config from an AgentConfig.
    
    Useful for training domain-specific adapters based on agent specs.
    """
    from ..agents.configs import AgentConfig
    
    config = {
        "lora_rank": agent_config.lora_spec.rank,
        "lora_alpha": agent_config.lora_spec.alpha,
        "lora_dropout": agent_config.lora_spec.dropout,
        "target_modules": agent_config.lora_spec.target_modules,
    }
    
    config.update(overrides)
    return config
