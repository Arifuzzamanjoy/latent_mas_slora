"""
Training Dataset Utilities

Handles dataset preparation for LoRA fine-tuning:
- Instruction formatting
- Tokenization
- Data collation
"""

import json
import torch
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset


@dataclass
class TrainingExample:
    """Single training example"""
    instruction: str
    input_text: str = ""
    output: str = ""
    system_prompt: str = ""
    metadata: Dict[str, Any] = None
    
    def to_prompt(self, template: str = "alpaca") -> str:
        """Convert to formatted prompt string"""
        if template == "alpaca":
            if self.input_text:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{self.instruction}

### Input:
{self.input_text}

### Response:
{self.output}"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{self.instruction}

### Response:
{self.output}"""
        
        elif template == "chatml":
            system = self.system_prompt or "You are a helpful assistant."
            user_content = f"{self.instruction}\n{self.input_text}" if self.input_text else self.instruction
            return f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{self.output}<|im_end|>"""
        
        elif template == "simple":
            return f"User: {self.instruction}\n{self.input_text}\nAssistant: {self.output}"
        
        else:
            raise ValueError(f"Unknown template: {template}")


class TrainingDataset(Dataset):
    """
    Dataset for LoRA fine-tuning.
    
    Supports multiple formats:
    - JSON/JSONL files
    - HuggingFace datasets
    - Custom data loaders
    
    Example:
        dataset = TrainingDataset.from_json("data.json", tokenizer)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=DataCollator(tokenizer))
    """
    
    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer,
        max_length: int = 2048,
        template: str = "chatml",
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        
        # Pre-tokenize all examples
        self._tokenized = []
        for example in examples:
            prompt = example.to_prompt(template)
            tokens = self._tokenize(prompt)
            self._tokenized.append(tokens)
    
    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text"""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        # Create labels (same as input_ids for causal LM)
        encoded["labels"] = encoded["input_ids"].copy()
        
        return encoded
    
    def __len__(self) -> int:
        return len(self._tokenized)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._tokenized[idx]
    
    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        tokenizer,
        max_length: int = 2048,
        template: str = "chatml",
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
    ) -> "TrainingDataset":
        """
        Load dataset from JSON or JSONL file.
        
        Expected format:
        [
            {"instruction": "...", "input": "...", "output": "..."},
            ...
        ]
        """
        path = Path(path)
        examples = []
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.jsonl':
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        examples.append(TrainingExample(
                            instruction=data.get(instruction_key, ""),
                            input_text=data.get(input_key, ""),
                            output=data.get(output_key, ""),
                        ))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        examples.append(TrainingExample(
                            instruction=item.get(instruction_key, ""),
                            input_text=item.get(input_key, ""),
                            output=item.get(output_key, ""),
                        ))
                elif isinstance(data, dict) and "data" in data:
                    for item in data["data"]:
                        examples.append(TrainingExample(
                            instruction=item.get(instruction_key, ""),
                            input_text=item.get(input_key, ""),
                            output=item.get(output_key, ""),
                        ))
        
        print(f"[Training] Loaded {len(examples)} examples from {path}")
        return cls(examples, tokenizer, max_length, template)
    
    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        tokenizer,
        split: str = "train",
        max_length: int = 2048,
        template: str = "chatml",
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> "TrainingDataset":
        """Load dataset from HuggingFace Hub"""
        from datasets import load_dataset
        
        print(f"[Training] Loading {dataset_name} from HuggingFace...")
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        examples = []
        for item in dataset:
            # Handle different dataset formats
            if "instruction" in item:
                examples.append(TrainingExample(
                    instruction=item.get("instruction", ""),
                    input_text=item.get("input", ""),
                    output=item.get("output", item.get("response", "")),
                ))
            elif "question" in item:
                examples.append(TrainingExample(
                    instruction=item["question"],
                    output=item.get("answer", item.get("response", "")),
                ))
            elif "prompt" in item:
                examples.append(TrainingExample(
                    instruction=item["prompt"],
                    output=item.get("completion", item.get("response", "")),
                ))
            elif "text" in item:
                # Raw text - split on common patterns
                text = item["text"]
                if "###" in text:
                    parts = text.split("###")
                    examples.append(TrainingExample(
                        instruction=parts[0].strip(),
                        output=parts[-1].strip() if len(parts) > 1 else "",
                    ))
                else:
                    examples.append(TrainingExample(instruction=text))
        
        print(f"[Training] Loaded {len(examples)} examples")
        return cls(examples, tokenizer, max_length, template)
    
    @classmethod
    def from_conversations(
        cls,
        conversations: List[List[Dict[str, str]]],
        tokenizer,
        max_length: int = 2048,
    ) -> "TrainingDataset":
        """
        Load from conversation format.
        
        Format: [
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
            ...
        ]
        """
        examples = []
        
        for conv in conversations:
            # Find user/assistant pairs
            user_msg = ""
            assistant_msg = ""
            system_msg = ""
            
            for msg in conv:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    system_msg = content
                elif role == "user":
                    user_msg = content
                elif role == "assistant":
                    assistant_msg = content
            
            if user_msg and assistant_msg:
                examples.append(TrainingExample(
                    instruction=user_msg,
                    output=assistant_msg,
                    system_prompt=system_msg,
                ))
        
        return cls(examples, tokenizer, max_length, template="chatml")


class DataCollator:
    """
    Data collator for training batches.
    
    Handles:
    - Dynamic padding
    - Label masking for instruction tokens
    - Batch creation
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        pad_to_multiple_of: int = 8,
        mask_instruction_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mask_instruction_tokens = mask_instruction_tokens
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of features"""
        # Find max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Pad to multiple
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // 
                      self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        max_len = min(max_len, self.max_length)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for feature in features:
            input_ids = feature["input_ids"][:max_len]
            labels = feature["labels"][:max_len]
            
            # Padding
            padding_length = max_len - len(input_ids)
            
            batch["input_ids"].append(
                input_ids + [pad_token_id] * padding_length
            )
            batch["attention_mask"].append(
                [1] * len(input_ids) + [0] * padding_length
            )
            # Labels: -100 for padding (ignored in loss)
            batch["labels"].append(
                labels + [-100] * padding_length
            )
        
        # Convert to tensors
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }
