"""
Sequential Pipeline - Chain-of-Agents

A simpler pipeline where agents process sequentially,
each building on the previous agent's output.
"""

import torch
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..core.latent_memory import LatentMemory
from ..core.latent_reasoner import LatentReasoner
from ..agents.agent_pool import AgentPool, AgentExecutor
from .hierarchical import PipelineResult


class SequentialPipeline:
    """
    Sequential Chain-of-Agents Pipeline.
    
    Simpler than hierarchical - agents process in sequence,
    each receiving the previous agent's text output as context.
    
    Use when:
    - Task decomposition is straightforward
    - Domain experts should process in specific order
    - Less need for critic/refine cycles
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        pool: AgentPool,
        memory: LatentMemory,
        reasoner: LatentReasoner,
        device: str = "cuda",
        latent_steps: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pool = pool
        self.memory = memory
        self.reasoner = reasoner
        self.device = device
        self.latent_steps = latent_steps
        
        self.executor = AgentExecutor(pool, tokenizer, device)
    
    @torch.no_grad()
    def run(
        self,
        question: str,
        agents: List[str],
        context: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        accumulate_context: bool = True,
    ) -> PipelineResult:
        """
        Run sequential pipeline.
        
        Args:
            question: Input question
            agents: List of agent names in order
            context: Initial context
            max_new_tokens: Override max tokens
            temperature: Override temperature
            accumulate_context: Whether to pass all previous outputs or just the last
            
        Returns:
            PipelineResult
        """
        self.memory.clear()
        
        agent_outputs = []
        total_tokens = 0
        start_time = time.time()
        accumulated_context = context
        
        for i, agent_name in enumerate(agents):
            config = self.pool.activate(agent_name)
            
            # Build context
            if accumulate_context:
                ctx = accumulated_context
            else:
                ctx = self.memory.get_agent_output(agents[i-1]) if i > 0 else context
            
            # Build prompt
            prompt = self.executor.build_prompt(config, question, ctx[:1500])
            
            # Tokenize
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            agent_start = time.time()
            
            # Latent reasoning
            latent_result = self.reasoner.reason(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_steps=self.latent_steps,
                question=question,  # For adaptive steps
            )
            
            self.memory.store_hidden_state(agent_name, latent_result.final_hidden)
            
            # Text generation
            gen_temp = temperature if temperature is not None else config.temperature
            gen_max = max_new_tokens if max_new_tokens is not None else config.max_tokens
            
            gen_kwargs = {
                "max_new_tokens": gen_max,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if gen_temp > 0:
                gen_kwargs.update({
                    "do_sample": True,
                    "temperature": gen_temp,
                    "top_p": config.top_p,
                })
            else:
                gen_kwargs["do_sample"] = False
            
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            
            new_tokens = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            self.memory.store_agent_output(agent_name, generated_text)
            
            # Update accumulated context
            if accumulate_context:
                accumulated_context = f"{accumulated_context}\n\n[{agent_name}]: {generated_text[:500]}"
            
            agent_latency = int((time.time() - agent_start) * 1000)
            
            agent_outputs.append({
                "agent": agent_name,
                "role": config.role.value,
                "adapter": config.adapter_name,
                "output": generated_text,
                "input_tokens": input_ids.shape[1],
                "output_tokens": len(new_tokens),
                "latent_steps": self.latent_steps,
                "latency_ms": agent_latency,
            })
            
            total_tokens += agent_outputs[-1]["input_tokens"] + agent_outputs[-1]["output_tokens"]
            
            print(f"[{agent_name.upper()}] Generated {agent_outputs[-1]['output_tokens']} tokens in {agent_latency}ms")
        
        total_latency = int((time.time() - start_time) * 1000)
        
        return PipelineResult(
            question=question,
            final_answer=agent_outputs[-1]["output"] if agent_outputs else "",
            agent_outputs=agent_outputs,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            latent_steps_total=self.latent_steps * len(agents),
            metadata={
                "pipeline_type": "sequential",
                "num_agents": len(agents),
                "accumulate_context": accumulate_context,
            },
        )
