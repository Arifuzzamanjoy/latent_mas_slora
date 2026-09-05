"""
Hierarchical Pipeline - Planner → Critic → Refiner → Judger

Implements the core LatentMAS hierarchical reasoning pattern
with latent-space communication between agents.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..agents.agent_pool import AgentExecutor, AgentPool
from ..core.latent_memory import LatentMemory
from ..core.latent_reasoner import LatentReasoner, get_cache_length


@dataclass
class PipelineResult:
    """Result from pipeline execution"""

    question: str
    final_answer: str
    agent_outputs: List[Dict[str, Any]]
    total_tokens: int
    total_latency_ms: int
    latent_steps_total: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalPipeline:
    """
    Hierarchical Multi-Agent Pipeline with Latent Collaboration.

    Pipeline: Planner → Critic → Refiner → Judger

    Each agent:
    1. Receives context via shared latent memory
    2. Performs latent reasoning steps (no token decoding)
    3. Generates text output for traceability
    4. Stores hidden states for next agent

    Optimized for 48GB VRAM:
    - Full BF16 precision
    - Up to 20 latent steps per agent
    - Batch size up to 4
    """

    def __init__(
        self,
        model,
        tokenizer,
        pool: AgentPool,
        memory: LatentMemory,
        reasoner: LatentReasoner,
        device: str = "cuda",
        latent_steps: int = 15,  # Increased for 48GB
        use_latent_transfer: bool = True,
        kv_handoff: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pool = pool
        self.memory = memory
        self.reasoner = reasoner
        self.device = device
        self.latent_steps = latent_steps
        self.use_latent_transfer = use_latent_transfer
        # When True (the LatentMAS reference behaviour) the final agent decodes
        # conditioned on the shared latent working memory. When False the cache is
        # built and then discarded, which is what this file did before - kept so
        # the two can be compared rather than silently swapped.
        self.kv_handoff = kv_handoff

        self.executor = AgentExecutor(pool, tokenizer, device)

    @torch.no_grad()
    def run(
        self,
        question: str,
        agents: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_hidden_states: bool = False,
        task_type: Optional[str] = None,
    ) -> PipelineResult:
        """
        Run the hierarchical pipeline.

        Args:
            question: Input question
            agents: List of agent names to use (default: standard hierarchy)
            max_new_tokens: Override max tokens per agent
            temperature: Override temperature for all agents
            return_hidden_states: Include hidden states in result
            task_type: "numeric" | "mcq" | "text"; selects the answer format
                instruction given to the agent whose output is scored

        Returns:
            PipelineResult with all agent outputs and final answer
        """
        # Clear memory for fresh run
        self.memory.clear()

        # Default agent order
        if agents is None:
            agents = ["Planner", "Critic", "Refiner", "Judger"]

        agent_outputs = []
        total_tokens = 0
        start_time = time.time()

        for i, agent_name in enumerate(agents):
            # Get agent config
            config = self.pool.activate(agent_name)

            # Build prompt
            context = self.memory.get_agent_output(agents[i - 1]) if i > 0 else ""
            # Only the last agent's text is returned as final_answer, so only it
            # is given the task's answer format instruction.
            prompt = self.executor.build_prompt(
                config,
                question,
                context[:500],
                task_type=task_type if i == len(agents) - 1 else None,
            )

            # Tokenize
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,  # Larger context for 48GB
                padding=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            agent_start = time.time()

            # Step 1: Latent reasoning (core LatentMAS)
            if self.use_latent_transfer:
                latent_result = self.reasoner.reason(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_steps=self.latent_steps,
                    past_key_values=self.memory.get_kv_cache() if i > 0 else None,
                )

                # Store in memory
                self.memory.store_hidden_state(agent_name, latent_result.final_hidden)
                self.memory.update_kv_cache(latent_result.kv_cache)

            # Step 2: Text generation (for traceability)
            gen_temp = temperature if temperature is not None else config.temperature
            gen_max = max_new_tokens if max_new_tokens is not None else config.max_tokens

            gen_kwargs = {
                "max_new_tokens": gen_max,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if gen_temp > 0:
                gen_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": gen_temp,
                        "top_p": config.top_p,
                    }
                )
            else:
                gen_kwargs["do_sample"] = False

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            # Decode output
            new_tokens = outputs[0][input_ids.shape[1] :]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Store text output
            self.memory.store_agent_output(agent_name, generated_text)

            agent_latency = int((time.time() - agent_start) * 1000)

            output_entry = {
                "agent": agent_name,
                "role": config.role.value,
                "adapter": config.adapter_name,
                "output": generated_text,
                "input_tokens": input_ids.shape[1],
                "output_tokens": len(new_tokens),
                "latent_steps": self.latent_steps if self.use_latent_transfer else 0,
                "latency_ms": agent_latency,
            }

            if return_hidden_states and self.use_latent_transfer:
                output_entry["hidden_state_norm"] = float(latent_result.final_hidden.norm().item())

            agent_outputs.append(output_entry)
            total_tokens += output_entry["input_tokens"] + output_entry["output_tokens"]

            print(
                f"[{agent_name.upper()}] Generated {output_entry['output_tokens']} tokens in {agent_latency}ms"
            )

        total_latency = int((time.time() - start_time) * 1000)

        return PipelineResult(
            question=question,
            final_answer=agent_outputs[-1]["output"] if agent_outputs else "",
            agent_outputs=agent_outputs,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            latent_steps_total=self.latent_steps * len(agents) if self.use_latent_transfer else 0,
            metadata={
                "num_agents": len(agents),
                "latent_transfer": self.use_latent_transfer,
                "memory_summary": self.memory.get_context_summary(),
            },
        )

    @torch.no_grad()
    def _decode_with_cache(
        self, input_ids, cache, temperature: float, top_p: float, max_new_tokens: int
    ):
        """
        Decode continuing from an accumulated latent KV cache.

        Hand-rolled rather than model.generate(past_key_values=...) because
        passing an external cache through generate() has version-dependent
        semantics; here the attention mask and cache growth are explicit.
        """
        from ..core.latent_reasoner import get_cache_length

        tok = self.tokenizer
        eos_ids = {tok.eos_token_id}
        extra_eos = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(extra_eos, int) and extra_eos >= 0:
            eos_ids.add(extra_eos)

        generated: List[int] = []
        cur = input_ids
        seen = get_cache_length(cache)

        for _ in range(max_new_tokens):
            seen += int(cur.shape[1])
            mask = torch.ones((1, seen), dtype=torch.long, device=self.device)
            out = self.model(
                input_ids=cur,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :].float()

            if temperature and temperature > 0:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                if top_p and 0 < top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True, dim=-1)
                    cut = (torch.cumsum(sp, dim=-1) - sp) > top_p
                    sp[cut] = 0.0
                    sp = sp / sp.sum(dim=-1, keepdim=True)
                    nxt = si.gather(-1, torch.multinomial(sp, 1))
                else:
                    nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)

            tid = int(nxt.item())
            if tid in eos_ids:
                break
            generated.append(tid)
            cur = nxt

        return tok.decode(generated, skip_special_tokens=True).strip(), len(generated)

    @torch.no_grad()
    def run_true_latent(
        self,
        question: str,
        agents: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_hidden_states: bool = False,
        task_type: Optional[str] = None,
    ) -> PipelineResult:
        """
        TRUE LatentMAS: Only final agent generates text.

        Intermediate agents communicate ONLY via latent space,
        achieving 3-7x speedup with 50-80% token reduction.

        Flow:
            Agent1 → [latent only] → Agent2 → [latent only] → ... → FinalAgent → [text]
        """
        self.memory.clear()

        if agents is None:
            agents = ["Planner", "Critic", "Refiner", "Judger"]

        agent_outputs = []
        total_tokens = 0
        start_time = time.time()

        for i, agent_name in enumerate(agents):
            is_final = i == len(agents) - 1
            config = self.pool.activate(agent_name)

            # Build prompt - for latent agents, use minimal context
            if i == 0:
                # First agent gets the question (and the format instruction too
                # when it is also the last, i.e. a single-agent pipeline)
                prompt = self.executor.build_prompt(
                    config, question, "", task_type=task_type if is_final else None
                )
            else:
                # Subsequent agents: minimal prompt, rely on latent state.
                # The final agent is the only one that decodes, so it is the one
                # that needs the task's answer format; it is also the one reading
                # the accumulated cache, so it gets the noise clause.
                prompt = self.executor.build_prompt(
                    config,
                    question,
                    "[Latent context from previous agents]",
                    task_type=task_type if is_final else None,
                    latent_context=is_final and self.kv_handoff,
                )

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
            pre_cache = self.memory.get_kv_cache() if i > 0 else None
            latent_result = None

            # The final agent skips its own latent pass when the cache is handed
            # to the decoder: the accumulated working memory is what it reads.
            if not (is_final and self.kv_handoff):
                latent_result = self.reasoner.reason(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_steps=self.latent_steps,
                    past_key_values=pre_cache,
                )
                self.memory.store_hidden_state(agent_name, latent_result.final_hidden)
                self.memory.update_kv_cache(latent_result.kv_cache)

            agent_latency = int((time.time() - agent_start) * 1000)

            if is_final:
                # ONLY final agent generates text
                gen_temp = temperature if temperature is not None else config.temperature
                gen_max = max_new_tokens if max_new_tokens is not None else config.max_tokens

                gen_kwargs = {
                    "max_new_tokens": gen_max,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                if gen_temp > 0:
                    gen_kwargs.update(
                        {
                            "do_sample": True,
                            "temperature": gen_temp,
                            "top_p": config.top_p,
                        }
                    )
                else:
                    gen_kwargs["do_sample"] = False

                if self.kv_handoff:
                    generated_text, n_new = self._decode_with_cache(
                        input_ids,
                        pre_cache,
                        gen_temp,
                        config.top_p,
                        gen_max,
                    )
                else:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_kwargs,
                    )
                    new_tokens = outputs[0][input_ids.shape[1] :]
                    generated_text = self.tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                    n_new = len(new_tokens)

                gen_latency = int((time.time() - agent_start) * 1000)

                output_entry = {
                    "agent": agent_name,
                    "role": config.role.value,
                    "adapter": config.adapter_name,
                    "output": generated_text,
                    "input_tokens": input_ids.shape[1],
                    "output_tokens": n_new,
                    "latent_steps": 0 if self.kv_handoff else self.latent_steps,
                    "latency_ms": gen_latency,
                    "mode": "latent+text(kv)" if self.kv_handoff else "latent+text",
                    "latent_prefix_len": get_cache_length(pre_cache),
                }
                total_tokens += output_entry["input_tokens"] + output_entry["output_tokens"]

                print(
                    f"[{agent_name.upper()}] Generated {n_new} tokens in {gen_latency}ms "
                    f"({'latent+text(kv)' if self.kv_handoff else 'latent+text'})"
                )
            else:
                # Intermediate agents: latent only, NO text generation
                output_entry = {
                    "agent": agent_name,
                    "role": config.role.value,
                    "adapter": config.adapter_name,
                    "output": "[Latent reasoning only - no text generated]",
                    "input_tokens": input_ids.shape[1],
                    "output_tokens": 0,
                    "latent_steps": self.latent_steps,
                    "latency_ms": agent_latency,
                    "mode": "latent_only",
                }
                total_tokens += output_entry["input_tokens"]

                print(f"[{agent_name.upper()}] Latent reasoning in {agent_latency}ms (no text)")

            if return_hidden_states and latent_result is not None:
                output_entry["hidden_state_norm"] = float(latent_result.final_hidden.norm().item())

            agent_outputs.append(output_entry)

        total_latency = int((time.time() - start_time) * 1000)
        n_latent_agents = len(agents) - 1 if self.kv_handoff else len(agents)

        return PipelineResult(
            question=question,
            final_answer=agent_outputs[-1]["output"] if agent_outputs else "",
            agent_outputs=agent_outputs,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            latent_steps_total=self.latent_steps * n_latent_agents,
            metadata={
                "num_agents": len(agents),
                "mode": "true_latent",
                "kv_handoff": self.kv_handoff,
                "latent_agents": len(agents) - 1,
                "text_agents": 1,
                "memory_summary": self.memory.get_context_summary(),
            },
        )

    def run_true_latent_with_self_consistency(
        self,
        question: str,
        num_samples: int = 3,
        **kwargs,
    ) -> PipelineResult:
        """
        Run TRUE LatentMAS multiple times and vote on final answer.

        Combines latent efficiency with self-consistency for higher accuracy.
        """
        from collections import Counter

        all_answers = []
        all_results = []

        for i in range(num_samples):
            result = self.run_true_latent(
                question,
                temperature=kwargs.get("temperature", 0.5),
                **{k: v for k, v in kwargs.items() if k != "temperature"},
            )

            all_results.append(result)
            answer = self._extract_answer(result.final_answer)
            all_answers.append(answer)
            print(f"  [SC {i + 1}/{num_samples}] Answer: {answer}")

        # Vote
        answer_counts = Counter(all_answers)
        final_answer = answer_counts.most_common(1)[0][0]

        # Return best result with voted answer
        best_result = all_results[0]
        best_result.final_answer = (
            f"[Self-Consistency Vote: {final_answer}]\n\n{best_result.final_answer}"
        )
        best_result.metadata["self_consistency"] = {
            "num_samples": num_samples,
            "all_answers": all_answers,
            "vote_counts": dict(answer_counts),
            "final_answer": final_answer,
        }
        best_result.total_tokens = sum(r.total_tokens for r in all_results)

        return best_result

    def run_with_self_consistency(
        self,
        question: str,
        num_samples: int = 3,
        **kwargs,
    ) -> PipelineResult:
        """
        Run pipeline multiple times and vote on answer.

        Self-consistency improves accuracy by ~5-10% on average.
        """
        from collections import Counter

        all_answers = []
        all_results = []

        for _i in range(num_samples):
            # Increase temperature for diversity
            result = self.run(
                question,
                temperature=kwargs.get("temperature", 0.7),
                **{k: v for k, v in kwargs.items() if k != "temperature"},
            )

            all_results.append(result)

            # Extract answer
            answer = self._extract_answer(result.final_answer)
            all_answers.append(answer)

        # Vote
        answer_counts = Counter(all_answers)
        final_answer = answer_counts.most_common(1)[0][0]

        # Return best result with voted answer
        best_result = all_results[0]
        best_result.final_answer = (
            f"[Self-Consistency Vote: {final_answer}]\n\n{best_result.final_answer}"
        )
        best_result.metadata["self_consistency"] = {
            "num_samples": num_samples,
            "all_answers": all_answers,
            "vote_counts": dict(answer_counts),
            "final_answer": final_answer,
        }
        best_result.total_tokens = sum(r.total_tokens for r in all_results)

        return best_result

    def _extract_answer(self, text: str) -> str:
        """Extract answer from text (for voting)"""
        import re

        # Look for \boxed{X}
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed:
            return boxed[-1].strip().upper()

        # Look for explicit answer patterns
        patterns = [
            r"(?:final\s+)?answer\s*:?\s*([A-D])",
            r"(?:correct\s+)?(?:answer|option)\s*:?\s*([A-D])",
            r"^([A-D])[.)\s]",
        ]

        text_upper = text.upper()
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)

        return "UNKNOWN"
