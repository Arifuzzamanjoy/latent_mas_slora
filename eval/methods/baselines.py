"""
Single-model baselines.

These are the controls. Any claim about multi-agent or latent reasoning is a
claim relative to one of these numbers, so they run on exactly the same items,
seeds and decoding settings as the pipelines.
"""

from typing import Any, Dict, List

from ..config import GenSettings
from ..data import EvalItem, LETTERS
from ..extract import UNKNOWN
from .base import Method, Sample

# ─── Prompts ─────────────────────────────────────────────────────────────────

SYS_DIRECT = (
    "You are a careful expert. Answer with the single best option and nothing else."
)
USR_DIRECT_MCQ = (
    "{q}\n\nRespond with only the letter of the correct option in the form \\boxed{{LETTER}}."
)
USR_DIRECT_NUM = (
    "{q}\n\nRespond with only the final numeric answer in the form \\boxed{{ANSWER}}."
)

SYS_COT = (
    "You are an expert clinician, mathematician and computer scientist. "
    "Reason carefully step by step, then commit to one answer."
)
USR_COT_MCQ = (
    "{q}\n\nWork through the problem step by step. Consider each option and rule out the "
    "wrong ones. Only after your reasoning is complete, give the final answer on its own "
    "last line as \\boxed{{LETTER}}."
)
USR_COT_NUM = (
    "{q}\n\nWork through the problem step by step. Only after your reasoning is complete, "
    "give the final answer on its own last line as \\boxed{{ANSWER}}."
)

# The repo's Judger agent prompt, run on the bare model. This isolates the
# prompt from the pipeline: text-mas / latent-mas end with this same prompt, so
# any gain they show over this baseline is attributable to the agents, not the
# wording.
SYS_JUDGER = (
    "You are a Judger Agent responsible for final decisions. "
    "Evaluate all evidence and reasoning to select the best answer. "
    "Be decisive and provide clear justification. "
    "You MUST always end your response with \\boxed{LETTER} where LETTER is the option letter. "
    "State your final answer early in your reasoning, then justify it."
)
USR_JUDGER = (
    "Make the final decision:\n\nQuestion: {q}\n\n"
    "Based on all analysis, select the best answer.\n"
    "You MUST format your final answer as: \\boxed{{LETTER}}\n"
    "State your chosen answer letter FIRST, then provide reasoning.\n\nFinal Answer:"
)


class _PromptBaseline(Method):
    backend_kind = "hf"
    sys_prompt = SYS_COT
    usr_mcq = USR_COT_MCQ
    usr_num = USR_COT_NUM

    def sample(self, item: EvalItem, gen: GenSettings) -> Sample:
        tmpl = self.usr_mcq if item.task_type == "mcq" else self.usr_num
        user = tmpl.format(q=item.question)
        prompt = self.backend.chat_prompt(self.sys_prompt, user)
        out = self.backend.generate(prompt, gen, num_choices=item.num_choices)
        return self._finish(out, item, {"prompt_style": self.name})


class DirectBaseline(_PromptBaseline):
    name = "baseline-direct"
    description = "Bare model, answer-only prompt (no reasoning). The floor."
    sys_prompt = SYS_DIRECT
    usr_mcq = USR_DIRECT_MCQ
    usr_num = USR_DIRECT_NUM


class CoTBaseline(_PromptBaseline):
    name = "baseline-cot"
    description = "Bare model, reason-first chain-of-thought prompt. The honest floor."
    sys_prompt = SYS_COT
    usr_mcq = USR_COT_MCQ
    usr_num = USR_COT_NUM


class JudgerBaseline(_PromptBaseline):
    name = "baseline-judger"
    description = "Bare model with the repo's Judger prompt. Isolates prompt from pipeline."
    sys_prompt = SYS_JUDGER
    usr_mcq = USR_JUDGER
    usr_num = USR_JUDGER


class LogLikelihoodBaseline(Method):
    """
    Extraction-free scoring: pick the option with the highest mean token
    logprob under the model. No generation, no regex, no parse failures - the
    number it produces is a property of the model alone.

    Multiple-choice only; free-form items are skipped.
    """
    name = "baseline-loglik"
    backend_kind = "hf"
    description = "Option log-likelihood scoring (no generation, no answer extraction)."

    def sample(self, item: EvalItem, gen: GenSettings) -> Sample:
        import time
        if item.task_type != "mcq" or not item.choices:
            return Sample("", UNKNOWN, "unsupported", True, 0, 0, 0,
                          {"skipped": "loglikelihood requires multiple choice"})

        mode = self.args.get("loglik_mode", "option")   # option | letter
        user = f"{item.question}\n\nAnswer:"
        prompt = self.backend.chat_prompt(SYS_DIRECT, user)

        if mode == "letter":
            conts = [f" {LETTERS[i]}" for i in range(len(item.choices))]
        else:
            conts = [f" {LETTERS[i]}. {c}" for i, c in enumerate(item.choices)]

        t0 = time.time()
        scores, n_ctx = self.backend.loglikelihood(prompt, conts)
        best = max(range(len(scores)), key=lambda i: scores[i])
        return Sample(
            text=f"\\boxed{{{LETTERS[best]}}}", pred=LETTERS[best], extract_rule="loglik",
            extract_failed=False, prompt_tokens=n_ctx, completion_tokens=0,
            latency_ms=int((time.time() - t0) * 1000),
            extra={"scores": [round(s, 4) for s in scores], "loglik_mode": mode},
        )
