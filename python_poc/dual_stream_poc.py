#!/usr/bin/env python3
import argparse, json, math, os, sys, time
from typing import Dict, List, Any, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from probes import ProbeEngine, MonologueFrame

def softmax_stable(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x = x - x.max(-1, keepdim=True).values
    return torch.softmax(x, dim=-1)

class DualStream:
    """
    Wrap an HF causal LM to emit two synchronized streams.
    """
    def __init__(self, model_name: str = "gpt2", device: str = None, top_k: int = 5):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # For causal LMs it's fine to set pad to eos for batching simplicity
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.top_k = top_k
        self.probes = ProbeEngine(self.model, self.tokenizer)

    @torch.no_grad()
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 50,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 ) -> Dict[str, Any]:

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)

        answer_tokens: List[int] = []
        monologue_frames: List[MonologueFrame] = []

        for step in range(max_new_tokens):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,  # we want full attentions at each forward
                return_dict=True,
            )
            # logits: [B, T, V]
            last_logits = outputs.logits[:, -1, :].squeeze(0)  # [V]
            probs = softmax_stable(last_logits)

            # Top-K for the logit lens
            k = min(self.top_k, probs.shape[-1])
            top_probs, top_idx = torch.topk(probs, k=k, dim=-1)
            top_pairs = [(int(top_idx[i].item()), float(top_probs[i].item())) for i in range(k)]

            # Sample or greedy
            if temperature <= 0.0:
                next_id = int(top_idx[0].item())
            else:
                # Top-p / nucleus (optional simple impl)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    nz = (cum > top_p).nonzero()
                    max_j = nz[0, 0].item() + 1 if nz.numel() > 0 else probs.numel()
                    probs_masked = torch.zeros_like(probs)
                    probs_masked[sorted_idx[:max_j]] = probs[sorted_idx[:max_j]]
                    probs = probs_masked / probs_masked.sum()
                # temperature
                logits_temp = torch.log(probs + 1e-9) / temperature
                probs = torch.softmax(logits_temp, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            # Build monologue frame BEFORE appending the new token
            frame = self.probes.build_frame(
                step=step,
                input_ids=input_ids,
                model_outputs=outputs,
                topk_ids=[tid for tid, _ in top_pairs],
                topk_probs=[p for _, p in top_pairs],
                chosen_id=next_id,
                prompt_text=prompt,
            )
            monologue_frames.append(frame)

            # Append token to sequence
            next_token = torch.tensor([[next_id]], device=self.device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attn_next = torch.ones_like(next_token)
            attn_mask = torch.cat([attn_mask, attn_next], dim=1)

            answer_tokens.append(next_id)

            # Early stop if EOS
            if next_id == self.tokenizer.eos_token_id:
                break

        answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        monologue_lines = [frame.to_string(self.tokenizer) for frame in monologue_frames]
        monologue_text = "\n".join(monologue_lines)

        return {
            "answer_text": answer_text,
            "monologue_frames": [frame.to_dict() for frame in monologue_frames],
            "monologue_text": monologue_text,
            "model": self.model_name,
            "top_k": self.top_k,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, required=True, help="User prompt")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 for greedy")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", type=str, default="dual_stream_output.json")
    args = ap.parse_args()

    ds = DualStream(model_name=args.model, top_k=args.top_k)
    result = ds.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== Answer Stream (A) ===\n")
    print(result["answer_text"])
    print("\n=== Monologue Stream (B) ===\n")
    print(result["monologue_text"])

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

