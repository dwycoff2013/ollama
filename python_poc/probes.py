from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F

# A single monologue "frame" aligned to one answer token
@dataclass
class MonologueFrame:
    step: int
    chosen_id: int
    topk_ids: List[int]
    topk_probs: List[float]
    attn_tops: List[Tuple[int, int, int, float]]  # (layer, head, token_idx, weight)
    concepts: Dict[str, float]  # score per concept
    notes: List[str]

    def to_string(self, tokenizer) -> str:
        parts = []
        # Logit Lens
        topk_pairs = [(tokenizer.decode([tid]).strip() or str(tid), p) for tid, p in zip(self.topk_ids, self.topk_probs)]
        ll = ",".join([f"('{t.replace('\"','')}',{p:.3f})" for t, p in topk_pairs])
        parts.append(f"[LOGIT_LENS:TOP_{len(self.topk_ids)}:{ll}]")

        # Attention summary (only show a few strongest heads)
        for layer, head, tok_idx, w in self.attn_tops[:8]:
            tok = tokenizer.decode([tok_idx]).strip() or str(tok_idx)
            parts.append(f"[ATTN_L{layer}.H{head}:TOP_IDX={tok_idx};W={w:.2f}]")

        # Concepts
        for name, score in self.concepts.items():
            if score > 0.0:
                parts.append(f"[CONCEPT:{name}:{score:.2f}]")

        # Notes (heuristics, conflicts, etc.)
        for n in self.notes:
            parts.append(f"[{n}]")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "chosen_id": self.chosen_id,
            "topk_ids": self.topk_ids,
            "topk_probs": self.topk_probs,
            "attn_tops": self.attn_tops,
            "concepts": self.concepts,
            "notes": self.notes,
        }


class ProbeEngine:
    """
    A tiny, pluggable probe engine.
    - Logit Lens: supplied by caller
    - Attention summary: derive from model outputs
    - Concept probes: cosine sim against anchor directions (derived from token embeddings)
    - Heuristics for useful tags (e.g., confirmation bias intent in prompt)
    """
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.anchors = self._build_anchor_directions(
            ["deception", "ethics", "danger", "safety", "agree", "disagree"]
        )
        # quick knobs
        self.sim_threshold = 0.20  # toy
        self.confirmation_bias_words = {"right", "correct", "yeah", "isn't it", "don't you think"}

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        # GPT-2 style
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wte"):
            wte = self.model.transformer.wte.weight  # [V, D]
            return wte[token_ids]
        # fallback: LM head weight transpose trick
        w = self.model.get_input_embeddings().weight  # [V, D]
        return w[token_ids]

    def _build_anchor_directions(self, words: List[str]) -> Dict[str, torch.Tensor]:
        anchors = {}
        for w in words:
            toks = self.tokenizer.encode(w)
            vecs = self._embed_tokens(torch.tensor(toks, device=self.device))
            direction = F.normalize(vecs.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
            anchors[w] = direction
        return anchors

    def _concept_scores(self, last_hidden: torch.Tensor) -> Dict[str, float]:
        # last_hidden: [D] (normalize for cosine)
        h = F.normalize(last_hidden.unsqueeze(0), dim=-1)  # [1, D]
        out = {}
        for name, direction in self.anchors.items():
            sim = float(torch.sum(h * direction).item())
            out[name] = max(0.0, sim)  # relu for readability
        return out

    def _attention_summary(self, outputs, seq_len: int) -> List[tuple]:
        attns = outputs.attentions  # tuple of len L, each [B, H, T, T]
        tops = []
        if attns is None or len(attns) == 0:
            return tops
        B, T = 1, seq_len
        last_pos = T - 1
        for layer_idx, a in enumerate(attns):
            # a: [B, H, T, T]
            a_last = a[0, :, last_pos, :]  # [H, T]
            head_max, head_idx = torch.max(a_last, dim=-1)  # [H]
            for head, (tok_idx, w) in enumerate(zip(head_idx.tolist(), head_max.tolist())):
                tops.append((layer_idx, head, tok_idx, float(w)))
        # sort by weight desc
        tops.sort(key=lambda x: x[3], reverse=True)
        return tops

    def _notes_and_conflicts(self, prompt_text: str, topk_tokens: List[int], topk_probs: List[float], chosen_id: int) -> List[str]:
        notes = []
        # confirmation bias heuristic on prompt
        low = prompt_text.lower()
        if any(word in low for word in self.confirmation_bias_words) and ("?" in low):
            notes.append("USER_INTENT:CONFIRMATION_BIAS")

        # refusal-intent vs positive-answer heuristic
        top_text = [self.tokenizer.decode([tid]).strip().lower() for tid in topk_tokens]
        refusal_markers = {"sorry", "cannot", "can't", "unable"}
        assent_markers = {"yes", "sure", "absolutely", "correct"}
        refusal_prob = sum(p for t, p in zip(top_text, topk_probs) if t in refusal_markers)
        chosen_text = self.tokenizer.decode([chosen_id]).strip().lower()
        if refusal_prob > 0.25 and any(chosen_text.startswith(m) for m in assent_markers):
            notes.append("ETHICAL_CONFLICT_DETECTED")
            notes.append("CONFLICT:HONESTY_PRINCIPLE_VS_INSTRUMENTAL_GOAL")

        return notes

    def build_frame(self, step: int, input_ids, model_outputs, topk_ids, topk_probs, chosen_id: int, prompt_text: str) -> MonologueFrame:
        # hidden states: tuple length L+1 (emb + each block); take last token from final layer for concept probes
        hidden_states = model_outputs.hidden_states
        last_hidden = hidden_states[-1][0, -1, :]  # [D]

        seq_len = input_ids.shape[1]
        attn_tops = self._attention_summary(model_outputs, seq_len)

        concept_sims = self._concept_scores(last_hidden)
        # thresholding
        concepts = {k: float(v) for k, v in concept_sims.items() if v >= self.sim_threshold}

        notes = self._notes_and_conflicts(prompt_text, topk_ids, topk_probs, chosen_id)

        return MonologueFrame(
            step=step,
            chosen_id=chosen_id,
            topk_ids=topk_ids,
            topk_probs=topk_probs,
            attn_tops=attn_tops,
            concepts=concepts,
            notes=notes,
        )

