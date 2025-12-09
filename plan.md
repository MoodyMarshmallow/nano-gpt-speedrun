Here’s a detailed, **staged experiment plan** to upgrade `train_gpt.py` using **SDPA-output gating (“G1”)** from *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*, with logging + contingencies baked in.

---

## 0) What “success” means + constraints (so we don’t accidentally cheat)

### Primary metric(s)

* **Validation loss** on the FineWeb val set (your script already fixes `val_tokens=10485760`). 
* **Speed**: “time-to-target-loss” or “loss-at-fixed-time.” The official modded-nanogpt speedrun targets **mean val loss ≤ 3.28** and forbids data-pipeline modifications. ([GitHub][1])

### Hard constraints (from README rules)

* **Do not modify the train/val data pipelines** (token streams/order). You *may* change batch size, sequence length, attention structure, etc. ([GitHub][1])
* **No extra `torch.compile` / `torch._inductor.config` flags** beyond what’s already in your baseline. ([GitHub][1])

### Experimental rigor expectations (class)

* Use **short runs to screen**, then fewer full runs to confirm. 
* Report **means + variance across seeds**, ideally confidence intervals / basic tests when claiming small effects. 
* Log **seed, commit hash, hyperparams, GPU type/count, runpod instance, time, loss**. 

---

## 1) Hypothesis (directly from the paper → testable knobs)

The paper’s central finding: **a simple head-specific sigmoid gate applied *after SDPA* (G1) consistently improves performance and stability**, largely via:

* **Non-linearity** inserted between the value projection and output projection’s low-rank mapping, increasing expressivity. 
* **Query-dependent sparsity** (many gate values near 0), reducing massive activations and mitigating attention sinks → fewer loss spikes and better training stability. 

**Practical recommendation from the paper:** G1 SDPA-output gating + **moderately increased learning rate**. 

**Speedrun-flavored hypothesis:** If G1-gating reduces loss spikes and improves sample efficiency, we can either:

1. Reach the same target loss in **fewer steps** (speedup), and/or
2. Use a **higher LR** without divergence (speedup), and/or
3. Trade extra accuracy for less compute (optional “shrink model a bit” experiments).

---

## 2) Implementation plan (minimal, compile-friendly, Muon-safe)

### Key gotcha: Muon only wants 2D params

Your script runs `Muon(raw_model.transformer.h.parameters(), ...)` and warns it shouldn’t see 0/1D params. So: **keep gating params 2D only** (i.e., `bias=False`, no scalar learnables). 

### Add new config knobs

Add these to `Hyperparameters` *and* pass into `GPTConfig`:

```python
@dataclass
class Hyperparameters:
    ...
    seed: int = 1337

    # gated attention experiment knobs
    attn_gate: str = "none"        # none|headwise|elementwise|const
    gate_pos: str = "sdpa"         # sdpa|value
    gate_act: str = "sigmoid"      # sigmoid|ns_sigmoid
```

```python
@dataclass
class GPTConfig:
    ...
    attn_gate: str = "none"
    gate_pos: str = "sdpa"
    gate_act: str = "sigmoid"
```

Then instantiate:

```python
model = GPT(GPTConfig(
    vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768,
    attn_gate=args.attn_gate, gate_pos=args.gate_pos, gate_act=args.gate_act
))
```

### Gating scaffold inside `CausalSelfAttention`

This implements the paper’s **G1 (after SDPA), multiplicative sigmoid**, in a **headwise** (cheap) or **elementwise** (stronger) version.

```python
def _apply_gate_act(logits: torch.Tensor, kind: str) -> torch.Tensor:
    # paper ablation: NS-sigmoid removes sparsity by forcing [0.5, 1.0]
    if kind == "sigmoid":
        return torch.sigmoid(logits)
    elif kind == "ns_sigmoid":
        return 0.5 + 0.5 * torch.sigmoid(logits)
    else:
        raise ValueError(f"unknown gate_act={kind}")

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        ...
        self.attn_gate = config.attn_gate
        self.gate_pos  = config.gate_pos
        self.gate_act  = config.gate_act

        # Keep bias=False (Muon expects only 2D params in transformer.h)
        if self.attn_gate == "headwise":
            self.c_gate = nn.Linear(self.n_embd, self.n_head, bias=False)
        elif self.attn_gate == "elementwise":
            self.c_gate = nn.Linear(self.n_embd, self.n_embd, bias=False)
        elif self.attn_gate == "const":
            # input-independent gate: learnable 2D parameter (Muon-safe)
            self.gate_param = nn.Parameter(torch.zeros(self.n_head, self.head_dim))
            self.c_gate = None
        else:
            self.c_gate = None

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Optional G2 (value gating)
        if self.attn_gate != "none" and self.gate_pos == "value":
            if self.attn_gate == "const":
                gate = _apply_gate_act(self.gate_param, self.gate_act)[None, None, :, :]
                v = v * gate
            else:
                gate_logits = self.c_gate(x)  # query-independent per-token gate (paper says weaker than G1)
                gate = _apply_gate_act(gate_logits, self.gate_act)
                if self.attn_gate == "headwise":
                    gate = gate.view(B, T, self.n_head, 1)
                else:
                    gate = gate.view(B, T, self.n_head, self.head_dim)
                v = v * gate

        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )  # (B, nh, T, hd)

        # G1: SDPA-output gating (paper's best)
        if self.attn_gate != "none" and self.gate_pos == "sdpa":
            y = y.transpose(1, 2)  # (B, T, nh, hd)
            if self.attn_gate == "const":
                gate = _apply_gate_act(self.gate_param, self.gate_act)[None, None, :, :]
            else:
                gate_logits = self.c_gate(x)  # uses pre-norm x => query-dependent in the paper framing
                gate = _apply_gate_act(gate_logits, self.gate_act)
                if self.attn_gate == "headwise":
                    gate = gate.view(B, T, self.n_head, 1)
                else:
                    gate = gate.view(B, T, self.n_head, self.head_dim)
            y = (y * gate).contiguous().view_as(x)
        else:
            y = y.transpose(1, 2).contiguous().view_as(x)

        return self.c_proj(y)
```

Why these variants map cleanly to paper claims:

* **`sdpa + sigmoid`**: main recommended path (non-linearity + sparsity). 
* **`ns_sigmoid`**: sparsity-removed control (paper shows weaker gains). 
* **`const`**: “input-independent gating” style control to isolate “non-linearity w/o query dependence.” 

---

## 3) Logging upgrades (so your appendix basically writes itself)

Your script already logs:

* Full code snapshot + `nvidia-smi` into `logs/<run_id>.txt`. 

Add **just enough** to satisfy the FAQ and make analysis painless:

### (A) Add seed, args dump, git commit to the logfile header

```python
import subprocess, json, dataclasses

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

if master_process:
    with open(logfile, "w") as f:
        ...
        f.write(f"git_commit: {get_git_commit()}\n")
        f.write(f"seed: {args.seed}\n")
        f.write("hyperparameters:\n")
        f.write(json.dumps(dataclasses.asdict(args), indent=2))
        f.write("\n")
```

### (B) Seed setting (reproducible without “deterministic mode”)

```python
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
```

That’s consistent with the FAQ’s “don’t force determinism if it hurts speed.” 

### (C) Create a **summary CSV** you append once per run

You want a single row per run for the appendix table:

`experiments/results.csv` columns:

* run_id, date, git_commit, seed
* gate_type, gate_pos, gate_act
* lr, batch_size, device_batch_size, seq_len, iters, warmdown
* final_val_loss, best_val_loss, train_time_ms_final, ms_per_step
* gpu_name, n_gpus, runpod_instance (manual), notes

---

## 4) Experiment design (staged, with decision points)

### Stage 0 — Baseline reproduction (mandatory)

Goal: establish variance + throughput reference.

Runs:

* **3 seeds**, same machine, back-to-back if possible. (More if effect is tiny.) 
  Record:
* final val loss, train_time_ms, ms/step, peak memory, curve.

Success criterion:

* Your baseline numbers are stable enough to detect ~1–3% changes with the seeds you can afford (otherwise increase repeats). 

---

### Stage 1 — “Is gating worth it?” screening (short but informative)

Run **partial-length** training to save budget (explicitly encouraged). 

Settings:

* Keep `val_tokens` fixed for comparability where possible (screening can relax it, but label clearly).
* Use e.g. `num_iterations=1500` for screening.

Experiments (2 seeds each):

1. Baseline
2. `attn_gate=headwise, gate_pos=sdpa, gate_act=sigmoid`
3. `attn_gate=elementwise, gate_pos=sdpa, gate_act=sigmoid`
4. `attn_gate=headwise, gate_pos=sdpa, gate_act=ns_sigmoid` (sparsity control)
5. `attn_gate=const, gate_pos=sdpa, gate_act=sigmoid` (query-dependence control)

Decision rule:

* If (2) or (3) shows **earlier/lower val loss at equal train_time** with ≤~2% step-time overhead → proceed.
* If gains are only with sigmoid (not ns_sigmoid), that supports “sparsity matters” and looks good in your write-up. 

---

### Stage 2 — “Exploit the stability”: learning-rate sweep (core speed lever)

The paper explicitly reports gating improves stability and can support **larger LR**, often improving results. 

Pick the best gating flavor from Stage 1 (likely **headwise+sigmoid** if speed matters, **elementwise+sigmoid** if accuracy matters).

Run a small LR sweep (2 seeds each, same iterations):

* LR multipliers: **1.0×, 1.1×, 1.2×, 1.3×**
* If divergence appears: introduce **warmup_iters (e.g., 100–200)** and retry the smallest failing LR.

Decision rule:

* Choose the highest LR that is stable and improves **val loss vs time**.

---

### Stage 3 — Full-length confirmation + stats (what you’ll report)

For the best config:

* Run **N seeds**:

  * If improvement looks big (≥10–15%): N=3–5 can be sufficient.
  * If chasing 1–3%: aim higher, as budget allows. 

Report:

* mean ± std (or 95% CI) of:

  * final val loss
  * time-to-target (or final train_time_ms at fixed steps)
* Optional: a simple t-test / bootstrap CI if you want a “statistically significant” statement. 

---

## 5) Analysis artifacts (what goes in 2 pages vs appendix)

### In the 2-page report (keep it surgical)

* One plot: **val_loss vs train_time** curves (baseline vs best).
* One table: baseline vs best (loss, time, seeds, hardware).
* A 3–5 sentence explanation tying back to paper’s mechanisms:

  * “G1 adds non-linearity + sparse gating ⇒ more stable ⇒ higher LR/fewer steps.”

### In appendices

* Full ablation table (all runs).
* Raw logs list.
* Optional “mechanism evidence”:

  * Gate score histogram (sigmoid vs ns_sigmoid)
  * Max activation stats (to back “massive activation reduction” claim) 

---

## 6) Tooling/code scaffolds for experiment execution

### A. Simple run launcher (bash)

Create `scripts/run_sweep.sh` that sets env vars and calls torchrun.

* Use env vars so you can avoid editing code between runs.

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

export ATTNGATE=headwise
export GATEPOS=sdpa
export GATEACT=sigmoid
export LR=0.0040
export SEED=1337

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

In `train_gpt.py`, override defaults like:

```python
args.learning_rate = float(os.environ.get("LR", args.learning_rate))
args.seed = int(os.environ.get("SEED", args.seed))
args.attn_gate = os.environ.get("ATTNGATE", args.attn_gate)
args.gate_pos = os.environ.get("GATEPOS", args.gate_pos)
args.gate_act = os.environ.get("GATEACT", args.gate_act)
```

### B. Log parser → CSV + plots (`scripts/parse_logs.py`)

Core idea:

* Parse `logs/<run_id>.txt` lines like:

  * `step:... val_loss:... train_time:...ms step_avg:...ms`
* Emit `results.csv`
* Make 1–2 plots for report/appendix.

Pseudo-code skeleton:

```python
import re, glob, csv

VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:([0-9.]+) train_time:(\d+)ms step_avg:([0-9.]+)ms")

def parse_log(path):
    vals = []
    for line in open(path):
        m = VAL_RE.search(line)
        if m:
            step = int(m.group(1)); val_loss = float(m.group(3)); t_ms = int(m.group(4))
            vals.append((step, t_ms, val_loss))
    final = vals[-1] if vals else None
    best = min(vals, key=lambda x: x[2]) if vals else None
    return {"logfile": path, "final_val": final[2], "final_t_ms": final[1], "best_val": best[2]}

rows = [parse_log(p) for p in glob.glob("logs/*.txt")]
with open("experiments/results.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
```

---

## 7) Contingency plans (pre-written “if X then Y”)

### If gating improves loss but slows step time too much

* Prefer **headwise** or **const** gate (almost no extra FLOPs).
* Use the gained sample efficiency to **reduce `num_iterations`** (primary speed win).
* Keep elementwise gating only if it buys enough steps reduction to offset overhead.

### If gating doesn’t help at all (or regresses)

Run paper-motivated fallbacks:

* **Gate position ablation:** try `gate_pos=value` (G2). Paper shows it helps somewhat but less than G1. 
* **Sparsity matters check:** if `sigmoid` helps but `ns_sigmoid` doesn’t, keep going; if neither helps, your implementation may be wrong or your regime differs. 
* **Non-linearity-only fallback:** apply per-head `F.rms_norm` to the SDPA output (parameter-free non-linearity-ish), as the paper reports normalization at SDPA output can reduce perplexity. 

### If higher LR diverges even with gating

* Add a small warmup (`warmup_iters=100–200`) and retry.
* Back off LR and/or increase warmdown span.

### If results are mixed (helps sometimes, hurts sometimes)

* Characterize *when* it helps (LR range? gating type? seed sensitivity?) and report honestly—this is explicitly acceptable and graded positively if rigorous. 

---

## 8) What you’ll be able to claim (cleanly) in the final write-up

With the above plan, you’ll have:

* A principled architectural change grounded in the gating paper (G1 SDPA-output gating). 
* Ablations that isolate **non-linearity vs sparsity vs query-dependence** (sigmoid vs ns_sigmoid vs const). 
* A methodology that matches class expectations (staged runs, seeds, logs, stats). 
* A paper-ready appendix (CSV summary + raw logs + plots), while keeping the main report within 2 pages.

---

If you want, I can also draft a **one-page “Experiment Log Template” (Markdown)** you can paste into your repo, plus a minimal **results table format** that’s optimized for a 2-page report layout (so you don’t end up fighting spacing on the night before).

[1]: https://github.com/KellerJordan/modded-nanogpt "GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3 minutes"
