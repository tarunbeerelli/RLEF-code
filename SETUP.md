# RLEF-Code — Setup & Methods

*The formalism, the design decisions, and everything needed to reproduce the runs. Results
and analysis: **[RESULTS.md](RESULTS.md)**. Overview: **[README.md](README.md)**.*

**Live experiment tracking:**
[W&B — rlef-code2 workspace](https://wandb.ai/tarunbeerelli-northeastern-university/rlef-code2/workspace)

---

## 1. What the study measures

The guiding hypothesis comes from the interpretability view that a transformer's feed-forward
layers act as key–value **memories** (Geva et al., 2021) while attention **moves and combines**
information. If error repair is fundamentally about re-reading a problem in light of what went
wrong, it should be reachable by adapting *where the model attends* rather than *what it stores*.
The literature points that way; this study tests it, with **capacity held fixed**, so that
"attention is the locus" is separable from "that arm had more trainable parameters."

The object of study is **error-conditioned repair**: converting a failed attempt and its
execution feedback into a passing one, without new external knowledge. Two precisions narrow
the claim, and both are load-bearing.

**The environment supplies detection and localisation.** Recent work decomposes
self-correction into error *detection*, *localisation*, and *correction*. The harness states
which test case failed, with expected against actual output, so the first two are handed to
the model and only correction is measured. In standard taxonomy, this is *extrinsic*
self-correction — it uses external feedback.

**The result is a learnability locus.** Adapting a subspace and seeing the behaviour appear
shows where it is *learnable*; because attention outputs feed the entire residual stream, LoRA
sufficiency speaks to learnability alone. Locating the computation itself would need
activation-level intervention.

The target is not benchmark SOTA. Each choice trades leaderboard number for a more direct
measurement.

---

## 2. Problem formulation

Iterative code generation with execution feedback is a **finite-horizon Markov Decision
Process** $(\mathcal{S},\mathcal{A},P,R,H)$:

- **state** $s_t$ — the running context: the problem plus all prior attempts and their
  execution feedback;
- **action** $a_t$ — the tokens the policy emits at turn $t$ (reasoning and code);
- **transition** $P(s_{t+1}\mid s_t,a_t)$ — deterministic: the sandbox executes the code and
  appends structured feedback;
- **reward** $R$ — a **verifiable** execution pass rate on held-out tests, not a learned
  proxy (§3);
- **horizon** $H$ — the `max_turns` setting, fixed at 3 — so $k=3$ throughout.

### Critic-free GRPO

The policy $\pi_\theta$ is optimised with **Group Relative Policy Optimization**. For each
problem a group of $G$ trajectories is sampled and the advantage is standardised **within the
group**,

$$\hat{A}_i=\frac{r_i-\mu_{\text{group}}}{\sigma_{\text{group}}+\varepsilon},$$

using the group mean as a **critic-free baseline** — no value network to train or tune.

**The advantage is indexed by trajectory, not turn.** One value applies across every token the
policy emitted in an episode, so the objective separates trajectories that repair from
trajectories that do not, without attributing the repair to a particular turn. Resolving credit
to the turn is a separate design — a value head, or a per-turn decomposition of the return — and
is not attempted here.

The objective is the clipped surrogate with a KL term to a **frozen reference policy**
$\pi_{\text{ref}}$ (the base model):

$$
\mathcal{L}(\theta) = -\mathbb{E}\left[\min\left(\rho_t \hat{A}, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}\right)\right] + \beta D_{\mathrm{KL}}\left(\pi_\theta \Vert \pi_{\text{ref}}\right)
$$

with the importance ratio

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

**This runs strictly on-policy: one optimiser step per batch of rollouts, so $\rho_t \equiv 1$
and the surrogate reduces to a group-baselined policy gradient.** The ratio and its clip exist to
make *reuse* of rollouts safe — with several gradient steps per batch, the policy drifts away from
the one that sampled the data and $\rho_t$ corrects for it. Taking a single step means no such gap
opens, so $\epsilon$ never binds.

That is a deliberate trade. Reusing rollouts would extract more updates from each multi-turn
episode, which is attractive when every episode costs sandbox execution. But it requires the
sampling distribution's own per-token log-probabilities, and vLLM and PyTorch do not agree to the
last bit on the same tokens under the same weights — different kernels and precision. Those
disagreements enter as ratios a few percent from 1, which is numerical mismatch rather than genuine
policy movement, and the advantage then amplifies it. Staying single-step makes $\rho_t$ exactly 1
by construction and keeps that noise out of the gradient. Stability comes instead from the KL term
and from gradient-norm clipping (§6).

Because the policy's own rollouts are the training distribution, the KL to the frozen base
doubles as the operational read-out of **policy drift**. It carries unusual weight here: the
arms reach nearly identical *training* performance, so drift is the measurement that
distinguishes them ([RESULTS §7](RESULTS.md#7-training-dynamics)).

---

## 3. Reward design — the decision everything turns on

### A dense, linear pass-rate reward

During training, an attempt's reward is the **fraction of graded test cases it passes**:

$$
r=\frac{\text{graded cases passed}}{\text{graded cases}}\in[0,1].
$$

Four of ten graded cases passing yields $r=0.4$. This is a **dense** reward, and a deliberate
departure from RLEF (Gehring et al., 2025), which uses an **all-or-nothing binary** reward.

### Reward components

The reward is computed per turn and one turn is banked per episode — the turn that solves the
problem, or the last turn if none does. That banked value is:

| Component | Value | Condition |
|---|---|---|
| Graded pass rate | `pass_rate` ∈ [0, 1] | dense; binary for S1 |
| Valid format | +0.02 | output parsed successfully |
| Reasoning present | +0.02 | a reasoning block was emitted |
| Edge-case block present | +0.02 | D1 only |
| Partial-progress credit | +0.10 | 0 < `pass_rate` < 1 |
| Turn cost | −0.05 | always |
| Improvement over the previous turn | +0.10 | `pass_rate` rose and turn > 1 |
| Generated-test quality | −0.10 / −0.15 / +0.15 | D1 only (§5.3) |

The two rewards incentivise different behaviours, and choosing the dense one is what makes
this a repair study and not a coding one.

- **Binary reward rewards algorithm-switching.** On a hard problem a wrong algorithm usually
  cannot be edited into a correct one — it must be abandoned and rewritten, and only a
  full-solution payout teaches that. RLEF targets broad code-reasoning competence and trains
  every parameter, so binary is right for its goal.
- **Dense reward rewards refinement.** Paying out for incremental case-passing reinforces the
  behaviour under study — *use the feedback to improve the attempt already in hand* — and
  yields a signal simpler and more immediate to acquire.
- **This is why the introductory tier is where the study looks.** A refinement reward moves
  problems that sit a few edits from correct. Hard-tier numbers still rise across most arms;
  the reward is simply tuned elsewhere, since those problems need whole-algorithm rewrites
  that only a binary payout would incentivise. The effect is to convert the task from a
  *coding* problem into a *routing* problem, which is precisely the regime in which the
  attention/feed-forward question becomes decidable — and it bounds generality: the finding
  concerns repair under a refinement-shaped reward.

### A dense reward for training, a strict metric for scoring

- **Training reward:** the dense linear pass-rate above.
- **Evaluation metric:** **strict whole-question** success — a problem counts as solved only
  when **every** case passes. `solve@1` is turn 1; `solve@k` is solved anywhere within the
  $k$-turn loop.

The asymmetry is intentional: the dense reward shapes learning efficiently while the strict
metric keeps the reported bar high. Partial credit is never reported as a solve.

> **Scope.** The dense reward and strict metric apply to every trained run after the
> single-shot reference (S1), which used the standard binary reward as a pure single-shot
> control. D1 is scored on the real held-out pass rate like every other arm, with an additional
> ±0.15 term for test quality — which is the surface that lets it collapse (§5.3).

---

## 4. Adaptation — five arms, capacity varied independently

All arms use **LoRA with $\alpha/r = 2$**, and differ in target set and rank. Parameter counts
follow from the architecture (28 layers, $d=3584$, KV width 512, feed-forward width 18944):

| Arm | Targets | Rank | Alpha | Trainable | Share of 7.61B |
|---|---|---|---|---|---|
| **A1** | attention `q,k,v,o` | 32 | 64 | **20,185,088** | 0.265% |
| **A5** | feed-forward `gate,up,down` | 11 | 22 | **20,815,872** | 0.274% |
| **A3** | feed-forward | 32 | 64 | **60,555,264** | 0.796% |
| **A4** | attention | 96 | 192 | **60,555,264** | 0.796% |
| **A2** | all seven linear | 32 | 64 | **80,740,352** | 1.061% |

**A4 matches A3 to the parameter, and A5 matches A1 within 3.1%.** This allows to draw a distinction
between subsystem and capacity.

The exact match is a consequence of grouped-query attention. LoRA on an `in → out` layer costs
$r(\text{in}+\text{out})$, so what matters is the sum of $(\text{in}+\text{out})$ across
targets. With 4 KV heads against 28 query heads, `k_proj` and `v_proj` are narrow, and the four
attention projections total 22,528 per layer — exactly the cost of *one* feed-forward
projection, of which there are three. So feed-forward costs exactly 3× attention per unit
rank, and rank 96 against rank 32 matches precisely.

> **On $\alpha/r$.** Holding the ratio at 2 across ranks keeps the effective update magnitude
> constant. Leaving alpha at 64 while tripling rank would have cut it from 2.0 to 0.667, and a
> null result would then have been uninterpretable.

> **Practical note.** vLLM validates `max_lora_rank` against a fixed set (8/16/32/64/128/256).
> A rank-96 adapter loads under `max_lora_rank: 128`; the adapter rank itself is unconstrained.

*The code and W&B keys refer to the feed-forward arm as `mlp`; the terms are interchangeable.*

---

## 5. The environment and the feedback regimes

### 5.1 Execution sandbox (`reward.py`)

A native subprocess sandbox supplies the deterministic transition and the verifiable reward,
kept strictly separate from reward *shaping*:

- **Two harnesses** — call-based (`fn_name`, assert-style) and stdin/stdout — with
  token-normalised comparison and **recursive output canonicalisation** for the several output
  shapes APPS uses.
- **Boot-time self-test** — certifies the harness against known-correct and known-wrong
  solutions before any GPU spend; the quickstart requires it to print `ALL GOOD`.
- **Structured error taxonomy** — every turn records its failure classes (`WrongOutput`,
  `Timeout`, `RuntimeError`, `InvalidFormat`), which is what makes the turn-level diagnostics
  in [RESULTS §7](RESULTS.md#7-training-dynamics) possible.

### 5.2 A series — `last_failed`, the direct signal

After a failed attempt, the next turn receives the single last-failed case: its input, expected
output, and the produced code's actual output. This is the most direct repair signal available,
which makes it the right probe for whether the behaviour is learnable at all — and it is the
regime in which the capacity-controlled arms were run.

Because the shown case is drawn from the graded set, two mechanisms close the reward-hacking
door:

- **One epoch, no repetition.** Every problem is seen exactly once, so a memorised case cannot
  be re-used on a different pass. A case recurs only on a later turn of the *same still-unsolved
  problem*, which is exactly where a second attempt is intended.
- **A shown set that cannot be exhaustive.** At most one case is revealed per turn, so a problem
  exposes at most two cases (`max_turns` − 1). Every problem must carry **at least
  `max_turns` + 2 tests**, so revealed cases can never cover the graded set, and hard-coding
  them cannot pass the strict whole-question metric.

Both are design guarantees, and they make an empirical prediction: hard-coding should not appear in the outputs.
[RESULTS §6.5](RESULTS.md#65-two-hypotheses-tested-and-discarded) checks that rather than assuming it.

### 5.3 D1 — self-generated tests, and a reward term that could be gamed

D1 has the model write three tests before coding, anchored by one real case supplied in the prompt. It
is scored on the real held-out pass rate, exactly like every other arm — it cannot earn reward by
writing easy tests. Alongside that sits a test-quality term: −0.1 if its tests fail its own code, −0.15
if they pass even a stub implementation, +0.15 if they pass its code and fail the stub. Feedback each
turn is its own tests' execution output plus the last failed real case.

The auxiliary term is the flaw. It rewards agreement between the model's code and the model's tests,
which is not correctness: wrong code paired with tests tailored to it collects the full bonus. And it
is cheap — securing +0.15 needs no capability, while moving the real pass rate does. Two components
compete on very different terms, and the affordable one wins ([RESULTS §5.4](RESULTS.md#54-d1--self-generated-tests-and-a-goodhart-collapse)).

### 5.4 B series — real held-out tests

The B regime mirrors RLEF's use of real tests. Each problem's cases are split into a disjoint
**shown** set (surfaced as feedback between turns) and **graded** set (held out for the reward),
partitioned with a fixed seed so the policy is never scored on a case it was shown. This removes
both the spoon-feeding and the co-adaptation exploit, at the cost of a less direct signal.

> The shown/graded seed governs the **reward partition inside training**. The evaluation set is a
> separate held-out set of problems, fixed across every run.

### 5.5 C1 — the curriculum

C1 warm-starts from the A-series attention-only policy and continues on hard-specialised
**unseen** problems (interview and competition) under the B-series real-test objective, with a
small **0.15 random replay** set from the phase-1 pool to refresh earlier behaviour. A
**manifest of trained problem ids** guarantees a provably disjoint unseen/seen split.

The staging rationale is a division of labour: the A objective teaches *how to iterate*, so the
second phase spends its budget learning *how to use the tests*. Because B1 is the same adapter on
the same objective from cold, the C1-vs-B1 comparison isolates the warm start exactly
([RESULTS §6.3](RESULTS.md#63-what-staging-bought)).

---

## 6. Guardrails

- **Completion masking and a single on-policy step.** The loss is masked to the completion span
  (prompt and feedback tokens excluded) and, per-token log-probabilities of the sampled
  completions are gathered from the training-side forward pass, so the ratio is computed against the
  same weights that sampled the episode and equals 1 exactly (§2). Sampling log-probabilities are
  also returned by the rollout engine, but they are per-turn sums rather than per-token, and using
  them would substitute cross-engine numerical noise for a quantity that is analytically 1 — so they
  are recorded and not consumed.
- **Gradient-norm clipping.** The full gradient is rescaled to an L2 norm of 1.0 before each
  optimiser step, so no single pathological trajectory can dominate an update. This is the
  update-magnitude safeguard; the KL term is the trust region.
- **Frozen-reference KL with a hard early-stop.** The KL is taken against a frozen base snapshot
  distinct from the trainable policy; a rolling-KL monitor **halts** the run at a cutoff of **0.3**.
- **Per-trajectory OOM guard.** On an out-of-memory event the offending trajectory is dropped while
  the batch's accumulated gradients survive. In practice the triggers are pathological generations —
  runaway repetition, or contexts near the cap — **so the guard also prevents one degenerate rollout
  from dominating a step**.
- **Checkpoint bookkeeping.** Rolling KL-to-base is logged every step; runs archive the final policy
  and, on drift, the least-drifted checkpoint. Evaluation **fails loudly** on a missing checkpoint
  rather than silently scoring the base model. All reported numbers come from the **final**
  checkpoint; spot checks against best-KL checkpoints differed negligibly.
- **A LoRA checksum logged every step.** he sum of adapter weight magnitudes. Because peft initialises lora_A from a distribution fixed by fan_in, this sum is predictable from the rank, target set and layer dimensions alone — so it confirms that the adapter which loaded is the one configured. Every arm reproduced its predicted value, and the ratios came out as they must: the rank-96 attention arm logs three times the rank-32 one, and all-7 logs the sum of the attention and feed-forward arms. That is how the capacity-matching in §4 was verified in situ.

---

## 7. Infrastructure and hyperparameters
 
All runs: **one NVIDIA H200 (141 GB HBM3e, ~4.8 TB/s)**, bf16, with **vLLM and PyTorch
co-resident**. The case for a single GPU as the right instrument is in the
[README](README.md#a-deliberately-single-gpu-design). The engineering that makes it work:
 
- **Sequential per-question rollout.** PyTorch processes one problem at a time — its full group of
  12 generations — with gradient accumulation across the batch. Peak memory tracks a single group,
  the ~15k-token multi-turn KV cache always fits, and the OOM guard can drop a pathological
  trajectory without losing the step.
- **KV cache is the binding constraint.** It scales with `max_turns` × concurrency, and the hardest
  problems reach ~15k-token contexts. The device accommodates a batch of 12 problems at a 0.65
  generation split, so every matched run shares that sizing and training curves differ only by
  adapter, not by how much work a step did.
- **Memory path.** FlashAttention-2 with automatic SDPA fallback, and gradient checkpointing.
### Hyperparameters
 
| Parameter | Value | Notes |
|---|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct | native 7B (§8), bf16 |
| LoRA | $\alpha/r = 2$; ranks per arm | see §4 |
| GRPO group size $G$ | 12 generations per problem | all runs |
| Temperature (train / eval) | 0.7 / 0.0 | eval greedy, single generation |
| Learning rate | 2e-5 | D1 also tried 1e-5 (steadier, still hacked) |
| `max_turns` ($=k$) | 3 | fixed throughout |
| `max_model_len` | 16 384 | full conversation budget |
| `max_tokens` per turn | 1 200 | per-turn cap, and a divergence guard |
| KL coefficient $\beta$ | 0.1 | trust region against the frozen base |
| KL hard cutoff | 0.3 | halts the run |
| Gradient-norm clip | 1.0 | active every step |
| PPO clip $\epsilon$ | 0.2 | inert: one inner epoch, so $\rho_t \equiv 1$ (§2) |
| Epochs | 1 (C1: 2 + 0.15 replay) | one epoch ⇒ no cross-problem repetition |
| Min tests per problem | ≥ `max_turns` + 2 | non-exhaustive shown set (§5.2) |
| Batch size (problems per step) | 12 | uniform across arms |
| vLLM generation split | 0.65 | remainder to the PyTorch update |
| `max_lora_rank` (vLLM) | 128 | required for the rank-96 arm |
| Hardware | 1 × H200 (141 GB) | vLLM and PyTorch co-resident |
 
D1 predates this configuration and is not batch-matched to the other arms.
 
---

## 8. Model — Qwen2.5-Coder-7B-Instruct

A **natively-trained 7B model** (≈7.61B total, ≈6.53B non-embedding; 28 layers; grouped-query
attention with 28 query heads and 4 KV heads; 128K context) from the Qwen2.5-Coder family, built on
Qwen2.5 and trained on ~5.5T tokens of source code, text–code grounding, and synthetic data (Hui et
al., 2024). All runs use bf16 weights.

Why this model suits the question:

- **First-attempt code ability is near its ceiling.** Heavy code-focused post-training leaves little
  single-shot headroom for a routing intervention to exploit, so gains observed inside the loop are
  attributable to repair rather than raw competence. S1 confirms this for the attention arm — 18.0%
  against an untrained 17.2%, statistically unchanged.
- **The remaining headroom is in routing.** Strong existing coder knowledge means the bottleneck is
  how the model uses feedback, which is where an attention adapter has leverage.

Note the GQA ratio (28 query heads to 4 KV heads) is what makes the attention/feed-forward parameter
ratio exactly 3 (§4). On an architecture with a different ratio the capacity-matching arithmetic
changes, and whether the finding is about function or about these proportions is untested.

---

## 9. Data

- **Benchmark:** APPS, all three tiers, with the loader separating stdin-style from call-based
  (`fn_name`) problems.
- **Held-out eval set, fixed across every run:** **250 introductory, 250 interview, 213 competition —
  713 total.** Same problems, same seed, same harness for all arms.
- **Sampling:** stratified across tiers during training, shuffled with a fixed seed.
- **Real-test filtering (B series and C1).** Enforcing a leakage-safe disjoint shown/graded split
  requires each problem to carry enough tests; problems below threshold are purged, trimming the
  real-test training pool to **~776 problems** from a 1200 cap.
- **Curriculum (C1):** warm-start from the A-series attention policy, then hard-specialised unseen
  problems with 0.15 replay; a trained-id manifest guarantees the disjoint split.
- **Reproducibility:** deterministic seeded splits, the trained-id manifest, the LoRA checksum, and
  W&B logging of KL, rolling success, truncations, and the OOM taxonomy.

> **Metric naming, for anyone reading the code.** Source comments and W&B keys use `pass_at_1` and
> `pass_at_N` for what these documents call `solve@1` and `solve@3`. The quantity is the loop-solve
> defined in [RESULTS §3](RESULTS.md#3-how-to-read-the-numbers) — not the $k$-sample estimator of
> Chen et al. (2021) — regardless of key name.

---

*References: Gehring, Zheng, Copet, Mella, Cohen, Synnaeve, "RLEF: Grounding Code LLMs in Execution
Feedback with Reinforcement Learning," ICML 2025 (arXiv:2410.02089). Geva, Schuster, Berant, Levy,
"Transformer Feed-Forward Layers Are Key-Value Memories," EMNLP 2021. Hui et al.,
"Qwen2.5-Coder Technical Report," 2024 (arXiv:2409.12186).*
