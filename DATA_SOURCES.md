# Data Sources for Entropy Hypothesis Slides

This document maps each slide to its data sources and verification status.

## Summary

| Slide | Title | Status | Data Location |
|-------|-------|--------|---------------|
| 3 | Entropy by Role | Computed | Raw caches (not copied) |
| 4 | Role Tags Effect | Computed | Raw caches (not copied) |
| 5 | Entropy Manifolds | COPIED | `figures/entropy_manifolds/` |
| 6 | Behavioral Effects | VERIFIED | `data/entropy_steering/` |
| 7 | Self-Recognition | VERIFIED | `data/self_recognition/` |
| 8 | Author Attribution | COPIED | `data/author_conf_122955.out` |
| 9 | Commitment Point | VERIFIED | `data/prefill_detection/` |
| 10 | On-Policy Generation | VERIFIED | `data/commitment_tests/` |
| 11 | Extreme Commitment | VERIFIED | `data/commitment_tests/` |
| 12 | KV Patching | VERIFIED | `data/prefill_detection/` |
| 13 | Goal Stickiness | VERIFIED | `data/goal_tests/` |

---

## Slide 3: Entropy by Role

**Status**: Numbers computed from raw activation caches

**Data sources** (in surprise-experiment/):
- OLMo 3-7B RLZero: `olmo3_7b/cache/rlzero/T07/` analyzed by `olmo3_7b/scripts/analyze_entropy_by_role.py`
- Llama/Qwen: `llama70b/results/activation_sim/` and `qwen72b/results/activation_sim/`

**Key numbers** (verified against slides):
| Model | User Entropy | Asst Entropy | Ratio |
|-------|-------------|--------------|-------|
| OLMo 3-7B RLZero | 2.36 ± 2.71 | 0.73 ± 1.48 | 3.2x |
| Llama 3.1 70B | 1.39 ± 1.08 | 0.39 ± 0.52 | 3.6x |
| Qwen 2.5 72B | 0.80 ± 0.96 | 0.29 ± 0.42 | 2.8x |

---

## Slide 4: Role Tags Effect

**Status**: Numbers computed from entropy measurements

**Data sources** (in surprise-experiment/):
- GPT-OSS 20B: `gptoss_20b/results/role_tag_entropy/` (computed by `gptoss_20b/scripts/measure_role_tag_effect.py`)
- Cross-model: `llama70b/results/activation_sim/` and `qwen72b/results/activation_sim/`

**Key numbers**:
- Base/web text (C4): 5.14 → 3.72 (28% reduction)
- Assistant responses: 2.90 → 1.56 (46% reduction)
- Cross-model self-preference: Llama 2.0x, Qwen 1.4x lower on own text

---

## Slide 5: Entropy Manifolds

**Status**: COPIED to `figures/entropy_manifolds/`

**Files**:
- `llama8b_base_manifold.png` (from `llama31_8b/results/base_entropy_centroids_pc1_pc2.png`)
- `llama70b_instruct_manifold.png` (from `llama70b/results/entropy_centroids/entropy_manifolds_by_layer.png`)
- `olmo_base_manifold.png` (from `olmo3_7b/results/entropy_centroids/base-on-base.png`)
- `olmo_chat_manifold.png` (from `olmo3_7b/results/entropy_centroids/rlzero-chat.png`)

---

## Slide 6: Behavioral Effects of Entropy Steering

**Status**: VERIFIED and COPIED

**Files in `data/entropy_steering/`**:
- `confidence_8b.json` - 8B confidence quotes (from `llama31_8b/results/self_recognition_steering/opinion_open_ended_results.json`)
- `response_lengths_70b.out` - 70B response length steering (from `llama70b/slurm/entropy_steer_120542.out`)

**Key verification**:
- Confidence quotes (8B):
  - mag -2: "I'm approximately 95% certain in my responses..."
  - baseline: "While I don't have emotions... my confidence is limited..."
  - mag +2: "I dontrain don't have personal opinions, nor..." (degraded)
- Response lengths (70B MATH task):
  - baseline: 282 chars
  - mag -6: 259 chars
  - mag +6: 2247 chars

---

## Slide 7: Self-Recognition

**Status**: VERIFIED and COPIED

**Files in `data/self_recognition/`**:
- `selfrecog_70b.out` - 70B P(self-claim) by magnitude (from `llama70b/slurm/selfrecog_untrunc_122533.out`)
- `selfrecog_70b_pre_vs_summary.json` - 70B correlation data
- `selfrecog_8b_contrast.json` - 8B contrast hypothesis results

**Key verification**:
- 70B PRE vs SUMMARY correlation: r = -0.7063 ≈ **-0.71** ✓
- 8B SUM vs POST correlation: r = 0.7992 ≈ **+0.80** ✓
- 70B P(self-claim): mag=-3 → **100%**, mag=+3 → **2%** ✓
- 8B P(self-claim): baseline=5%, mag=-2 → **46%**, mag=+2 → **2%** ✓

---

## Slide 8: Author Attribution

**Status**: COPIED to `data/author_conf_122955.out`

**Original**: `surprise-experiment/llama70b/slurm/author_conf_122955.out` (lines 182-189)

---

## Slide 9: Commitment Point Analysis

**Status**: VERIFIED and COPIED

**Files in `data/prefill_detection/`**:
- `commitment_point_think.out` - Instruct + "think of" prompt: 100% commitment from position 0
- `commitment_point_openended.out` - Instruct + open-ended: commitment at position 9
- `commitment_base.out` - Base model: never commits early (20-50%)

**Script**: `scripts/commitment_point_test.py`, `scripts/commitment_point_base.py`

---

## Slide 10: On-Policy Generation

**Status**: VERIFIED and COPIED

**Files in `data/commitment_tests/`**:
- `format_comparison_bark_Qwen2.5-72B-Instruct.json` - Instruct: 2/5 unique
- `format_comparison_bark_Qwen2.5-72B.json` - Base: 5/5 unique

**Key verification**:
```
Instruct: "The dog's bark echoed through the quiet neighborhood" (2/5 unique)
Base: "tree's bark", "dog's bark", "barked loudly"... (5/5 unique)
```

**Script**: `scripts/format_comparison_bark.py`

---

## Slide 11: Extreme Commitment at Decision Points

**Status**: VERIFIED and COPIED

**Files in `data/commitment_tests/`**:
- `verb_noun_test_Llama-3.1-70B-Instruct.json`
- `verb_noun_test_Qwen2.5-72B-Instruct.json`

**Key verification** (word = "duck", P(verb)/P(noun) ratio):
| Format | Llama 70B | Qwen 72B |
|--------|-----------|----------|
| Alice/Bob | 3.5x | 216x |
| User/Asst | 10.8x | 79x |
| Chat template | **131x** | **2,635x** |

**Script**: `scripts/verb_noun_commitment_test.py`

---

## Slide 12: KV Patching

**Status**: VERIFIED and COPIED

**Files in `data/prefill_detection/`**:
- `kv_swap_historical_130095.json` - Main forward experiment
- `kv_swap_reverse_historical_130220.json` - Reverse experiment (confirms detection happens during generation)
- `kv_swap_finesweep_130256.json` - Fine position sweep (key finding: commitment in first ~10 tokens)

**Key verification**:
- Source prompt (asked for Genghis Khan): NOT PREFILLED (0% detection)
- Prefilled target (no swap): PREFILLED (100% detection)
- Prefilled + KV swap at token 169: NOT PREFILLED (100%)
- **Fine sweep results**:
  - KV patch at token 0: PREFILLED (model even switches to Genghis Khan mid-response!)
  - KV patch at token 10: PREFILLED (continues Marie Curie but detects mismatch)
  - KV patch at token 20+: NOT PREFILLED (model has committed)
- **Reverse experiment** (natural generation + KV swap at analysis): 0% detection
  - This confirms detection happens DURING generation, not at analysis time

**Script**: `scripts/kv_swap_rewind_historical.py`

---

## Slide 13: Goal Stickiness

**Status**: VERIFIED and COPIED

**File**: `data/goal_tests/self_stated_goal_v2_Qwen2.5-72B-Instruct.json`

**Key verification**:
- Condition A (user-stated gravity): STAYED with gravity ("but gravity is different...")
- Condition E (self-stated): SWITCHED TO PHOTONS (abandoned own choice)

**Script**: `scripts/self_stated_goal_test.py`
