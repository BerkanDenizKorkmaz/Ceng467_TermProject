# CENG 467 – Topic 10: LLM-as-a-Judge for Evaluating Generation (llama-3.3-70b-versatile)

## Overview
This project implements an automated evaluation pipeline that uses a Chain-of-Thought (CoT)
prompted LLM to assess machine-generated summaries across three quality dimensions:
**fluency**, **coherence**, and **consistency**. Results are benchmarked against three
baselines and correlated with SummEval expert human ratings using Spearman ρ.

---

## Project Structure
```
llm_judge/
├── main.py            ← Full pipeline (baselines + CoT judge + ablation + error analysis)
├── requirements.txt
├── README.md
└── results/           ← Auto-created; CSV outputs saved here
```

---

## Dataset: SummEval (mteb/summeval)

> **Rearrangement applied** (see Section below)

- Source: https://huggingface.co/datasets/mteb/summeval  
- ~1,600 (article, machine-summary) pairs  
- Human expert ratings on 4 dimensions: coherence, consistency, fluency, relevance  

### Dataset Rearrangement

The raw SummEval dataset stores per-expert scores as **lists** (4 annotators per summary).
We apply two transformations before evaluation:

1. **Average across annotators** — For each (summary, dimension) pair, the four expert
   scores are averaged into a single float. This is the standard protocol from the original
   Fabbri et al. (2021) paper and is required to compute Spearman ρ.

2. **Explode machine_summaries** — Each dataset row contains a *list* of machine summaries
   for one source article. We explode this so every (article, summary) pair is a separate
   row, yielding ~1,600 evaluation instances instead of ~100 articles.

---

## Methods

| # | Method | Description |
|---|--------|-------------|
| B1 | **ROUGE-1/2/L** | Lexical overlap (n-gram F1) |
| B2 | **BERTScore** | Contextual embedding similarity (DeBERTa-XL) |
| B3 | **Naive Zero-Shot LLM** | LLM asked for a single score, no rubric or reasoning |
| ✓ | **CoT LLM Judge** | Multi-step reasoning + structured rubric (fluency/coherence/consistency) |

---

## Running the Code

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with local model (free, CPU)
```bash
python main.py
```

### 3. Quick test with fewer examples
```bash
MAX_EXAMPLES=20 python main.py
```

---

## Outputs (`results/`)

| File | Contents |
|------|----------|
| `all_scores.csv` | All automated scores + human scores per row |
| `spearman_results.csv` | Spearman ρ table (method × human dimension) |

---

## Evaluation Metric

**Spearman ρ** (rank correlation) between automated scores and expert human ratings.
Higher is better; values > 0.4 are considered meaningful agreement in NLP evaluation literature.

---

## Reproducibility

Set `random_state=42` is used for sampling. Full results require ~100 API calls (OpenAI)
or ~30 min on CPU (local model). Prompts are deterministic (temperature=0).

---

## References

- Fabbri et al. (2021). *SummEval: Re-evaluating Summarization Evaluation*. TACL.
- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS.
- Zhang et al. (2020). *BERTScore: Evaluating Text Generation with BERT*. ICLR.