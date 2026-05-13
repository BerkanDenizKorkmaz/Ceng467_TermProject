"""
CENG 467 Term Project - Topic 10: LLM-as-a-Judge for Evaluating Generation
========================================================================
Dataset   : SummEval (mteb/summeval on HuggingFace)
Judge LLM : google/flan-t5-large (local) OR Groq Llama-3-70B (Free API)
Baselines :
  1. Lexical Overlap   — ROUGE-1/2/L
  2. Semantic Similarity — BERTScore (microsoft/deberta-xlarge-mnli)
  3. Naive Direct-Scoring — Zero-Shot LLM (no rubric, no CoT)
Proposed  : CoT LLM Judge with multi-dimensional rubric (fluency, coherence, consistency)
Evaluation: Spearman ρ between automated scores and SummEval human expert ratings
"""

import os, re, json, time, warnings
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")




# ─────────────────────────────────────────────
# CONFIG  (edit here or override via env vars)
# ─────────────────────────────────────────────
# Set to True to use the free Groq API, False to use local CPU/GPU model
USE_FREE_API    = os.getenv("USE_FREE_API", "true").lower() == "true" 

# Load API keys from environment variables (set in .env file or your system environment)
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Warn if API key is missing when using free API
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found. Please check your .env file.")

# Llama 3 70B is an excellent, highly capable reasoning model for judging
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# How many examples to evaluate (set to None for all ~1600)
MAX_EXAMPLES    = int(os.getenv("MAX_EXAMPLES", "100"))

RESULTS_DIR     = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 1.  DATASET LOADING & REARRANGEMENT NOTES
# ══════════════════════════════════════════════════════════

def load_summeval(max_examples=None):
    print("Loading SummEval from HuggingFace …")
    ds = load_dataset("mteb/summeval", split="test")
    rows = []
    for item in ds:
        doc = item["text"]          # source article
        for i, summ in enumerate(item["machine_summaries"]):
            row = {
                "document": doc,
                "summary": summ,
                # average expert scores across annotators
                "human_coherence":   np.mean(item["coherence"][i])   if isinstance(item["coherence"][i], list)   else item["coherence"][i],
                "human_consistency": np.mean(item["consistency"][i]) if isinstance(item["consistency"][i], list) else item["consistency"][i],
                "human_fluency":     np.mean(item["fluency"][i])     if isinstance(item["fluency"][i], list)     else item["fluency"][i],
                "human_relevance":   np.mean(item["relevance"][i])   if isinstance(item["relevance"][i], list)   else item["relevance"][i],
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if max_examples:
        df = df.sample(n=min(max_examples, len(df)), random_state=42).reset_index(drop=True)
    print(f"  → {len(df)} evaluation instances loaded.")
    return df


# ══════════════════════════════════════════════════════════
# 2.  BASELINE 1 — Lexical Overlap (ROUGE)
# ══════════════════════════════════════════════════════════

def compute_rouge(df: pd.DataFrame) -> pd.DataFrame:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ROUGE"):
        scores = scorer.score(row["document"], row["summary"])
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    df = df.copy()
    df["rouge1"], df["rouge2"], df["rougeL"] = r1, r2, rL
    return df


# ══════════════════════════════════════════════════════════
# 3.  BASELINE 2 — Semantic Similarity (BERTScore)
# ══════════════════════════════════════════════════════════

def compute_bertscore(df: pd.DataFrame) -> pd.DataFrame:
    from bert_score import score as bert_score_fn
    print("BERTScore (this may take a few minutes on CPU) …")
    P, R, F = bert_score_fn(
        cands=df["summary"].tolist(),
        refs=df["document"].tolist(),
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=False,
        batch_size=8,
    )
    df = df.copy()
    df["bertscore_p"] = P.numpy()
    df["bertscore_r"] = R.numpy()
    df["bertscore_f"] = F.numpy()
    return df


# ══════════════════════════════════════════════════════════
# 4.  LLM JUDGE HELPERS
# ══════════════════════════════════════════════════════════

def _call_free_api(prompt: str, system: str = "") -> str:
    import groq
    client = groq.Groq(api_key=GROQ_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0)
    return resp.choices[0].message.content


def _call_local_llm(prompt: str, pipe) -> str:
    out = pipe(prompt, max_new_tokens=256, do_sample=False)
    return out[0]["generated_text"] if isinstance(out[0], dict) else str(out[0])


def _extract_score(text: str, lo=1, hi=5) -> float:
    """Parse first integer in [lo, hi] from LLM output."""
    matches = re.findall(r'\b([1-5])\b', text)
    if matches:
        return float(matches[-1])           # last number = most likely the final score
    return 3.0                              # fallback to mid-point


def build_local_pipeline():
    from transformers import pipeline
    print("Loading local LLM (google/flan-t5-large) for judge calls …")
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=-1,                          # CPU; set device=0 for GPU
    )
    return pipe


# ══════════════════════════════════════════════════════════
# 5.  BASELINE 3 — Naive Zero-Shot LLM (no rubric, no CoT)
# ══════════════════════════════════════════════════════════

NAIVE_PROMPT_TEMPLATE = """You are an expert evaluator. Rate the quality of the following summary on a scale from 1 to 5.
Output ONLY a single integer (1-5). Do not explain.

Article:
{document}

Summary:
{summary}

Score (1-5):"""


def compute_naive_llm(df: pd.DataFrame, pipe=None) -> pd.DataFrame:
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Naive LLM"):
        prompt = NAIVE_PROMPT_TEMPLATE.format(
            document=row["document"][:1500],
            summary=row["summary"][:500],
        )
        if USE_FREE_API:
            resp = _call_free_api(prompt)
            time.sleep(2.1) # Groq Free Tier allows 30 req/min (1 every 2 seconds)
        else:
            resp = _call_local_llm(prompt, pipe)
        scores.append(_extract_score(resp))
    df = df.copy()
    df["naive_llm_score"] = scores
    return df


# ══════════════════════════════════════════════════════════
# 6.  PROPOSED — CoT LLM Judge with Multi-Dimensional Rubric
# ══════════════════════════════════════════════════════════

COT_SYSTEM = (
    "You are a rigorous NLP evaluation expert. You reason step by step "
    "before assigning scores. You always follow the rubric exactly."
)

COT_PROMPT_TEMPLATE = """You are evaluating a machine-generated summary of a news article.

First, reason step by step for EACH dimension. Then provide your final scores.

---
ARTICLE (truncated):
{document}

SUMMARY:
{summary}
---

EVALUATION RUBRIC:
1. FLUENCY (1-5): Is the summary grammatically correct, well-formed, and easy to read?
   1=Very poor grammar/disfluent, 5=Perfect grammar and very readable.

2. COHERENCE (1-5): Does the summary have a clear logical structure and flow?
   1=Completely incoherent/disconnected, 5=Highly coherent and well-organized.

3. CONSISTENCY (1-5): Is every fact in the summary supported by the article?
   1=Contains major hallucinations, 5=Entirely factually grounded.

INSTRUCTIONS:
- Think step by step for each dimension.
- After reasoning, output scores in this EXACT format on the last lines:
  FLUENCY_SCORE: <1-5>
  COHERENCE_SCORE: <1-5>
  CONSISTENCY_SCORE: <1-5>

Begin your evaluation:"""


def _parse_cot_scores(text: str):
    dims = {"fluency": 3.0, "coherence": 3.0, "consistency": 3.0}
    patterns = {
        "fluency":     r"FLUENCY_SCORE\s*:\s*([1-5])",
        "coherence":   r"COHERENCE_SCORE\s*:\s*([1-5])",
        "consistency": r"CONSISTENCY_SCORE\s*:\s*([1-5])",
    }
    for dim, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            dims[dim] = float(m.group(1))
    return dims


def compute_cot_judge(df: pd.DataFrame, pipe=None) -> pd.DataFrame:
    fluency, coherence, consistency = [], [], []
    raw_outputs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="CoT Judge"):
        prompt = COT_PROMPT_TEMPLATE.format(
            document=row["document"][:1500],
            summary=row["summary"][:500],
        )
        if USE_FREE_API:
            resp = _call_free_api(prompt, system=COT_SYSTEM)
            time.sleep(2.1) # Groq Free Tier allows 30 req/min (1 every 2 seconds)
        else:
            resp = _call_local_llm(prompt, pipe)
        scores = _parse_cot_scores(resp)
        fluency.append(scores["fluency"])
        coherence.append(scores["coherence"])
        consistency.append(scores["consistency"])
        raw_outputs.append(resp)
    df = df.copy()
    df["cot_fluency"]      = fluency
    df["cot_coherence"]    = coherence
    df["cot_consistency"]  = consistency
    df["cot_raw"]          = raw_outputs
    return df


# ══════════════════════════════════════════════════════════
# 7.  EVALUATION — Spearman ρ
# ══════════════════════════════════════════════════════════

def evaluate_all(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    def spear(a, b):
        rho, p = spearmanr(a, b)
        return round(rho, 4), round(p, 4)

    # ── ROUGE vs human dimensions ──────────────────────────
    for rouge_col in ["rouge1", "rouge2", "rougeL"]:
        if rouge_col not in df.columns:
            continue
        for human_col in ["human_coherence", "human_consistency", "human_fluency", "human_relevance"]:
            rho, p = spear(df[rouge_col], df[human_col])
            results.append({"Method": rouge_col.upper(), "Human Dimension": human_col.replace("human_",""), "Spearman ρ": rho, "p-value": p})

    # ── BERTScore vs human dimensions ─────────────────────
    if "bertscore_f" in df.columns:
        for human_col in ["human_coherence", "human_consistency", "human_fluency", "human_relevance"]:
            rho, p = spear(df["bertscore_f"], df[human_col])
            results.append({"Method": "BERTScore-F", "Human Dimension": human_col.replace("human_",""), "Spearman ρ": rho, "p-value": p})

    # ── Naive LLM vs overall human mean ───────────────────
    if "naive_llm_score" in df.columns:
        df["human_mean"] = df[["human_coherence","human_consistency","human_fluency","human_relevance"]].mean(axis=1)
        rho, p = spear(df["naive_llm_score"], df["human_mean"])
        results.append({"Method": "Naive Zero-Shot LLM", "Human Dimension": "mean", "Spearman ρ": rho, "p-value": p})

    # ── CoT Judge vs matching human dimensions ─────────────
    cot_map = {
        "cot_fluency":     "human_fluency",
        "cot_coherence":   "human_coherence",
        "cot_consistency": "human_consistency",
    }
    for cot_col, human_col in cot_map.items():
        if cot_col in df.columns:
            rho, p = spear(df[cot_col], df[human_col])
            results.append({"Method": f"CoT Judge ({cot_col.replace('cot_','')})", "Human Dimension": human_col.replace("human_",""), "Spearman ρ": rho, "p-value": p})

    result_df = pd.DataFrame(results)
    return result_df


# ══════════════════════════════════════════════════════════
# 8.  ABLATION STUDY — CoT vs Score-Only prompt
# ══════════════════════════════════════════════════════════

SCORE_ONLY_TEMPLATE = """Rate this summary for fluency, coherence, and consistency (1-5 each).
Output ONLY three lines:
FLUENCY_SCORE: X
COHERENCE_SCORE: X
CONSISTENCY_SCORE: X

Article: {document}
Summary: {summary}"""


def compute_scoreonly_judge(df: pd.DataFrame, pipe=None) -> pd.DataFrame:
    """Ablation: same rubric dimensions but no CoT reasoning step."""
    fluency, coherence, consistency = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Score-Only (Ablation)"):
        prompt = SCORE_ONLY_TEMPLATE.format(
            document=row["document"][:1500],
            summary=row["summary"][:500],
        )
        if USE_FREE_API:
            resp = _call_free_api(prompt)
            time.sleep(2.1) # Groq Free Tier allows 30 req/min (1 every 2 seconds)
        else:
            resp = _call_local_llm(prompt, pipe)
        scores = _parse_cot_scores(resp)
        fluency.append(scores["fluency"])
        coherence.append(scores["coherence"])
        consistency.append(scores["consistency"])
    df = df.copy()
    df["ablation_fluency"]     = fluency
    df["ablation_coherence"]   = coherence
    df["ablation_consistency"] = consistency
    return df


# ══════════════════════════════════════════════════════════
# 9.  ERROR ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════

def error_analysis(df: pd.DataFrame, n=10) -> None:
    """Print cases where CoT judge deviates most from human scores."""
    if "cot_coherence" not in df.columns:
        return
    df = df.copy()
    df["cot_error"] = abs(df["cot_coherence"] - df["human_coherence"])
    worst = df.nlargest(n, "cot_error")[["summary","human_coherence","cot_coherence","cot_error","cot_raw"]]
    print("\n── Top error cases (CoT coherence vs human) ──")
    for i, (_, row) in enumerate(worst.iterrows()):
        print(f"\n[{i+1}] Summary  : {row['summary'][:200]} …")
        print(f"      Human ρ  : {row['human_coherence']} | CoT : {row['cot_coherence']} | Error : {row['cot_error']:.2f}")
        print(f"      Raw CoT  : {str(row['cot_raw'])[:300]} …")


# ══════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  CENG 467 – LLM-as-a-Judge Pipeline")
    print("=" * 65)

    # ── Load dataset ───────────────────────────────────────
    df = load_summeval(max_examples=MAX_EXAMPLES)

    # ── Local LLM (used only if not Free API) ───────────────
    pipe = None
    if not USE_FREE_API:
        pipe = build_local_pipeline()

    # ── Baseline 1 : ROUGE ────────────────────────────────
    df = compute_rouge(df)
    print("✓ ROUGE scores computed.")

    # ── Baseline 2 : BERTScore ────────────────────────────
    df = compute_bertscore(df)
    print("✓ BERTScore computed.")

    # ── Baseline 3 : Naive LLM ────────────────────────────
    df = compute_naive_llm(df, pipe=pipe)
    print("✓ Naive Zero-Shot LLM scores computed.")

    # ── Proposed : CoT Judge ──────────────────────────────
    df = compute_cot_judge(df, pipe=pipe)
    print("✓ CoT Judge scores computed.")

    # ── Ablation : Score-Only (no CoT) ────────────────────
    df = compute_scoreonly_judge(df, pipe=pipe)
    print("✓ Ablation (score-only) scores computed.")

    # ── Save full results ─────────────────────────────────
    df.drop(columns=["cot_raw"], errors="ignore").to_csv(
        f"{RESULTS_DIR}/all_scores.csv", index=False
    )

    # ── Spearman evaluation ───────────────────────────────
    metrics_df = evaluate_all(df)
    metrics_df.to_csv(f"{RESULTS_DIR}/spearman_results.csv", index=False)

    print("\n" + "=" * 65)
    print("  Spearman ρ Results")
    print("=" * 65)
    print(metrics_df.to_string(index=False))

    # ── Ablation comparison ───────────────────────────────
    from scipy.stats import spearmanr
    print("\n── Ablation Study: CoT vs Score-Only ──")
    for dim in ["fluency", "coherence", "consistency"]:
        cot_col    = f"cot_{dim}"
        abl_col    = f"ablation_{dim}"
        human_col  = f"human_{dim}"
        if cot_col in df.columns and abl_col in df.columns:
            cot_rho,  _ = spearmanr(df[cot_col],  df[human_col])
            abl_rho,  _ = spearmanr(df[abl_col],  df[human_col])
            delta = cot_rho - abl_rho
            print(f"  {dim:12s}  CoT ρ={cot_rho:.4f}  Score-Only ρ={abl_rho:.4f}  Δ={delta:+.4f}")

    # ── Error analysis ────────────────────────────────────
    error_analysis(df, n=5)

    print(f"\nAll results saved to '{RESULTS_DIR}/' directory.")


if __name__ == "__main__":
    main()