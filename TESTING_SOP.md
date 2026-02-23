# Testing SOP + RAG Architecture Commentary

---

## Part A: What's Already Implemented (Code Complete)

All code is written and ready. Nothing needs to be coded — you just need to **run and evaluate**. Here's the full picture of what each file does:

### Pipeline Files (you run these)

| File | What it does | How you run it |
|------|-------------|----------------|
| `ask.py` | CLI entry point. Single-query mode OR report mode (`--report`). | `python ask.py --parser marker --report --q "..." --verbose --save` |
| `src/rag/report_pipeline.py` | Report engine: decomposes prompt → retrieves per sub-question → deduplicates → generates report. Called internally by `ask.py --report`. | You don't run this directly. |
| `src/rag/pipeline_e2e.py` | Single-query engine: retrieve k=15 → rerank top-5 → generate answer. Called internally by `ask.py` (no `--report`). | You don't run this directly. |
| `src/llm/load_generator_llm.py` | Loads Gemini 2.0 Flash. `max_tokens=2048` for single-query, `8192` for report. | Automatic based on `--report` flag. |

### Evaluation Files (you run these)

| File | What it does | How you run it |
|------|-------------|----------------|
| `src/eval/evaluate.py` | Runs all 10 golden set prompts through report pipeline, judges each, saves results. With `--baseline`, also runs the same prompts through single-query pipeline for comparison. | `python -m src.eval.evaluate --parser marker --baseline` |
| `src/eval/judge.py` | Contains two judges: `judge_answer()` (4 dims, for single-query) and `judge_report()` (6 dims, for reports). | Called automatically by `evaluate.py`. |
| `src/eval/retrieval_metrics.py` | Computes Recall@k, MRR, NDCG by comparing retrieved pages against `relevant_pages`. | Called automatically by `evaluate.py`. |
| `src/eval/golden_set.json` | 10 report prompts (R01-R10) with `expected_sections`, `must_include_facts`, and `relevant_pages` (currently empty — you fill these on Day 2). | You edit this manually. |
| `postprocess.py` | Aggregates evaluation JSON into summary stats (means, medians, coverage rates). | `python postprocess.py --eval-results eval_results/marker_report_evaluation.json --output stats.json` |

### What `--baseline` Does (Important)

When you run `python -m src.eval.evaluate --parser marker --baseline`, for **each** of the 10 prompts the system does:

1. **Report pipeline**: Decompose → multi-retrieve (k=15 per sub-question) → rerank (top-5 per sub-question) → deduplicate → merge (~20 chunks) → generate report (8192 tokens) → judge with `judge_report()` (6 dimensions)
2. **Baseline (single-query)**: Send the same prompt as-is → retrieve k=15 → rerank top-5 → generate answer (2048 tokens) → judge with `judge_answer()` (4 dimensions)

The output JSON has **both** results side-by-side for each prompt. This is your comparison data — you do NOT need Run 4 or any prior experiments. The baseline runs on the exact same prompts, making the comparison fair and self-contained.

### CLI Flags Reference

```
python ask.py --parser marker --report --q "..."   # Basic report generation
                              --k 20               # Chunks retrieved per sub-question (default: 15)
                              --top-n 7            # Kept after reranking per sub-question (default: 5)
                              --max-chunks 25      # Max unique chunks after dedup (default: 20)
                              --verbose            # Show sub-questions, retrieval metadata, chunk details
                              --save               # Save JSON to query_outputs/
```

---

## Part B: Your RAG Type — What It Is and Why It Matters

### Classification

Your system is a **Two-Stage Retrieval-Augmented Generation (RAG) pipeline with Query Decomposition for Report Synthesis**. In the RAG taxonomy literature (Gao et al. 2024, "Retrieval-Augmented Generation for Large Language Models: A Survey"), this falls under **Advanced RAG**:

| Component | Your Choice | Category |
|-----------|-------------|----------|
| Retrieval | Dense (MiniLM-L12-v2) + Cross-Encoder Reranking (BGE) | Two-stage retrieval |
| Chunking | Markdown-header-aware + token-budget | Hierarchical chunking |
| Generation | Gemini 2.0 Flash with citation enforcement | Constrained generation |
| Report extension | Query decomposition → multi-retrieval → synthesis | Compositional RAG |
| Evaluation | LLM-as-a-judge (6 dimensions) + IR metrics | Hybrid evaluation |

### What Makes This Publishable

**Strengths for a workshop paper:**

1. **Domain-specific RAG with math preservation** — OpenFOAM documentation contains LaTeX equations, code blocks, and nested dictionary syntax. Your parser comparison (Marker vs Docling vs PyMuPDF) with `redo_inline_math=True` is a concrete contribution. Most RAG papers test on Wikipedia or generic QA.

2. **Two-stage retrieval is well-evidenced** — Dense retrieval for recall (k=15) followed by cross-encoder reranking for precision (top-5) is current best practice (Nogueira et al., "Passage Re-ranking with BERT"). Your system implements this cleanly.

3. **Query decomposition for report generation** — This is the novel angle for RAG4Report. Single-query RAG retrieves from one region of the embedding space and can't cover multiple topics. Decomposing into 3-5 sub-questions and deduplicating solves this. The `--baseline` comparison makes it empirically testable.

4. **Checklist-based evaluation** — The `expected_sections` + `must_include_facts` approach with programmatic weighted scoring is more rigorous than pure LLM-as-a-judge. Coverage and Factual Recall are measurable dimensions that reviewers can verify.

**Limitations to acknowledge in the paper:**

1. **Small embedding model** — MiniLM-L12-v2 (384-dim) is lightweight. Note this was a deliberate trade-off (runs on CPU, fast indexing). Larger models (E5-large, BGE-large) could improve retrieval. Fine as a discussion point.

2. **Single-document corpus** — The OpenFOAM User Guide is one large PDF. Frame as "focused domain evaluation," appropriate for a workshop.

3. **LLM-as-a-judge reliability** — Gemini 2.5 Flash at temp=0.0. LLM judges have known biases. The programmatic `overall_score` (weighted average, not LLM-computed) and checklist-based dimensions partially mitigate this.

4. **No human evaluation** — Manually scoring even 3-4 reports and comparing with LLM judge scores would strengthen the paper.

### Paper Narrative

"Single-query RAG fails for report-level tasks; query decomposition with deduplication solves this; here's how to evaluate it with checklist-based judging."

---

## Part C: Day-by-Day Testing Checklist

### Prerequisites

Before Day 1, ensure:
- [ ] Vector DB exists: `db/marker_db/` directory present. If not, run: `python main.py --parser marker --embed`
- [ ] `GOOGLE_API_KEY` is set in your environment
- [ ] Dependencies installed: `pip install -r requirements.txt`

---

### Day 1: Smoke Test — Does the Report Pipeline Work?

**Goal:** Verify end-to-end on a single prompt.

#### Step 1.1: Generate one report (default settings)
```bash
python ask.py --parser marker --report --q "Write a technical overview of the discretization framework in OpenFOAM, covering spatial and temporal schemes, the fvSchemes dictionary structure, and how users configure gradient, divergence, Laplacian, interpolation, and surface normal gradient schemes." --verbose --save
```

**Check in terminal:**
- [ ] 3-5 sub-questions printed
- [ ] Retrieval metadata: `total_before_dedup`, `total_after_dedup`, `final_chunks_used`
- [ ] Report has `##` section headings
- [ ] Report has `[n]` inline citations
- [ ] Report ends with a `## References` section

**Check in saved JSON (`query_outputs/query_TIMESTAMP.json`):**
- [ ] `"mode": "report"` present
- [ ] `"sub_questions"` array has 3-5 entries
- [ ] `"chunks"` array has entries with both `rerank_score` and `similarity_score`
- [ ] `"retrieval_metadata"` has the three expected fields

#### Step 1.2: Try higher k values
```bash
python ask.py --parser marker --report --q "Write a technical overview of the discretization framework in OpenFOAM, covering spatial and temporal schemes, the fvSchemes dictionary structure, and how users configure gradient, divergence, Laplacian, interpolation, and surface normal gradient schemes." --k 20 --top-n 7 --max-chunks 25 --verbose --save
```

**Compare with 1.1:**
- [ ] `final_chunks_used` increased?
- [ ] `total_after_dedup` < `total_before_dedup`? (dedup is working)
- [ ] Report noticeably more detailed?

#### Step 1.3: Regression check — single-query mode still works
```bash
python ask.py --parser marker --q "Explain fvSchemes" --verbose
```
- [ ] Normal answer generated (not a report)

**Decision:** Based on 1.1 vs 1.2, pick your k values for the rest of testing. Default (k=15, top-n=5, max-chunks=20) is fine unless you see clear improvement with higher values.

---

### Day 2: Full Evaluation — All 10 Prompts

**Goal:** Generate + judge all 10 golden set reports, then populate ground truth pages.

#### Step 2.1: Run evaluation (report pipeline only)
```bash
python -m src.eval.evaluate --parser marker
```

**Output file:** `eval_results/marker_report_evaluation.json`

**Check:**
- [ ] File contains 10 entries (R01-R10)
- [ ] Each entry has `"report"` → `"evaluation"` with 6 scores
- [ ] Each entry has `"sub_questions"` (3-5 per prompt)
- [ ] No `"error"` keys in any evaluation

#### Step 2.2: Scan report quality

Open the evaluation JSON. For each R01-R10:
- [ ] Report is not truncated (ends properly with References)
- [ ] Has multiple `##` sections (not a wall of text)
- [ ] Has `[n]` citations throughout
- [ ] Reasonable length (~1500-2500 words)

**Note down:**
- Best/worst reports by quality
- Any judge scores that seem wrong (e.g., score 5 for a clearly weak report)

#### Step 2.3: Populate `relevant_pages` in golden_set.json (MANUAL STEP)

For each R01-R10:
1. Look at the `"chunks"` → `"page"` values in the evaluation JSON
2. Open the OpenFOAM PDF, verify those pages are actually relevant
3. Check if the retriever missed any obviously relevant pages
4. Update `src/eval/golden_set.json` — fill the `"relevant_pages": []` arrays

**Budget ~1-2 hours.** This enables IR metrics (Recall@k, MRR, NDCG) on Day 3.

---

### Day 3: Baseline Comparison + Full Metrics

**Goal:** Get side-by-side numbers (report vs single-query) and aggregate everything.

#### Step 3.1: Run evaluation WITH baseline
```bash
python -m src.eval.evaluate --parser marker --baseline
```

This overwrites the previous file. Each entry now has both `"report"` and `"baseline"`.

**Check:**
- [ ] Each entry has both `"report"` and `"baseline"` keys
- [ ] Report entries: 6-dimension scores (groundedness, accuracy, citation, coverage, factual_recall, structure)
- [ ] Baseline entries: 4-dimension scores (groundedness, accuracy, citation, completeness)
- [ ] `retrieval_metrics` populated for entries where you filled `relevant_pages`

#### Step 3.2: Aggregate all metrics
```bash
python postprocess.py --eval-results eval_results/marker_report_evaluation.json --output eval_results/report_stats.json
```

**Check printed summary for:**
- [ ] Overall score mean/median
- [ ] Per-dimension means
- [ ] Section coverage: X/Y (rate)
- [ ] Fact coverage: X/Y (rate)
- [ ] IR metrics (if `relevant_pages` were populated)

#### Step 3.3: Record key findings

Fill in this table from the results:

| Metric | Report Pipeline | Baseline (single-query) |
|--------|----------------|------------------------|
| Overall Score | ? | ? |
| Groundedness | ? | ? |
| Technical Accuracy | ? | ? |
| Citation Correctness | ? | ? |
| Coverage | ? | N/A |
| Factual Recall | ? | N/A |
| Section coverage (found/expected) | ?/? | N/A |
| Fact coverage (found/expected) | ?/? | N/A |
| Chunks used per query | ~20 | 5 |

---

### Day 4: Iterate on Weak Spots

**Goal:** Diagnose and improve the weakest prompts.

#### Step 4.1: Find bottom 3 prompts by `overall_score`

Open `eval_results/marker_report_evaluation.json` and sort entries by `report.evaluation.overall_score`.

Common failure modes:
- **Low coverage** → retriever missed sections → try `--k 20`
- **Low factual recall** → key facts filtered out during reranking → check if facts appear in top-15 but not top-5
- **Low citation correctness** → generator hallucinating citation numbers → check chunk content
- **Low structure** → wall of text → prompt tuning needed

#### Step 4.2: Re-run weak prompts individually
```bash
python ask.py --parser marker --report --q "<weak prompt text here>" --k 20 --top-n 7 --max-chunks 25 --verbose --save
```

**Examine:** Are the right chunks being retrieved? Is the generator ignoring relevant chunks?

#### Step 4.3: Adjust golden_set.json if needed

If some `must_include_facts` are too specific or not actually in the PDF, revise them. The checklist should be fair — only test facts that ARE in the documentation.

---

### Day 5: Final Run + Paper-Ready Numbers

**Goal:** Clean final results.

#### Step 5.1: Final evaluation run (with baseline)
```bash
python -m src.eval.evaluate --parser marker --baseline
```

#### Step 5.2: Final aggregation
```bash
python postprocess.py --eval-results eval_results/marker_report_evaluation.json --output eval_results/final_report_stats.json
```

#### Step 5.3: Manual spot-check (recommended)

Pick 2-3 reports. Read them yourself and score on the 6 dimensions (1-5 each). Compare with LLM judge scores. Note agreement/disagreement. This gives you a "human evaluation" data point for the paper.

---

### Day 6: Buffer + Write-Up

Reserved for:
- Finalizing the paper/presentation
- Re-running any failed evaluations
- Creating figures/tables from the JSON stats

---

## Part D: Output File Map

| What | Path | When Created |
|------|------|-------------|
| Single report (smoke test) | `query_outputs/query_YYYYMMDD_HHMMSS.json` | Day 1 |
| Full evaluation (10 prompts + optional baseline) | `eval_results/marker_report_evaluation.json` | Day 2 / Day 3 |
| Aggregated stats | `eval_results/report_stats.json` | Day 3 |
| Final stats | `eval_results/final_report_stats.json` | Day 5 |
| Golden set with your checklists | `src/eval/golden_set.json` | Exists, updated Day 2 |

## Part E: Verification Milestones

**After Day 1:** Report pipeline works. At least 2 saved JSONs in `query_outputs/`. You've confirmed sub-questions, dedup, citations, and sections all appear.

**After Day 3:** 10 reports + 10 baselines judged. Aggregated metrics in hand. You can already draft the results section. The key comparison: report pipeline has coverage/factual_recall scores while baseline doesn't even measure those (it only has completeness).

**After Day 5:** Clean final numbers. Optional human evaluation data. Ready to write the paper.
