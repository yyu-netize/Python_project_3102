import json
import re
import random
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from multi_turn_chat import MultiTurnRAGChat, TurnRecord

from generator import UltimateRAGWithGenerator, client, MODEL_NAME

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common_tokens = pred_counter & gold_counter
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def load_qa_dataset(path: str = "qa_dataset_clean.json") -> List[Dict[str, Any]]:

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found at: {path}")
        raise
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON at: {path}")
        raise

    if not isinstance(data, list):
        raise ValueError(f"Dataset at {path} should be a list of QA items.")

    print(f"[INFO] Loaded {len(data)} QA items from {path}")
    return data


@dataclass
class RAGHandles:
    rag_gen: UltimateRAGWithGenerator   
    rag_core: Any                       

def build_rag_handles() -> RAGHandles:
    print("[INFO] Initializing RAG (this may take some time the first run)...")
    rag_gen = UltimateRAGWithGenerator()
    rag_core = rag_gen.rag
    print("[INFO] RAG initialized.")
    return RAGHandles(rag_gen=rag_gen, rag_core=rag_core)

def get_retrieved_ids_for_query(
    rag_core: Any,
    query: str,
    retrieve_mode: str = "hybrid",
    top_k: int = 10,
    use_rerank: bool = False,
) -> List[str]:

    # 1) base on generator functions
    if retrieve_mode == "dense":
        candidates = rag_core.retrieve_dense(query, top_k=top_k)
    elif retrieve_mode in ("bm25", "sparse"):
        candidates = rag_core.retrieve_bm25(query, top_k=top_k)
    elif retrieve_mode == "hybrid":
        candidates = rag_core.retrieve_hybrid(query, top_k=top_k)
    elif retrieve_mode == "hyde":
        pseudo_doc = rag_core.hyde_generate_doc(query)
        candidates = rag_core.retrieve_dense(pseudo_doc, top_k=top_k)
    else:
        raise ValueError(f"Unknown retrieve_mode: {retrieve_mode}")

    if not use_rerank:
        return [c["id"] for c in candidates[:top_k]]

    reranked = rag_core.rerank(query, candidates, top_n=top_k)
    return [c["id"] for c in reranked[:top_k]]


def eval_retrieval_metrics(
    qa_path: str = "qa_dataset_clean.json",
    retrieve_mode: str = "hybrid",
    k_list: List[int] = [1, 3, 5, 10],
    sample_size: Optional[int] = 100,
    random_seed: int = 42,
    use_rerank: bool = False,
) -> Dict[str, float]:

    qa_items = load_qa_dataset(qa_path)
    if sample_size is not None and 0 < sample_size < len(qa_items):
        random.seed(random_seed)
        qa_items = random.sample(qa_items, sample_size)
        print(f"[INFO] Sampled {len(qa_items)} QA items for retrieval evaluation.")
    else:
        print(f"[INFO] Using all {len(qa_items)} QA items for retrieval evaluation.")

    handles = build_rag_handles()
    rag_core = handles.rag_core

    k_list = sorted(k_list)
    max_k = max(k_list)

    hit_counts = {k: 0 for k in k_list}
    mrr_sum = 0.0
    total = 0

    for idx, item in enumerate(qa_items, start=1):
        q = item["question"]
        gt_id = item.get("chunk_id")
        if not gt_id:
            continue

        print(f"[Retrieval Eval {idx}/{len(qa_items)}] QID={item.get('id', idx)}")
        print("Question:", q)
        print("GT chunk_id:", gt_id)

        doc_ids = get_retrieved_ids_for_query(
            rag_core=rag_core,
            query=q,
            retrieve_mode=retrieve_mode,
            top_k=max_k,
            use_rerank=use_rerank,
        )

        total += 1

        for k in k_list:
            if gt_id in doc_ids[:k]:
                hit_counts[k] += 1

        rr = 0.0
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id == gt_id:
                rr = 1.0 / rank
                break
        mrr_sum += rr

    if total == 0:
        print("[WARN] No valid QA items for retrieval evaluation.")
        return {}

    metrics = {}
    for k in k_list:
        metrics[f"recall@{k}"] = hit_counts[k] / total
    metrics["mrr"] = mrr_sum / total

    print("\n[RESULT] Retrieval metrics "
          f"(mode={retrieve_mode}, rerank={use_rerank}, total={total})")
    for k in k_list:
        print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print(f"  MRR: {metrics['mrr']:.4f}")

    tag_rerank = "rerank_on" if use_rerank else "rerank_off"
    out_path = f"retrieval_metrics_{retrieve_mode}_{tag_rerank}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "retrieve_mode": retrieve_mode,
                "use_rerank": use_rerank,
                "total_eval": total,
                "k_list": k_list,
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved retrieval metrics to {out_path}")
    return metrics

def run_retrieval_ablation(
    qa_path: str = "qa_dataset_clean.json",
    sample_size: int = 100,
    k_list: List[int] = [1, 3, 5, 10],
    random_seed: int = 42,
) -> str:

    modes = ["bm25", "dense", "hybrid", "hyde"]
    rerank_options = [False, True]

    summary: List[Dict[str, Any]] = []

    print("\n========== Retrieval Ablation (mode × rerank) ==========")
    print(f"QA Path: {qa_path}")
    print(f"Sample size: {sample_size}, k_list = {k_list}\n")

    for mode in modes:
        for rerank in rerank_options:
            print(f"\n----- Running eval: mode={mode}, rerank={rerank} -----")
            metrics = eval_retrieval_metrics(
                qa_path=qa_path,
                retrieve_mode=mode,
                k_list=k_list,
                sample_size=sample_size,
                random_seed=random_seed,
                use_rerank=rerank,
            )
            if not metrics:
                continue

            row = {
                "mode": mode,
                "rerank": rerank,
                **metrics,  
            }
            summary.append(row)

    out_path = "retrieval_ablation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved retrieval ablation summary to {out_path}")
    print("Summary rows:")
    for row in summary:
        print(
            f"  mode={row['mode']:6s}, rerank={row['rerank']}: "
            + ", ".join(
                f"{k}={row[k]:.4f}"
                for k in row.keys()
                if k.startswith("recall@") or k == "mrr"
            )
        )

    return out_path

def eval_answer_quality_single_turn(
    qa_path: str = "qa_dataset_clean.json",
    enable_rewrite: bool = False,
    retrieve_mode: str = "hyde",
    prompt_mode: str = "instruction",
    message_mode: str = "with_system",
    sample_size: Optional[int] = 50,
    random_seed: int = 42,
) -> Tuple[float, float, str]:

    qa_items = load_qa_dataset(qa_path)
    if sample_size is not None and 0 < sample_size < len(qa_items):
        random.seed(random_seed)
        qa_items = random.sample(qa_items, sample_size)
        print(f"[INFO] Sampled {len(qa_items)} QA items for answer evaluation.")
    else:
        print(f"[INFO] Using all {len(qa_items)} QA items for answer evaluation.")

    chat = MultiTurnRAGChat(
        enable_rewrite=enable_rewrite,
        default_retrieve_mode=retrieve_mode,
        default_prompt_mode=prompt_mode,
        default_message_mode=message_mode,
    )

    em_scores: List[float] = []
    f1_scores: List[float] = []
    results: List[Dict[str, Any]] = []

    rewrite_tag = "rewrite_on" if enable_rewrite else "rewrite_off"

    for idx, item in enumerate(qa_items, start=1):
        q = item["question"]
        gold = item["answer"]
        qid = item.get("id", f"idx_{idx}")

        print(f"\n[Answer Eval {idx}/{len(qa_items)}] QID={qid}")
        print("Question:", q)
        print("Gold:", gold)

        chat.reset()
        pred = chat.ask(q)
        print("Pred:", pred)

        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        em_scores.append(em)
        f1_scores.append(f1)
        history = chat.get_history()
        last_turn: Optional[TurnRecord] = history[-1] if history else None

        results.append(
            {
                "id": qid,
                "question": q,
                "gold_answer": gold,
                "pred_answer": pred,
                "em": em,
                "f1": f1,
                "chunk_id": item.get("chunk_id", ""),
                "source_title": item.get("source_title", ""),
                "enable_rewrite": enable_rewrite,
                "retrieve_mode": retrieve_mode,
                "prompt_mode": prompt_mode,
                "message_mode": message_mode,
                "rewritten_query": last_turn.rewritten_query if last_turn else q,
            }
        )

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print(f"\n[RESULT] Single-turn answer eval "
          f"(rewrite={rewrite_tag}, mode={retrieve_mode}) - "
          f"Avg EM = {avg_em:.4f}, Avg F1 = {avg_f1:.4f}")

    out_path = f"answer_eval_single_turn_{retrieve_mode}_{rewrite_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "qa_path": qa_path,
                "enable_rewrite": enable_rewrite,
                "retrieve_mode": retrieve_mode,
                "prompt_mode": prompt_mode,
                "message_mode": message_mode,
                "avg_em": avg_em,
                "avg_f1": avg_f1,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Saved answer eval results to {out_path}")
    return avg_em, avg_f1, out_path


def llm_judge_answers(
    eval_result_path: str,
    sample_size: int = 30,
    random_seed: int = 123,
    output_path: Optional[str] = None,
) -> str:

    with open(eval_result_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    records = eval_data["results"]
    print(f"[INFO] Loaded {len(records)} answer records from {eval_result_path}")

    # 2. Sample subset if needed
    if sample_size is not None and 0 < sample_size < len(records):
        random.seed(random_seed)
        records = random.sample(records, sample_size)
        print(f"[INFO] Sampled {len(records)} records for LLM-as-Judge.")
    else:
        print(f"[INFO] Using all {len(records)} records for LLM-as-Judge.")

    judged_results = []

    # ---- English system prompt, JSON-only, short comments ----
    system_prompt = (
        "You are an expert evaluator for a Plants vs. Zombies (PvZ) "
        "Retrieval-Augmented Generation (RAG) system.\n"
        "Given a question, the gold reference answer, and the model's predicted answer, "
        "your task is to rate the model answer on three dimensions:\n"
        "1. faithfulness — whether the answer is factually consistent with the reference (no hallucination),\n"
        "2. relevance — whether the answer directly addresses the question,\n"
        "3. overall — your holistic judgment (not necessarily an average).\n\n"
        "Scoring scale: 0.0 to 1.0\n\n"
        "STRICT OUTPUT RULES:\n"
        "- Output ONLY a JSON object.\n"
        "- The JSON must contain: faithfulness, relevance, overall, comment.\n"
        "- The 'comment' must be concise (maximum 20 words).\n"
        "- Do NOT include explanations outside the JSON.\n"
        "Example:\n"
        "{\n"
        "  \"faithfulness\": 0.90,\n"
        "  \"relevance\": 0.95,\n"
        "  \"overall\": 0.92,\n"
        "  \"comment\": \"Accurate and focused answer.\"\n"
        "}"
    )

    for idx, rec in enumerate(records, start=1):
        gold = rec.get("gold_answer", rec.get("gold", ""))
        pred = rec.get("pred_answer", rec.get("pred", ""))
        q = rec.get("question", "")

        user_prompt = (
            f"Question:\n{q}\n\n"
            f"Gold reference answer:\n{gold}\n\n"
            f"Model's predicted answer:\n{pred}\n\n"
            "Evaluate the answer. Output only the JSON object. "
            "The comment must be brief (<= 20 words)."
        )

        print(f"\n[LLM-Judge {idx}/{len(records)}] QID={rec.get('id')}")
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                top_p=0.9,
                n=1,
            )
            raw = resp.choices[0].message.content.strip()

            # Try to parse JSON; if it fails, wrap raw text as comment
            try:
                judge_obj = json.loads(raw)
            except json.JSONDecodeError:
                judge_obj = {
                    "faithfulness": None,
                    "relevance": None,
                    "overall": None,
                    "comment": raw,
                }

        except Exception as e:
            print(f"[WARN] LLM judge failed for this sample: {e}")
            judge_obj = {
                "faithfulness": None,
                "relevance": None,
                "overall": None,
                "comment": f"LLM judge failed: {e}",
            }

        judged_rec = dict(rec)
        judged_rec["judge"] = judge_obj
        judged_results.append(judged_rec)

    if output_path is None:
        base = eval_result_path.rsplit(".", 1)[0]
        output_path = base + "_judged.json"

    out_data = dict(eval_data)
    out_data["judged_results"] = judged_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved LLM-as-Judge results to {output_path}")
    return output_path


def sanity_check_chunk_ids(qa_path="qa_dataset_clean.json"):
    qa_items = load_qa_dataset(qa_path)
    handles = build_rag_handles()
    rag_core = handles.rag_core

    vec_ids = set(rag_core.collection.get()["ids"])
    bm25_ids = set(chunk["id"] for chunk in rag_core.bm25_chunks)

    all_ids = vec_ids | bm25_ids
    print(f"[CHECK] total unique doc ids in index: {len(all_ids)}")

    missing = []
    for item in qa_items:
        cid = item.get("chunk_id")
        if cid and cid not in all_ids:
            missing.append(cid)

    missing = sorted(set(missing))
    print(f"[CHECK] QA items with chunk_id not in index: {len(missing)}")
    if missing:
        print("  Examples:", missing[:10])




from rouge import Rouge

rouge_scorer = Rouge()


def evaluate_single_prediction(question: str, gold: str, pred: str):
    em = exact_match(pred, gold)
    f1 = f1_score(pred, gold)

    try:
        rouge_scores = rouge_scorer.get_scores(pred, gold)
        rouge_l = rouge_scores[0]["rouge-l"]["f"]
    except Exception:
        rouge_l = 0.0

    return em, f1, rouge_l

def eval_answer_quality_single_turn(
    qa_path: str = "qa_dataset_clean.json",
    retrieve_mode: str = "hybrid",
    use_rerank: bool = True,      
    enable_rewrite: bool = False,
    sample_size: int = 100,
    random_seed: int = 42,
) -> str:
    import random
    import json

    print("\n=============================")
    print("  Answer Quality Evaluation")
    print("=============================")
    print(f"Mode={retrieve_mode}, Rerank={use_rerank}, Rewrite={enable_rewrite}")
    print("=============================\n")

    qa_items = load_qa_dataset(qa_path)
    if 0 < sample_size < len(qa_items):
        random.seed(random_seed)
        qa_items = random.sample(qa_items, sample_size)
        print(f"[INFO] Sampled {len(qa_items)} QA items for answer evaluation.")
    else:
        print(f"[INFO] Using all {len(qa_items)} QA items for answer evaluation.")

    results = []
    total_em, total_f1, total_rouge = 0.0, 0.0, 0.0

    chat = MultiTurnRAGChat(
        enable_rewrite=enable_rewrite,
        default_retrieve_mode=retrieve_mode,
        default_rerank_mode=use_rerank,
        default_prompt_mode="instruction",
        default_message_mode="with_system",
    )

    for idx, item in enumerate(qa_items, 1):
        q = item["question"]
        gold = item["answer"]

        chat.reset()  

        pred = chat.ask(q)  

        em, f1, rouge_l = evaluate_single_prediction(q, gold, pred)

        total_em += em
        total_f1 += f1
        total_rouge += rouge_l

        results.append({
            "id": item.get("id", f"idx_{idx}"),
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "rouge_l": rouge_l,
            "chunk_id": item.get("chunk_id", ""),
            "source_title": item.get("source_title", ""),
            "retrieve_mode": retrieve_mode,
            "use_rerank": use_rerank,
            "enable_rewrite": enable_rewrite,
        })

        print(f"[{idx}/{len(qa_items)}] EM={em:.2f}, F1={f1:.2f}, ROUGE-L={rouge_l:.2f}")

    n = len(qa_items)
    avg_em = total_em / n if n > 0 else 0.0
    avg_f1 = total_f1 / n if n > 0 else 0.0
    avg_rouge = total_rouge / n if n > 0 else 0.0

    print("\n========== Summary ==========")
    print(f"Avg EM:       {avg_em:.4f}")
    print(f"Avg F1:       {avg_f1:.4f}")
    print(f"Avg ROUGE-L:  {avg_rouge:.4f}")

    out_path = f"eval_answers_{retrieve_mode}_rerank{use_rerank}_rewrite{enable_rewrite}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": retrieve_mode,
                "rerank": use_rerank,
                "rewrite": enable_rewrite,
                "avg_em": avg_em,
                "avg_f1": avg_f1,
                "avg_rouge": avg_rouge,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Saved answer eval results → {out_path}")
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PvZ RAG - Retrieval & Answer Metrics + LLM-as-Judge"
    )
    subparsers = parser.add_subparsers(dest="command")

    # 7.1 info retrieval metrics
    p_ret = subparsers.add_parser("eval-retrieval", help="Evaluate retrieval metrics (Recall@k, MRR)")
    p_ret.add_argument("--qa-path", type=str, default="qa_dataset_clean.json")
    p_ret.add_argument(
        "--mode", "--retrieve-mode",
        dest="retrieve_mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense", "bm25", "hyde"]
    )
    p_ret.add_argument("--k-list", type=int, nargs="+", default=[1, 3, 5, 10])
    p_ret.add_argument("--sample-size", type=int, default=100)
    p_ret.add_argument(
        "--rerank", "--use-rerank",
        dest="use_rerank",
        action="store_true",
        help="Use cross-encoder re-ranker"
    )

    # 7.2 eval answer quality
    p_ans = subparsers.add_parser("eval-answers", help="Evaluate answer EM/F1/ROUGE")
    p_ans.add_argument("--qa-path", type=str, default="qa_dataset_clean.json")
    p_ans.add_argument(
        "--rewrite", "--enable-rewrite",
        dest="enable_rewrite",
        action="store_true",
        help="Enable query rewriting (condense question)"
    )
    p_ans.add_argument(
        "--mode", "--retrieve-mode",
        dest="retrieve_mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense", "bm25", "hyde"]
    )
    p_ans.add_argument("--sample-size", type=int, default=50)
    p_ans.add_argument(
        "--rerank", "--use-rerank",
        dest="use_rerank",
        action="store_true",
        help="Use cross-encoder re-ranker when generating answers"
    )

    # 7.3 LLM-as-Judge
    p_judge = subparsers.add_parser("judge", help="LLM-as-Judge on answer eval results")
    p_judge.add_argument("--eval-path", type=str, required=True, help="Path to eval_answers_*.json")
    p_judge.add_argument("--sample-size", type=int, default=30)

    # 7.4 ablation retrieval
    p_ablation = subparsers.add_parser(
        "ablation-retrieval",
        help="Run retrieval ablation over modes (bm25/dense/hybrid/hyde) × rerank on/off",
    )
    p_ablation.add_argument("--qa-path", type=str, default="qa_dataset_clean.json")
    p_ablation.add_argument("--sample-size", type=int, default=100)
    p_ablation.add_argument("--k-list", type=int, nargs="+", default=[1, 3, 5, 10])

    args = parser.parse_args()

    if args.command == "eval-retrieval":
        eval_retrieval_metrics(
            qa_path=args.qa_path,
            retrieve_mode=args.retrieve_mode,
            k_list=args.k_list,
            sample_size=args.sample_size,
            use_rerank=args.use_rerank,
        )
    elif args.command == "eval-answers":
        eval_answer_quality_single_turn(
            qa_path=args.qa_path,
            retrieve_mode=args.retrieve_mode,
            use_rerank=args.use_rerank,
            enable_rewrite=args.enable_rewrite,
            sample_size=args.sample_size,
        )
    elif args.command == "judge":
        llm_judge_answers(
            eval_result_path=args.eval_path,
            sample_size=args.sample_size,
        )
    elif args.command == "ablation-retrieval":
        run_retrieval_ablation(
            qa_path=args.qa_path,
            sample_size=args.sample_size,
            k_list=args.k_list,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


