
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI
import matplotlib.pyplot as plt  # 用于画饼图
import os  # 为了安全覆盖写文件（临时文件 + replace）


# --- CONFIGURATION ---
SILICONFLOW_API_KEY = "sk-qxocptxsyigkhfruvqjevqjngthhmzdwetgonkditizarmue"  # SiliconFlow API 密钥
LLM_NAME = "Qwen/Qwen3-8B"      

# Initialize the Client pointing to SiliconFlow
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1/",  
    api_key=SILICONFLOW_API_KEY,
)

# 输入 & 输出路径
INPUT_PATH = Path("qa_dataset_single.json")              
REVIEW_LOG_PATH = Path("qa_dataset_llm_reviewed.jsonl") # 每条 QA 的审稿结果（带打分、理由、建议）
FINAL_PATH = Path("qa_dataset_clean.json")         # 模型自动筛选后的最终 QA 集

MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"  # 如果你有更强的模型，可以改这里


# ========== 工具：安全保存 JSON（先写临时文件，再原子替换） ==========

def safe_save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """
    安全写入 JSON：
    - 先写到 path.tmp
    - 写完关闭后，用 replace 覆盖正式文件
    这样即使中途 KeyboardInterrupt，旧文件也不会被破坏。
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


# ========== 1. 构造给 LLM 的提示词（prompt） ==========

def build_prompt(item: Dict[str, Any]) -> str:
    """
    把 question / answer / source_text 拼成一条指令，
    要求模型输出 JSON（verdict + 各项评分 + 可选改写）。
    """
    q = item["question"]
    a = item["answer"]
    ctx = item["source_text"]

    prompt = f"""
You are a QA quality reviewer for a Plants vs. Zombies RAG system.

You are given:
- A question
- An answer
- The source_text (the wiki chunk from which the answer should be grounded)

Your job:
1. Judge the QUALITY of this QA pair.
2. Decide whether to:
   - "keep": the QA is already good, no need to change.
   - "rewrite": the QA is salvageable but needs a better question and/or answer.
   - "drop": the QA is bad and should be removed (e.g., unclear, meta-only, ungrounded, nonsense).

3. When you "rewrite", you MUST ensure:
   - The new question is self-contained (NO pronouns like "this", "that", "last time" without a clear referent).
   - The new question and answer are clearly grounded in the source_text.
   - The answer is a complete sentence or short paragraph, not just "Yes"/"No"/"Same as before".

4. Output a SINGLE JSON object with the following fields:
{{
  "verdict": "keep" | "rewrite" | "drop",
  "quality_score": float between 0 and 1,
  "clarity": float between 0 and 1,
  "groundedness": float between 0 and 1,
  "completeness": float between 0 and 1,
  "non_triviality": float between 0 and 1,
  "reason": "short natural language explanation",
  "suggested_question": "string or null",
  "suggested_answer": "string or null"
}}

Important:
- If you choose "keep", still fill in all scores and give a short reason.
- If you choose "rewrite", provide BOTH suggested_question and suggested_answer.
- If you choose "drop", you can set suggested_question and suggested_answer to null.

Now here is the QA pair:

[QUESTION]
{q}

[ANSWER]
{a}

[SOURCE_TEXT]
{ctx}
"""
    return prompt.strip()


# ========== 2. 调用模型 ==========

def call_llm(prompt: str) -> str:
    """
    调用 OpenAI Chat Completions，返回文本形式的回复。
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a careful JSON-only QA reviewer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ========== 3. 从模型回复中提取 JSON ==========

def extract_json_blob(text: str) -> str:
    """
    有些模型会在 JSON 外面多加解释文字。
    这个函数尝试从中截取第一个 {...} JSON 段。
    """
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError(f"没有找到有效 JSON: {text}")
    return text[first:last+1]


def parse_llm_json(raw_text: str) -> Dict[str, Any]:
    blob = extract_json_blob(raw_text)
    return json.loads(blob)


# ========== 4. 主流程：逐条审稿 + 自动替换 answer + 支持 drop + 画饼图 + 定期保存 ==========

def main():
    items: List[Dict[str, Any]] = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print("Loaded QA items:", len(items))

    reviewed_log: List[Dict[str, Any]] = []
    final_items: List[Dict[str, Any]] = []

    keep_count = 0
    rewrite_count = 0
    drop_count = 0
    kept_or_rewritten = 0  # 只统计 keep + rewrite，用于“每 10 条保存一次”

    try:
        for idx, item in enumerate(items):
            qid = item.get("id", f"no_id_{idx}")
            print(f"\n=== Reviewing #{idx+1}/{len(items)} | id={qid} ===")

            prompt = build_prompt(item)

            # 调模型 + 解析 JSON
            try:
                raw = call_llm(prompt)
                result = parse_llm_json(raw)
            except Exception as e:
                print(f"[Error] LLM 调用或解析失败，id={qid}, error={e}")
                # 保守处理：失败就视为 drop，避免脏数据进入 final set
                drop_count += 1
                continue

            verdict = result.get("verdict", "drop")
            suggested_q = result.get("suggested_question")
            suggested_a = result.get("suggested_answer")

            # 记录日志
            log_entry = {
                "id": qid,
                "original_question": item["question"],
                "original_answer": item["answer"],
                "verdict": verdict,
                "quality_score": result.get("quality_score"),
                "clarity": result.get("clarity"),
                "groundedness": result.get("groundedness"),
                "completeness": result.get("completeness"),
                "non_triviality": result.get("non_triviality"),
                "reason": result.get("reason"),
                "suggested_question": suggested_q,
                "suggested_answer": suggested_a,
            }
            reviewed_log.append(log_entry)

            # 根据 verdict 处理
            if verdict == "keep":
                final_items.append(item)
                keep_count += 1
                kept_or_rewritten += 1

            elif verdict == "rewrite":
                if suggested_a:
                    item["answer"] = suggested_a
                final_items.append(item)
                rewrite_count += 1
                kept_or_rewritten += 1

            elif verdict == "drop":
                drop_count += 1

            else:
                drop_count += 1

            # 每处理 10 条 keep/rewrite，就安全保存一次当前结果
            if kept_or_rewritten > 0 and kept_or_rewritten % 10 == 0:
                print(f"[Autosave] Processed {kept_or_rewritten} kept/rewrite items, saving partial results...")
                safe_save_json(FINAL_PATH, final_items)

            # 防止过快请求，可以按需调整或删除
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Saving current progress before exit...")
        # 保存当前已处理的 final_items
        safe_save_json(FINAL_PATH, final_items)
        # 保存当前审稿日志
        with REVIEW_LOG_PATH.open("w", encoding="utf-8") as f:
            for row in reviewed_log:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[INFO] Partial results saved. You can safely resume or inspect the file.")
        return

    # 正常跑完后的最终保存（再保存一次保证完整）
    with REVIEW_LOG_PATH.open("w", encoding="utf-8") as f:
        for row in reviewed_log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    safe_save_json(FINAL_PATH, final_items)

    print("\n=== Summary ===")
    print("Original QA items:", len(items))
    print("Kept:", keep_count)
    print("Rewritten:", rewrite_count)
    print("Dropped:", drop_count)
    print("Final QA items:", len(final_items))
    print(f"Saved review log to {REVIEW_LOG_PATH.name}")
    print(f"Saved auto final eval set to {FINAL_PATH.name}")

    # ========== 5. 画 keep vs rewrite 的饼图 ==========
    kept_plus_rewrite = keep_count + rewrite_count
    if kept_plus_rewrite > 0:
        labels = []
        sizes = []

        if keep_count > 0:
            labels.append("keep")
            sizes.append(keep_count)
        if rewrite_count > 0:
            labels.append("rewrite")
            sizes.append(rewrite_count)

        plt.figure(figsize=(5, 5))
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title(f"Keep vs Rewrite (n={kept_plus_rewrite})")
        plt.axis("equal")

        pie_path = "qa_keep_rewrite_pie.png"
        plt.savefig(pie_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved keep/rewrite pie chart to {pie_path}")
    else:
        print("[INFO] No items kept or rewritten; skip pie chart generation.")


if __name__ == "__main__":
    main()
