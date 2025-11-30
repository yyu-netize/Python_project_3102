from openai import OpenAI
import torch
from search_bgem3 import UltimateRAG

# --- CONFIGURATION ---
SILICONFLOW_API_KEY = "sk-bpdybiceumehnyfudsglnizvhqssgpsjpusvienlfgchijdl"  # SiliconFlow API 密钥
MODEL_NAME = "Qwen/Qwen3-8B"      

# Initialize the Client pointing to SiliconFlow
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1/",  
    api_key=SILICONFLOW_API_KEY,
)

print(f"\nSUCCESS: Client initialized using model: {MODEL_NAME}")

class UltimateRAGWithGenerator:
    def __init__(self):
        # 模型配置
        self.client = client
        self.model_name = MODEL_NAME
        # 初始化 UltimateRAG 实例
        self.rag = UltimateRAG()

    def generate_answer(self, query, candidates, prompt_mode, message_mode): #prompt_mode可选：1.vanilla，2.instruction，  message_mode可选：1.with_system，2.no_system
        """
        使用 Qwen3-8B 生成最终的回答
        """
        # 合并候选文本作为上下文
        unique_candidates = {cand['text'] for cand in candidates}
        context = " ".join(unique_candidates)

        system_msg = ""
        user_msg = ""
        prompt = ""
        if message_mode == "with_system":
            if prompt_mode == "vanilla":
                system_msg = "You are a helpful AI assistant."
                user_msg = f"""
Question: {query}
---
Context:
{context}
---
Answer the question using the context. If the answer is not mentioned, say you don't know.
"""
            elif prompt_mode == "instruction":
                system_msg = (
                    "You are a retrieval-augmented QA assistant for Plants vs Zombies.\n"
                    "You must answer ONLY based on the provided context.\n"
                    "Never hallucinate facts that do not appear in the context."
                )
                user_msg = f"""
Question:
{query}
---
Context:
{context}
---
Now please provide accurate, concise, and complete answers using ONLY the provided context. 
Do not hallucinate external details. Avoid redundancy.
Write complete sentences.
If the answer is not mentioned, say you don't know.
"""
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    stop=None,
                    temperature=0.7,
                    top_p=1.0,
                    n=1,
                )
                answer = response.choices[0].message.content.strip()  # 获取生成的回答
                return answer
            
            except Exception as e:
                print(f"Error during generation: {e}")
                return "Sorry, I couldn't generate an answer at the moment."

        elif message_mode == "no_system":
            if prompt_mode == "vanilla":
                prompt = f"""
Use the following context to answer the question. 
If the answer is not in the context, respond with "I don't know".
---
Question:
{query}
---
Context:
{context}
---
Now please provide an answer.
"""
            elif prompt_mode == "instruction":
                prompt = f"""
You are a Plants vs. Zombies domain expert.
Your job is to provide accurate, concise, and complete answers using ONLY the provided context. 
Do not hallucinate external details. Avoid redundancy.
Write complete sentences.
If the answer is not in the context, respond with "I don't know".
---
Question:
{query}
---
Context:
{context}
---
Now provide an answer.
"""
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    stop=None,
                    temperature=0.7,
                    top_p=1.0,
                    n=1,
                )
                answer = response.choices[0].message.content.strip()  # 获取生成的回答
                return answer
            
            except Exception as e:
                print(f"Error during generation: {e}")
                return "Sorry, I couldn't generate an answer at the moment."

    def search(self, query, retrieve_mode, prompt_mode, message_mode):
        """
        进行检索并生成最终的回答
        """
        print(f"\n🔎 Query: {query}")
        candidate = []

        # 1. 混合召回 (Recall) - 获取候选文本 (例如 30 个)
        if retrieve_mode == "hyde":
            hyde_doc = self.rag.hyde_generate_doc(query)
            print(f"\n📄 HyDE Generated Document:\n{hyde_doc}\n")
            # dense embedding from HyDE doc
            candidates = self.rag.retrieve_dense(hyde_doc, top_k=30)
            print(f"   - HyDE Dense Retrieval 找到 {len(candidates)} 个候选片段")
        if retrieve_mode == "hybrid":
            candidates = self.rag.retrieve_hybrid(query, top_k=30)
            print(f"   - 召回阶段找到 {len(candidates)} 个候选片段 (Vector + BM25)")
        elif retrieve_mode == "dense":
            candidates = self.rag.retrieve_dense(query, top_k=30)
            print(f"   - 召回阶段找到 {len(candidates)} 个候选片段 (Vector)")
        elif retrieve_mode == "bm25":
            candidates = self.rag.retrieve_bm25(query, top_k=30)
            print(f"   - 召回阶段找到 {len(candidates)} 个候选片段 (BM25)")
        

        # 2. 重排序 (Rerank) - 提炼 Top 5
        final_results = self.rag.rerank(query, candidates, top_n=5)

        # 3. 使用 Qwen3-8B 生成最终答案
        answer = self.generate_answer(query, final_results, prompt_mode, message_mode)#prompt_mode可选：1.vanilla，2.instruction，  message_mode可选：1.with_system，2.no_system

        # 4. 展示生成的答案
        print(f"Answer:\n {answer}")
        return answer


if __name__ == "__main__":
    # 初始化 RAG 引擎
    rag_with_generator = UltimateRAGWithGenerator()
    
    # 测试查询
    rag_with_generator.search("Which plant can slow down zombies?", retrieve_mode="hybrid", prompt_mode="instruction", message_mode="with_system")
    rag_with_generator.search("What is the sun cost of Peashooter?", retrieve_mode="dense", prompt_mode="instruction", message_mode="with_system")
    rag_with_generator.search("Difference between Cherry Bomb and Jalapeno", retrieve_mode="bm25", prompt_mode="instruction", message_mode="with_system")

    # 手动输入查询
    # 手动输入查询（支持参数配置）
while True:
    print("\n=== RAG 查询系统 ===")
    print("输入格式示例: 你的问题 | retrieve_mode | prompt_mode | message_mode")
    print("参数说明:")
    print("- retrieve_mode: hybrid / dense / sparse (默认: hybrid)")
    print("- prompt_mode: vanilla / instruction (默认: instruction)")
    print("- message_mode: with_system / no_system (默认: with_system)")
    print("直接输入 q 退出，只输入问题则使用默认参数")
    
    user_input = input("\n请输入查询内容: ")
    
    # 退出条件
    if user_input.lower() == 'q':
        break
    
    # 解析输入内容
    parts = [part.strip() for part in user_input.split('|')]
    query = parts[0] if parts[0] else None
    
    # 设置默认参数
    retrieve_mode = "hybrid"
    prompt_mode = "instruction"
    message_mode = "with_system"
    
    # 更新参数（如果用户提供了）
    if len(parts) >= 2 and parts[1]:
        retrieve_mode = parts[1]
    if len(parts) >= 3 and parts[2]:
        prompt_mode = parts[2]
    if len(parts) >= 4 and parts[3]:
        message_mode = parts[3]
    
    # 验证参数有效性
    valid_retrieve_modes = ["hybrid", "dense", "sparse"]
    valid_prompt_modes = ["vanilla", "instruction"]
    valid_message_modes = ["with_system", "no_system"]
    
    if retrieve_mode not in valid_retrieve_modes:
        print(f"无效的 retrieve_mode: {retrieve_mode}，使用默认值 hybrid")
        retrieve_mode = "hybrid"
    
    if prompt_mode not in valid_prompt_modes:
        print(f"无效的 prompt_mode: {prompt_mode}，使用默认值 instruction")
        prompt_mode = "instruction"
    
    if message_mode not in valid_message_modes:
        print(f"无效的 message_mode: {message_mode}，使用默认值 with_system")
        message_mode = "with_system"
    
    # 执行查询
    if query:
        print(f"\n执行查询 - retrieve_mode: {retrieve_mode}, prompt_mode: {prompt_mode}, message_mode: {message_mode}")
        rag_with_generator.search(
            query, 
            retrieve_mode=retrieve_mode, 
            prompt_mode=prompt_mode, 
            message_mode=message_mode
        )
    else:
        print("查询内容不能为空！")