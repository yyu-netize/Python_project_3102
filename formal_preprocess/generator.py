from openai import OpenAI
import torch
from search_bgem3 import UltimateRAG

# --- CONFIGURATION ---
SILICONFLOW_API_KEY = "sk-bpdybiceumehnyfudsglnizvhqssgpsjpusvienlfgchijdl"  # SiliconFlow API key
MODEL_NAME = "Qwen/Qwen3-8B"      

# Initialize the Client pointing to SiliconFlow
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1/",  
    api_key=SILICONFLOW_API_KEY,
)

print(f"\nSUCCESS: Client initialized using model: {MODEL_NAME}")

class UltimateRAGWithGenerator:
    def __init__(self):
        
        self.client = client
        self.model_name = MODEL_NAME
        self.rag = UltimateRAG()

    def generate_answer(self, query, candidates, prompt_mode, message_mode):
        """
        Use Qwen3-8B to generate the final answer
        """
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
                answer = response.choices[0].message.content.strip()  
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
                answer = response.choices[0].message.content.strip() 
                return answer
            
            except Exception as e:
                print(f"Error during generation: {e}")
                return "Sorry, I couldn't generate an answer at the moment."

    def search(self, query, retrieve_mode, prompt_mode, message_mode):
        """
        Do the retrieval based on the specified mode, then rerank and tranfer to LLM for answer generation.
        1. retrieve_mode: hybrid / dense / bm25 / hyde
        2. prompt_mode: vanilla / instruction
        3. message_mode: with_system / no_system
        """
        print(f"\n🔎 Query: {query}")
        candidates = []

        if retrieve_mode == "hyde":
            hyde_doc = self.hyde_generate_doc(query)
            print(f"\n📄 HyDE Generated Document:\n{hyde_doc}\n")

            # Dense embedding from HyDE doc
            candidates = self.retrieve_dense(hyde_doc, top_k=30)
            print(f"   - HyDE Dense Retriever finds {len(candidates)} candidates.")

        if (retrieve_mode == "hybrid"):
            candidates = self.retrieve_hybrid(query, top_k=30)
            print(f" - Dense Retriever + BM25 finds {len(candidates)} candidates.")
            
        if (retrieve_mode == "dense"):
            candidates = self.retrieve_dense(query, top_k=30)
            print(f" - Dense Retriever finds {len(candidates)} candidates.")
            
        if (retrieve_mode == "bm25"):
            candidates = self.retrieve_bm25(query, top_k=30)
            print(f" - BM25 finds {len(candidates)} candidates.")
        
        # Rerank candidates, get top N
        final_results = self.rerank(query, candidates, top_n=5)

        # Use Qwen3-8B to generate the final answer
        answer = self.generate_answer(query, final_results, prompt_mode, message_mode)

        # Show final answer
        print(f"Answer:\n {answer}")
        return answer


if __name__ == "__main__":
    rag_with_generator = UltimateRAGWithGenerator()
    
    # test cases
    rag_with_generator.search("Which plant can slow down zombies?", retrieve_mode="hybrid", prompt_mode="instruction", message_mode="with_system")
    rag_with_generator.search("What is the sun cost of Peashooter?", retrieve_mode="dense", prompt_mode="instruction", message_mode="with_system")
    rag_with_generator.search("Difference between Cherry Bomb and Jalapeno", retrieve_mode="bm25", prompt_mode="instruction", message_mode="with_system")


while True:
    print("\n=== RAG Query System ===")
    print("Input format example: Your question | retrieve_mode | prompt_mode | message_mode")
    print("Parameter description:")
    print("- retrieve_mode: hybrid / dense / bm25 / hyde (default: hybrid)")
    print("- prompt_mode: vanilla / instruction (default: instruction)")
    print("- message_mode: with_system / no_system (default: with_system)")
    print("Enter q directly to exit, enter only the question to use default parameters")
    
    user_input = input("\nPlease enter query content: ")
    
    # Exit condition
    if user_input.lower() == 'q':
        break
    
    # Parse input content
    parts = [part.strip() for part in user_input.split('|')]
    query = parts[0] if parts[0] else None
    
    # Set default parameters
    retrieve_mode = "hybrid"
    prompt_mode = "instruction"
    message_mode = "with_system"
    
    # Update parameters (if provided by user)
    if len(parts) >= 2 and parts[1]:
        retrieve_mode = parts[1]
    if len(parts) >= 3 and parts[2]:
        prompt_mode = parts[2]
    if len(parts) >= 4 and parts[3]:
        message_mode = parts[3]
    
    # Validate parameter validity
    valid_retrieve_modes = ["hybrid", "dense", "bm25", "hyde"]
    valid_prompt_modes = ["vanilla", "instruction"]
    valid_message_modes = ["with_system", "no_system"]
    
    if retrieve_mode not in valid_retrieve_modes:
        print(f"Invalid retrieve_mode: {retrieve_mode}, using default value hybrid")
        retrieve_mode = "hybrid"
    
    if prompt_mode not in valid_prompt_modes:
        print(f"Invalid prompt_mode: {prompt_mode}, using default value instruction")
        prompt_mode = "instruction"
    
    if message_mode not in valid_message_modes:
        print(f"Invalid message_mode: {message_mode}, using default value with_system")
        message_mode = "with_system"
    
    # Execute query
    if query:
        print(f"\nExecuting query - retrieve_mode: {retrieve_mode}, prompt_mode: {prompt_mode}, message_mode: {message_mode}")
        rag_with_generator.search(
            query, 
            retrieve_mode=retrieve_mode, 
            prompt_mode=prompt_mode, 
            message_mode=message_mode
        )
    else:
        print("Query content cannot be empty!")