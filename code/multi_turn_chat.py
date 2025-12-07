# Cell 1: 基础导入 + 多轮对话数据结构

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid

from generator import UltimateRAGWithGenerator, client, MODEL_NAME

@dataclass
class TurnRecord:
    """
    一轮对话的记录，后面可以用于：
    - 分析多轮效果
    - 画图/导出日志
    """
    turn_id: str
    user_query: str
    rewritten_query: str
    answer: str
    retrieve_mode: str
    rerank_mode: bool
    prompt_mode: str
    message_mode: str
    timestamp: float

@dataclass
class ConversationState:
    """
    管理一个会话的历史状态。
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: List[TurnRecord] = field(default_factory=list)

    def add_turn(self, turn: TurnRecord):
        self.turns.append(turn)

    def last_k_turns(self, k: int = 3) -> List[TurnRecord]:
        return self.turns[-k:]
    
def rewrite_query_with_history(
    user_query: str,
    history: List[TurnRecord],
    enable_rewrite: bool = True,
    max_history_turns: int = 3,
) -> str:
    """
    将当前用户问题在对话历史的上下文中，改写为一个“独立、可检索、可评估”的英文问题。
    
    参数：
    - user_query: 当前用户输入的问题（可能是追问，例如：“那它的冷却时间呢？”）
    - history: 之前的对话轮次（TurnRecord 列表）
    - enable_rewrite: 如果为 False，则直接返回原问题
    - max_history_turns: 使用最近多少轮对话来帮助理解当前追问
    
    返回：
    - rewritten_query: 适合丢给 PvZ RAG 检索的英文独立问题
    """
    # 情况 1：不开启 rewrite 或者没有历史 → 直接返回原问题
    if not enable_rewrite or not history:
        return user_query

    # === 1. 准备历史文本 ===
    # 取最近的 max_history_turns 轮对话，避免上下文过长
    selected_history = history[-max_history_turns:]
    history_text_blocks = []
    for turn in selected_history:
        history_text_blocks.append(
            f"User: {turn.user_query}\nAssistant: {turn.answer}"
        )
    history_block = "\n\n".join(history_text_blocks)

    # === 2. System Prompt：你的高质量规则版本 ===
    system_prompt = (
        "You are an expert query rewriter for a Plants vs. Zombies (PvZ) knowledge base. "
        "Your task is to rewrite a follow-up question into a standalone, self-contained English question "
        "that can be easily understood and answered using a PvZ wiki.\n\n"
        "RULES:\n"
        "1. Preserve Intent: The rewritten question must have the same core meaning as the original follow-up.\n"
        "2. Be Standalone: It must be fully understandable without any of the previous chat history.\n"
        "3. Use PvZ Terminology: Use correct and specific terms from the game (e.g., 'sun cost', 'recharge time', 'plant food effect').\n"
        "4. Do Not Answer: Your only job is to rewrite the question, not to answer it.\n"
        "5. Handle Standalone Queries: If the original question is already a good, standalone question, return it as is.\n"
        "6. Strict Output: Output ONLY the final rewritten question. Do not include any explanations, greetings, or quotation marks."
    )

    # === 3. User Prompt：历史 + 原问题 + 示例 ===
    user_prompt = (
        "Chat History:\n"
        "---\n"
        f"{history_block}\n"
        "---\n\n"
        f"Latest Follow-up Question: {user_query}\n\n"
        "Rewrite the latest follow-up question into a standalone question based on the rules and the provided history.\n\n"
        "EXAMPLE:\n"
        "History:\n"
        "User: How much does a Peashooter cost?\n"
        "Assistant: A Peashooter costs 100 sun.\n"
        "Latest Follow-up Question: What about the Snow Pea?\n"
        "Rewritten Question: What is the sun cost for a Snow Pea in Plants vs. Zombies?\n\n"
        "Now, rewrite the given question:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",  "content": user_prompt},
    ]

    # === 4. 调用模型并处理响应 ===
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,  # 稍低温度，追求稳定、可复现的改写
            top_p=0.9,
            n=1,
        )
        rewritten = response.choices[0].message.content.strip()

        # 增强的后备机制：如果模型返回了空内容或明显无效的短句，则回退到原始查询
        if not rewritten or len(rewritten.split()) < 3:
            print("[WARN] Model returned an invalid or too-short rewrite. Falling back to the original query.")
            return user_query

        return rewritten

    except Exception as e:
        # 捕获所有可能的 API 调用异常，并优雅降级
        print(f"[WARN] Query rewrite failed due to an API error. Falling back to the original query. Error: {e}")
        return user_query
    
# Cell 3: 多轮 RAG 对话管理器（封装原有 UltimateRAGWithGenerator）

class MultiTurnRAGChat:


    def __init__(
        self,
        enable_rewrite: bool = True,
        default_retrieve_mode: str = "hybrid", 
        default_rerank_mode: bool = True,
        default_prompt_mode: str = "instruction",
        default_message_mode: str = "with_system",
    ):
        self.state = ConversationState()
        self.enable_rewrite = enable_rewrite

        # 默认的 RAG 配置（和 generator.py 里的交互式设置兼容）
        self.default_retrieve_mode = default_retrieve_mode
        self.default_rerank_mode = default_rerank_mode
        self.default_prompt_mode = default_prompt_mode
        self.default_message_mode = default_message_mode

        self.rag_gen = UltimateRAGWithGenerator()

    def ask(
        self,
        user_query: str,
        retrieve_mode: Optional[str] = None,
        rerank_mode: Optional[bool] = None,
        prompt_mode: Optional[str] = None,
        message_mode: Optional[str] = None,
    ) -> str:
        """
        对外接口：用户问一句 → 返回 RAG 答案。

        流程：
        1. 从 ConversationState 取最近几轮历史
        2. 如果开启了 rewrite，用历史 + Cell2 的函数改写为独立问题
        3. 调用 self.rag_gen.search(...) 做检索 + 生成
        4. 把本轮记录成 TurnRecord，追加到 state.turns
        """
        # 如果调用者没指定，就用默认模式
        r_mode = retrieve_mode or self.default_retrieve_mode
        rr_mode = rerank_mode or self.default_rerank_mode
        p_mode = prompt_mode or self.default_prompt_mode
        m_mode = message_mode or self.default_message_mode

        # 1) 取最近几轮历史，用于改写
        history_turns = self.state.last_k_turns(3)

        rewritten = rewrite_query_with_history(
            user_query=user_query,
            history=history_turns,
            enable_rewrite=self.enable_rewrite,
            max_history_turns=3,
        )

        print(f"\n[MultiTurn] 原始问题: {user_query}")
        if self.enable_rewrite and history_turns:
            print(f"[MultiTurn] 改写后用于检索的问题: {rewritten}")

        # 2) use rewritten query to search + generate answer
        answer = self.rag_gen.search(
            query=rewritten,
            retrieve_mode=r_mode,
            rerank_mode=rr_mode,
            prompt_mode=p_mode,
            message_mode=m_mode,
        )

        # 3) record this turn
        turn = TurnRecord(
            turn_id=str(uuid.uuid4()),
            user_query=user_query,
            rewritten_query=rewritten,
            answer=answer,
            retrieve_mode=r_mode,
            rerank_mode=rr_mode,
            prompt_mode=p_mode,
            message_mode=m_mode,
            timestamp=time.time(),
        )
        self.state.add_turn(turn)

        return answer

    def get_history(self) -> List[TurnRecord]:
        """返回当前会话的全部历史轮次。"""
        return self.state.turns

    def reset(self):
        """重置整个对话（换一个新的 session_id）。"""
        self.state = ConversationState()

    def export_history_as_dicts(self) -> List[Dict[str, Any]]:
        """
        将历史记录导出为列表(dict)，方便：
        - 存成 JSON 用于报告附录
        - 做错误分析和可视化（e.g. 哪类问题 rewrite 帮助大）
        """
        records = []
        for t in self.state.turns:
            records.append(
                {
                    "turn_id": t.turn_id,
                    "session_id": self.state.session_id,
                    "user_query": t.user_query,
                    "rewritten_query": t.rewritten_query,
                    "answer": t.answer,
                    "retrieve_mode": t.retrieve_mode,
                    "rerank_mode": t.rerank_mode,
                    "prompt_mode": t.prompt_mode,
                    "message_mode": t.message_mode,
                    "timestamp": t.timestamp,
                }
            )
        return records

# Cell 4: 构建多轮 RAG 对话引擎 + 小型 Demo

#    - chat_with_rewrite: 开启 query rewrite（多轮增强版）
#    - chat_no_rewrite:   关闭 query rewrite（baseline，对照组）

chat_with_rewrite = MultiTurnRAGChat(
    enable_rewrite=True,
    default_retrieve_mode="hybrid",
    default_rerank_mode=True,
    default_prompt_mode="instruction",
    default_message_mode="with_system",
)

chat_no_rewrite = MultiTurnRAGChat(
    enable_rewrite=False,   # 关闭 rewrite，当作 baseline
    default_retrieve_mode="hybrid",
    default_rerank_mode=True,
    default_prompt_mode="instruction",
    default_message_mode="with_system",
)

def run_demo_conversation(chat: MultiTurnRAGChat, title: str):

    print(f"\n========== {title} ==========")

    demo_questions = [
        "In Versus Mode, what is special about Catapult Zombie?",
        "Why should I protect him?",
        "Which columns does he mainly attack?",
    ]

    for idx, q in enumerate(demo_questions, start=1):
        print(f"\n[Turn {idx}]")
        answer = chat.ask(q)
        print("Bot:", answer[:400], "...\n")  # 截断一点，避免输出太长

    # 打印一下重写前后的对照，方便你以后拷进报告
    print("\n----- Conversation History -----")
    for t in chat.get_history():
        print("\nUser:      ", t.user_query)
        print("Rewritten: ", t.rewritten_query)
        print("Answer:    ", t.answer[:200], "...")

# 2) 跑一次“开启 rewrite”的多轮 Demo
run_demo_conversation(chat_with_rewrite, title="Multi-turn RAG with Query Rewrite (增强版)")

# 3) 如需对比效果，可以再运行下面这一行，看看关闭 rewrite 时的表现
#    建议先观察上面的输出，再决定是否要跑 baseline（会多调用几次模型）

# run_demo_conversation(chat_no_rewrite, title="Multi-turn RAG without Query Rewrite (Baseline)")

