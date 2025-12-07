# multi_turn_chat.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid
import threading

# 下面这三行根据你的工程路径调整（你原来写的是 from generator import UltimateRAGWithGenerator, client, MODEL_NAME）
# 我沿用了你原来的导入名
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
    （保留了你原来的 prompt 逻辑）
    """
    if not enable_rewrite or not history:
        return user_query

    selected_history = history[-max_history_turns:]
    history_text_blocks = []
    for turn in selected_history:
        history_text_blocks.append(
            f"User: {turn.user_query}\nAssistant: {turn.answer}"
        )
    history_block = "\n\n".join(history_text_blocks)

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

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            n=1,
        )
        rewritten = response.choices[0].message.content.strip()
        if not rewritten or len(rewritten.split()) < 3:
            # fallback
            return user_query
        return rewritten
    except Exception as e:
        # graceful fallback
        print(f"[WARN] Query rewrite failed: {e}")
        return user_query

class MultiTurnRAGChat:
    """
    多轮 RAG 对话管理器。封装了 rewrite + 调用底层 RAG generator。
    """

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

        self.default_retrieve_mode = default_retrieve_mode
        self.default_rerank_mode = default_rerank_mode
        self.default_prompt_mode = default_prompt_mode
        self.default_message_mode = default_message_mode

        # 底层 RAG 生成器（与原来保持一致）
        self.rag_gen = UltimateRAGWithGenerator()

        # 锁：防止并发 ask 导致内部状态竞争（同一实例在多线程/异步环境下）
        self._lock = threading.Lock()

    def ask(
        self,
        user_query: str,
        retrieve_mode: Optional[str] = None,
        rerank_mode: Optional[bool] = None,
        prompt_mode: Optional[str] = None,
        message_mode: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        对外接口：用户问一句 → 返回 RAG 答案。
        options: 来自前端的附加选项，如 {'model':..., 'language':..., 'style':..., 'format':...}
        """
        r_mode = retrieve_mode or self.default_retrieve_mode
        rr_mode = rerank_mode or self.default_rerank_mode
        p_mode = prompt_mode or self.default_prompt_mode
        m_mode = message_mode or self.default_message_mode

        history_turns = self.state.last_k_turns(3)

        rewritten = rewrite_query_with_history(
            user_query=user_query,
            history=history_turns,
            enable_rewrite=self.enable_rewrite,
            max_history_turns=3,
        )

        # 打印方便本地调试
        print(f"[MultiTurn] Original: {user_query}")
        if self.enable_rewrite and history_turns:
            print(f"[MultiTurn] Rewritten: {rewritten}")

        # 调用底层 RAG（注意：根据你的 UltimateRAGWithGenerator.search 接口修改下面一行）
        try:
            with self._lock:
                # 我把 options 统一传给 search，便于选项驱动检索/生成行为
                # 如果你的 search 不支持 options，请把 options 从参数中移除或在这儿做兼容包装
                answer = self.rag_gen.search(
                    query=rewritten,
                    retrieve_mode=r_mode,
                    rerank_mode=rr_mode,
                    prompt_mode=p_mode,
                    message_mode=m_mode,
                    options=options or {}
                )
        except TypeError:
            # 兼容没有 options 参数的旧接口：尝试不传 options
            with self._lock:
                answer = self.rag_gen.search(
                    query=rewritten,
                    retrieve_mode=r_mode,
                    rerank_mode=rr_mode,
                    prompt_mode=p_mode,
                    message_mode=m_mode
                )
        except Exception as e:
            # 让调用端知道发生了问题（由调用处降级）
            raise RuntimeError(f"RAG search failed: {e}")

        # 记录本轮
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
        return self.state.turns

    def reset(self):
        """重置整个会话（新的 session_id，清空历史）"""
        self.state = ConversationState()

# 工厂函数：方便外部 import build_chat_bot() 来创建实例
def build_chat_bot(enable_rewrite: bool = True) -> MultiTurnRAGChat:
    return MultiTurnRAGChat(enable_rewrite=enable_rewrite)
