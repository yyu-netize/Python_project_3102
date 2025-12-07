from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid

from generator import UltimateRAGWithGenerator, client, MODEL_NAME

@dataclass
class TurnRecord:
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
            print("[WARN] Model returned an invalid or too-short rewrite. Falling back to the original query.")
            return user_query

        return rewritten

    except Exception as e:
        print(f"[WARN] Query rewrite failed due to an API error. Falling back to the original query. Error: {e}")
        return user_query
    
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

        print(f"\n[MultiTurn] 原始问题: {user_query}")
        if self.enable_rewrite and history_turns:
            print(f"[MultiTurn] 改写后用于检索的问题: {rewritten}")

        answer = self.rag_gen.search(
            query=rewritten,
            retrieve_mode=r_mode,
            rerank_mode=rr_mode,
            prompt_mode=p_mode,
            message_mode=m_mode,
        )

        #record this turn
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
        self.state = ConversationState()

    def export_history_as_dicts(self) -> List[Dict[str, Any]]:
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

chat_with_rewrite = MultiTurnRAGChat(
    enable_rewrite=True,
    default_retrieve_mode="hybrid",
    default_rerank_mode=True,
    default_prompt_mode="instruction",
    default_message_mode="with_system",
)

chat_no_rewrite = MultiTurnRAGChat(
    enable_rewrite=False,   
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
        print("Bot:", answer[:400], "...\n")  

    print("\n----- Conversation History -----")
    for t in chat.get_history():
        print("\nUser:      ", t.user_query)
        print("Rewritten: ", t.rewritten_query)
        print("Answer:    ", t.answer[:200], "...")

run_demo_conversation(chat_with_rewrite, title="Multi-turn RAG with Query Rewrite (增强版)")

# run_demo_conversation(chat_no_rewrite, title="Multi-turn RAG without Query Rewrite (Baseline)")

