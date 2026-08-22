from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ChatMessage:
    role: str # 'user', 'assistant', 'system'
    content: str
    intent: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_label: Optional[str] = None
    evidence_cards: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""

class ConversationMemory:
    def __init__(self):
        self.history: List[ChatMessage] = []
        self.active_topic: Optional[str] = None

    def add_user_message(self, message: str) -> ChatMessage:
        msg = ChatMessage(role="user", content=message)
        self.history.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        intent: str = "general_qa",
        confidence_score: float = 0.9,
        confidence_label: str = "High",
        evidence_cards: Optional[List[Dict[str, Any]]] = None
    ) -> ChatMessage:
        msg = ChatMessage(
            role="assistant",
            content=content,
            intent=intent,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            evidence_cards=evidence_cards or []
        )
        self.history.append(msg)
        return msg

    def contextualize_query(self, current_query: str) -> str:
        """
        Resolves pronouns and follow-up ambiguities (e.g. 'What happens if it is below that?').
        """
        q_lower = current_query.lower()
        if len(self.history) < 2:
            return current_query

        # Check for pronouns / references like 'it', 'that', 'they', 'below that', 'above that'
        pronoun_triggers = ["it ", " it", "that", "this", "they", "below that", "above that", "what happens if", "why?"]
        has_pronoun = any(p in q_lower for p in pronoun_triggers)

        if has_pronoun:
            # Find the last user-assistant interaction
            last_user_query = ""
            last_assistant_answer = ""
            for msg in reversed(self.history[:-1]):
                if msg.role == "user" and not last_user_query:
                    last_user_query = msg.content
                elif msg.role == "assistant" and not last_assistant_answer:
                    last_assistant_answer = msg.content

            if last_user_query:
                # Extract key noun phrase
                if "attendance" in last_user_query.lower():
                    return f"{current_query} (Context: Attendance requirement / policy consequences)"
                elif "remote work" in last_user_query.lower():
                    return f"{current_query} (Context: Remote work authorization guidelines)"
                elif "process x" in last_user_query.lower():
                    return f"{current_query} (Context: Process X incident response)"
                elif "regulation" in last_user_query.lower() or "sr-402" in last_user_query.lower():
                    return f"{current_query} (Context: Enterprise Security Regulation SR-402)"
                else:
                    return f"{current_query} (Context regarding: {last_user_query})"

        return current_query

    def clear(self):
        self.history = []
