import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .retriever import VectorRetriever, RetrievalResult

@dataclass
class RAGResponse:
    query: str
    answer: str
    evidence: List[RetrievalResult]
    citations: List[str]
    top_similarity_score: float
    chunks_retrieved: int
    is_abstention: bool = False
    llm_provider_used: str = "offline_grounded"
    error_message: Optional[str] = None

class RAGService:
    def __init__(
        self,
        retriever: VectorRetriever,
        llm_provider: str = "offline",
        api_key: str = "",
        model_name: str = "gemini-2.0-flash",
        similarity_threshold: float = 0.12
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

    def answer_question(self, query: str, top_k: int = 5) -> RAGResponse:
        evidence = self.retriever.retrieve(query, top_k=top_k)

        # Check for empty retrieval or below similarity threshold
        if not evidence or (evidence and evidence[0].similarity_score < self.similarity_threshold):
            return RAGResponse(
                query=query,
                answer="The uploaded documents do not contain sufficient evidence to answer this question reliably.",
                evidence=evidence,
                citations=[],
                top_similarity_score=evidence[0].similarity_score if evidence else 0.0,
                chunks_retrieved=len(evidence),
                is_abstention=True,
                llm_provider_used=self.llm_provider
            )

        top_score = evidence[0].similarity_score

        # Extract citations
        citations = []
        for e in evidence:
            loc = f"Page {e.page_number}" if e.page_number else (f"Sheet: {e.sheet_name}" if e.sheet_name else "")
            sec = f"Section: {e.section_title}" if e.section_title else ""
            cite_str = f"{e.document_name} ({', '.join(filter(None, [loc, sec]))})"
            if cite_str not in citations:
                citations.append(cite_str)

        # Build Grounded Context
        context_blocks = []
        for i, ev in enumerate(evidence):
            loc_str = f"Page {ev.page_number}" if ev.page_number else (f"Sheet: {ev.sheet_name}" if ev.sheet_name else "Section")
            sec_str = f"Section: {ev.section_title}" if ev.section_title else ""
            context_blocks.append(
                f"[Evidence {i+1}] Document: {ev.document_name} | {loc_str} | {sec_str}\n"
                f"Content: {ev.text}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are NexusRAG, an Evidence-First Enterprise Knowledge Intelligence system. "
            "Your answers MUST be strictly derived ONLY from the provided retrieved evidence. "
            "Do NOT invent facts, use outside knowledge, or fabricate sources. "
            "For every factual statement, explicitly cite the source document, page, and section. "
            "If the provided evidence is insufficient to answer the question, clearly state: "
            "'The uploaded documents do not contain sufficient information to answer this question.'"
        )

        user_prompt = (
            f"QUESTION: {query}\n\n"
            f"=== RETRIEVED DOCUMENT EVIDENCE ===\n"
            f"{context_text}\n\n"
            f"Provide a clear, grounded answer with explicit citations."
        )

        answer_text, provider_used = self._call_llm(user_prompt, system_prompt, evidence, query)

        return RAGResponse(
            query=query,
            answer=answer_text,
            evidence=evidence,
            citations=citations,
            top_similarity_score=top_score,
            chunks_retrieved=len(evidence),
            is_abstention=False,
            llm_provider_used=provider_used
        )

    def _call_llm(self, user_prompt: str, system_prompt: str, evidence: List[RetrievalResult], query: str) -> (str, str):
        # 1. Gemini API
        if self.llm_provider == "gemini" and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(
                    model_name=self.model_name or "gemini-2.0-flash",
                    system_instruction=system_prompt
                )
                resp = model.generate_content(user_prompt)
                if resp and resp.text:
                    return resp.text.strip(), "Gemini (" + (self.model_name or "gemini-2.0-flash") + ")"
            except Exception as e:
                print(f"[Gemini API Warning] {e}. Falling back to deterministic grounded response.")

        # 2. OpenAI API
        elif self.llm_provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                resp = client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                if resp.choices and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip(), "OpenAI (" + (self.model_name or "gpt-4o-mini") + ")"
            except Exception as e:
                print(f"[OpenAI API Warning] {e}. Falling back to deterministic grounded response.")

        # 3. Deterministic Grounded Synthesis (Offline zero-cost fallback)
        return self._generate_deterministic_grounded(evidence, query), "Local Grounded Engine"

    def _generate_deterministic_grounded(self, evidence: List[RetrievalResult], query: str) -> str:
        q_lower = query.lower()

        # Check for core policy questions in sample dataset
        if "attendance" in q_lower or "working hours" in q_lower:
            points = []
            for ev in evidence:
                if "attendance" in ev.text.lower() or "working hours" in ev.text.lower():
                    loc = f"Page {ev.page_number}" if ev.page_number else ""
                    points.append(f"• **{ev.document_name} ({loc}, {ev.section_title}):**\n  {ev.text}")
            if points:
                return "Based on the retrieved document evidence:\n\n" + "\n\n".join(points)

        if "remote work" in q_lower or "remote working" in q_lower:
            points = []
            for ev in evidence:
                if "remote" in ev.text.lower():
                    loc = f"Page {ev.page_number}" if ev.page_number else ""
                    points.append(f"• **{ev.document_name} ({loc}, {ev.section_title}):**\n  {ev.text}")
            if points:
                return "Based on the retrieved document evidence:\n\n" + "\n\n".join(points)

        if "security" in q_lower or "mfa" in q_lower or "encryption" in q_lower or "sr402" in q_lower or "sr-402" in q_lower:
            points = []
            for ev in evidence:
                if any(k in ev.text.lower() for k in ["mfa", "security", "aes-256", "encryption", "sr-402", "incident"]):
                    loc = f"Page {ev.page_number}" if ev.page_number else ""
                    points.append(f"• **{ev.document_name} ({loc}, {ev.section_title}):**\n  {ev.text}")
            if points:
                return "Based on the retrieved document evidence:\n\n" + "\n\n".join(points)

        # General grounded response from top 2 evidence chunks
        snippets = []
        for i, ev in enumerate(evidence[:2]):
            loc = f"Page {ev.page_number}" if ev.page_number else (f"Sheet: {ev.sheet_name}" if ev.sheet_name else "")
            sec = f"Section: {ev.section_title}" if ev.section_title else ""
            cite = ", ".join(filter(None, [ev.document_name, loc, sec]))
            trimmed_text = ev.text[:300] + "..." if len(ev.text) > 300 else ev.text
            snippets.append(f"• **{cite}:**\n  {trimmed_text}")

        return "Based on the retrieved document evidence:\n\n" + "\n\n".join(snippets)
