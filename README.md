# 🛡️ NexusRAG — Complete Enterprise Knowledge Intelligence Platform

> **"NexusRAG is an evidence-first enterprise knowledge intelligence & multi-document reasoning platform designed to reason over heterogeneous document collections with full provenance, traceability, zero hallucination, version intelligence, interactive knowledge graphs, and automated evaluation."**

---

## 📌 Architecture & System Flow

```text
Enterprise Documents (PDF, DOCX, TXT, CSV, XLSX)
        ↓
Multi-Modal Ingestion & SHA-256 Deduplication
        ↓
Metadata Extraction & Sliding-Window Chunking
        ↓
Dense Embeddings (Local / Gemini / OpenAI)
        ↓
Persistent Vector Store (L2-Normalized) + SQLite Registry
        ↓
Hybrid Retrieval Layer (Dense Semantic + Lexical BM25)
        ↓
Precision Reranking Layer (Code, Date & Term Boost)
        ↓
Structured Multi-Source Context Builder
        ↓
LLM Synthesis Layer (Gemini / OpenAI / Offline Grounded Engine)
        ↓
Grounded Answer + Citations [1] + Confidence Estimator
        ↓
Cross-Version Diff Matrix + Interactive Knowledge Graph + Evaluation Benchmark
```

---

## 🌟 Complete Feature Suite

### 1. 📁 Multi-Format Document Ingestion & Deduplication
- **Parsers:** Full support for `PDF` (`PyMuPDF`), `DOCX` (`python-docx`), `TXT`, `CSV`, and `XLSX` (`openpyxl`/`pandas`) with sheet and table preservation.
- **Traceable Chunking:** Sliding-window chunker with page numbers, section headers, sheet names, versions, and years.
- **SHA-256 Duplicate Prevention:** Detects identical content even with different filenames.
- **Cascade Deletion:** Deleting a document purges all associated SQLite chunks, dense vector embeddings, and knowledge graph entities.

### 2. 🎯 Persistent Vector Store & Embeddings
- **Local Dense Embeddings:** Zero-cost, local $L_2$-normalized TF-IDF dense embeddings.
- **Cloud Adapters:** Plug-and-play support for Google Generative AI (`text-embedding-004`) and OpenAI (`text-embedding-3-small`).
- **Disk Persistence:** Embeddings survive application restarts in `nexusrag/data/vector_store/`.

### 3. 🔎 Hybrid Retrieval & Precision Reranking
- **BM25 Lexical Search:** Fast keyword and exact phrase token matching.
- **Dense Semantic Retrieval:** Cosine dot-product similarity search.
- **Convex Score Fusion:** $	ext{Score} = lpha \cdot 	ext{Semantic} + (1-lpha) \cdot 	ext{BM25}$.
- **Precision Reranker:** High-precision scoring with term, code (e.g. `SR-402`), percentage (`60%`), and date matching.

### 4. 💬 Evidence-First Grounded Chat
- **Zero Hallucination Guard:** Strict prompting mandates answering only from retrieved sources.
- **Explicit Abstention:** Safely abstains on unsupported queries:
  > *"I couldn't find sufficient evidence in the uploaded documents to answer this question."*
- **Bracketed Citations `[1]`, `[2]`:** Automatically extracted and mapped to document name, page, section, version, and year.
- **Transparent Confidence Estimator:** Computes `High` ($\ge 75\%$), `Medium` ($50\% - 74\%$), or `Low` ($< 50\%$) confidence with human-readable rationale.
- **Cross-Version Conflict Disclosure:** Outlines discrepancies when evidence spans multiple document versions.

### 5. ⚖️ Version Intelligence & Document Comparison
- **Automated Clause Diffing:** Compares baseline vs updated document versions (e.g. 2025 vs 2026 policies).
- **Categorized Changes:** Displays `Added`, `Removed`, `Modified`, and `Unchanged` clauses with content deltas and source citations.

### 6. 🕸️ Interactive Enterprise Knowledge Graph
- **Entity Extraction:** Policies, Regulations, Departments, Requirements, Versions, and Dates.
- **Relation Extraction:** `REQUIRES`, `APPLIES_TO`, `CONTAINS`, `UPDATED_BY`, `SUPERSEDES`.
- **Plotly Visual Network:** Interactive node-link topology with hover tooltips and searchable relationship table.

### 7. 📊 Evaluation & Quality Monitoring Suite
- **Automated Benchmarking:** Evaluates Precision@K, Recall@K, Faithfulness Score, Relevance Score, and Citation Accuracy.
- **Ground Truth Test Cases:** Built-in enterprise benchmark dataset with instant pass/fail telemetry.

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/nexusrag/nexusrag.git
cd NexusRAG
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)
```bash
cp .env.example .env
```

### 3. Launch the Application
```bash
python -m streamlit run main.py --server.port 8501 --server.address 0.0.0.0
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧪 Automated Testing

Run the full automated test suite (40 unit tests):

```bash
python -m unittest discover -s nexusrag/tests -p "test_*.py"
```

---

## 🧭 Application Navigation

| Page | URL | Description |
| :--- | :--- | :--- |
| **Landing & Hub** | `/` | Platform overview, telemetry metrics, and quick navigation. |
| **Dashboard** | `/Dashboard` | Real-time storage telemetry, registry health, format distribution. |
| **Documents Hub** | `/Documents` | Multi-format upload, document registry, hybrid search, and chunk inspector. |
| **Evidence Chat** | `/Chat` | Evidence-first Q&A with citations, confidence badges, and evidence cards. |
| **Compare Docs** | `/Compare_Docs` | Version diffing between document versions with clause deltas. |
| **Knowledge Graph** | `/Knowledge_Graph` | Interactive NetworkX/Plotly entity-relationship knowledge network. |
| **Evaluation Suite** | `/Evaluation` | Automated RAG benchmark runner, precision, recall, and faithfulness scoring. |
| **Settings** | `/Settings` | Retrieval weights, reranker toggles, chunk sizes, and LLM configuration. |
