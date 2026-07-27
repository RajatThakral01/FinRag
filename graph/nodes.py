import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

import config
from tools.output_parsers import parse_route, parse_companies, parse_grade, parse_hallucination, parse_calculation
from tools.calculator import compute
from graph.state import create_initial_state, GraphState
from tools.vectorstore import get_vectorstore
from tools.company_names import SHORT_TO_FULL, get_all_full_names
from tools.bm25_index import bm25_query
from tools.retrieval_cache import get_cache, put_cache

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial query router for a system that has full 10-K filings "
     "(Business, Risk Factors, MD&A, Financial Statements) for 9 companies. "
     "Every filing already contains explicit financial figures — revenue, net income, "
     "R&D expense, operating income, total assets, risk factors, business description, etc. "
     "These are directly stated in the documents and do NOT need to be computed.\n\n"
     "Classify the question into exactly one of these three routes:\n\n"
     "\"retrieve\"  — the answer is a specific fact or figure already stated somewhere "
     "in a filing. This includes any single financial metric, even ones that sound "
     "like they require math (e.g. revenue, net income, R&D expense) — these are "
     "reported numbers, not numbers you compute.\n"
     "\"calculate\" — answering requires performing arithmetic ON TOP OF numbers "
     "from the filings — e.g. computing a ratio, a percentage change, a growth rate, "
     "or comparing derived values ACROSS companies (\"highest\", \"lowest\", \"which company\").\n"
     "\"direct\"    — a general finance/accounting concept question that needs no "
     "document lookup at all (e.g. defining a term).\n\n"
     "Examples:\n"
     "Q: What was Apple's revenue in 2024? -> retrieve (revenue is a stated figure)\n"
     "Q: What was Apple's R&D expense? -> retrieve (stated figure)\n"
     "Q: What was the YoY revenue growth rate for Amazon? -> calculate (requires computing % change from two stated numbers)\n"
     "Q: Which company had the highest gross profit margin? -> calculate (requires computing margin, then comparing across companies)\n"
     "Q: What does EBITDA stand for? -> direct (general definition, no document needed)\n\n"
     "Return ONLY the single word. No explanation. No punctuation."),
    ("human", "{question}")
])


company_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are extracting company names mentioned in a question, so a financial "
     "search system knows which company's 10-K filing to search. "
     "Only these 9 companies exist in the system: "
     "Apple, Microsoft, Amazon, NVIDIA, Tesla, Meta, Alphabet, Netflix, Adobe.\n\n"
     "Alias: 'Google' and 'Alphabet' both refer to the same company — return "
     "\"Google\" if the user wrote Google, \"Alphabet\" if the user wrote Alphabet. "
     "Either spelling is accepted and will resolve to the same filing.\n\n"
     "If the question names one or more of these companies (by name, ticker, or "
     "clear reference), return a JSON array of exactly those company names, "
     "using the exact spelling the user wrote (from the list above, including "
     "the Google alias).\n"
     "If the question does not mention any specific company from the list "
     "(e.g. it asks about \"which company\" or makes no company reference at all), "
     "return [\"all\"].\n\n"
     "Examples:\n"
     "Q: What was Apple's revenue? -> [\"Apple\"]\n"
     "Q: Compare Apple and Microsoft's margins -> [\"Apple\", \"Microsoft\"]\n"
     "Q: Which of Apple, Google, and Amazon had the highest R&D? -> [\"Apple\", \"Google\", \"Amazon\"]\n"
     "Q: Which company had the highest revenue? -> [\"all\"]\n"
     "Q: What does EBITDA mean? -> [\"all\"]\n\n"
     "Return ONLY the JSON array. No explanation, no markdown formatting."),
    ("human", "{question}")
])

grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are grading whether retrieved document excerpts contain enough "
     "information to answer a financial question. The excerpts may be prose "
     "paragraphs or financial tables (marked with | characters).\n\n"
     "IMPORTANT: Grade the excerpts against the question EXACTLY AS ASKED. "
     "Do not reinterpret, reword, or assume the question meant something else "
     "just because the excerpts are about a different company or topic. "
     "If the question asks about Company A but the excerpts are about "
     "Company B, the answer is \"no\" — even if the excerpts fully answer "
     "the equivalent question for Company B.\n\n"
     "This question requires information about the following companies: "
     "{companies}.\n\n"
     "MULTI-COMPANY RULE: if more than one company is listed above, check "
     "EACH company individually. Answer \"yes\" ONLY if the excerpts "
     "contain the specific fact or figure needed for EVERY company "
     "listed — not most of them, not the majority, not \"enough to "
     "attempt an answer.\" If even ONE listed company is missing its "
     "specific figure, the answer is \"no,\" even if every other company "
     "is fully covered.\n\n"
     "First, briefly reason company-by-company: for each company listed, "
     "state whether its specific figure is present or missing in the "
     "excerpts. Excerpts that only mention the topic in passing (e.g. a "
     "table of contents entry, or a table from that company on a "
     "completely unrelated topic) do not count as containing the answer.\n\n"
     "Then, on the FINAL line of your response, write ONLY the word "
     "\"yes\" or \"no\" — nothing else on that line."),
    ("human",
     "Question: {question}\n\n"
     "Retrieved excerpts:\n{chunks}")
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are reformulating a financial question that failed to retrieve "
     "useful information from a document search. The original question may "
     "use informal language, slang, or idioms instead of standard financial "
     "terminology. Identify the actual financial concept being asked about "
     "and reword it using standard financial statement terminology and "
     "line-item names (e.g. \"net sales\", \"total revenue\", \"operating "
     "income\") — do not translate informal words literally.\n\n"
     "CRITICAL: Do not invent or assume any details that were not in the "
     "original question — no specific dates, fiscal years, note numbers, "
     "section names, or figures unless the original question already stated "
     "them. If the original question did not specify a year, do not add one. "
     "Keep the SAME underlying question — only clarify the wording.\n\n"
     "Return ONLY the rewritten question. No explanation."),
    ("human", "Original question: {question}")
])

generate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst assistant. Answer the question using "
     "ONLY the provided context from 10-K filings. Do not use any "
     "information from your training data for specific numbers, dates, "
     "or financial figures. If the context does not contain enough "
     "information to answer fully, say so clearly rather than guessing. "
     "Cite which company and section each figure comes from."
     "{conversation_context_text}"),
    ("human", "Question: {question}\nContext: {context}"),
])


def generate_node(state: GraphState) -> dict:
    question = state["rewritten_question"] or state["question"]
    context = _format_context(state["retrieved_chunks"], state["chunk_sources"])

    conv_context = state.get("conversation_context")
    if conv_context:
        conv_text = (
            "\n\nHere are the last few Q&A turns for conversational context "
            "(use this to understand the flow and adopt a natural conversational "
            "tone, but DO NOT use it for factual numbers — rely only on the "
            "provided 10-K Context for facts):\n"
            f"{conv_context}"
        )
    else:
        conv_text = ""

    llm = ChatNVIDIA(model=config.MODEL_GENERATOR, temperature=0.0)
    chain = generate_prompt | llm

    response = chain.invoke({
        "question": question,
        "context": context,
        "conversation_context_text": conv_text
    })
    answer = response.content.strip()

    if not state.get("cache_hit") and state.get("relevant") == "yes":
        vs = get_vectorstore()
        emb = vs.embeddings.embed_query(question)
        put_cache(
            route=state["route"],
            companies=state["companies_mentioned"],
            query_embedding=emb,
            resolved_question=question,
            retrieved_chunks=state["retrieved_chunks"],
            chunk_sources=state["chunk_sources"],
            grade_result=state["relevant"]
        )

    return {"answer": answer}


direct_answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst assistant. Answer this general finance "
     "or accounting question using your own knowledge — no company-specific "
     "10-K data is provided for this question. Keep the answer concise and "
     "accurate. If the question actually requires a specific company's "
     "figures (a real number, date, or value from a filing) rather than a "
     "general concept, say so rather than guessing at a company-specific "
     "figure."),
    ("human", "Question: {question}"),
])


def direct_answer_node(state: GraphState) -> dict:
    question = state["question"]

    llm = ChatNVIDIA(model=config.MODEL_GENERATOR, temperature=0.0)
    chain = direct_answer_prompt | llm

    response = chain.invoke({"question": question})
    answer = response.content.strip()

    return {"answer": answer, "final_answer": answer}


calculator_extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract the exact numbers needed to answer a financial "
     "calculation question, using ONLY the provided source chunks — never "
     "numbers from memory or training data. "
     "Identify the single arithmetic operation needed: one of "
     "'percent_change', 'difference', 'sum', 'average', 'ratio', 'margin', "
     "'max', 'min'.\n\n"
     "OPERATION SELECTION RULES:\n"
     "- When comparing two PERCENTAGE-based metrics (margins, growth "
     "rates, percentages of revenue, etc.), use 'difference' to get the "
     "point gap — NEVER 'ratio' of two percentages; a ratio of two "
     "percentages is not a meaningful financial comparison.\n"
     "- When comparing two DOLLAR amounts (revenue, R&D spend, net "
     "income, etc.), 'ratio' (how many times larger) or 'difference' "
     "(dollar gap) are both valid — pick whichever the question implies "
     "('how many times' -> ratio; 'by how much' -> difference).\n\n"
     "CONSOLIDATED FIGURE RULE: if a company reports multiple segment- or "
     "product-level breakdowns of a metric (e.g. automotive margin vs. "
     "energy margin) rather than one total company-wide figure, ALWAYS "
     "prefer the CONSOLIDATED total (e.g. 'Total revenues' and 'Total "
     "gross profit') over any single segment's figure. If the "
     "consolidated percentage isn't directly stated but the consolidated "
     "dollar totals ARE present, extract those totals and use the "
     "'margin' operation to compute it — never substitute an unlabeled "
     "segment percentage for the company-wide total.\n\n"
     "MISSING COMPANY RULE: if the question involves multiple companies "
     "and the chunks do NOT contain the specific figure for one or more "
     "of them, do not silently omit that company. Include it in 'values' "
     "with value 0 and a label ending in ' (not found in retrieved "
     "chunks)', so the gap is visible rather than hidden.\n\n"
     "ORDERING RULES:\n"
     "For 'percent_change', 'difference', and 'ratio', list the "
     "BASE/OLDER/DENOMINATOR value first, NEW/COMPARISON/NUMERATOR "
     "second.\n"
     "For 'margin', list the base value (e.g. total revenue) first, the "
     "amount to subtract (e.g. total cost of revenue) second.\n"
     "For 'max' and 'min', list every value being compared — not just two.\n\n"
     "Respond with ONLY a JSON object, no other text, in this exact shape: "
     '{{"operation": "percent_change", "values": '
     '[{{"label": "Apple FY2023 revenue", "value": 383285000000}}, '
     '{{"label": "Apple FY2024 revenue", "value": 391035000000}}]}}. '
     "If the source chunks do not contain the specific numbers needed for "
     'ANY of the companies involved, respond with '
     '{{"operation": "insufficient_data", "values": []}}.'),
    ("human", "Question: {question}\nSource chunks: {chunks}"),
])


def calculator_node(state: GraphState) -> dict:
    question = state["rewritten_question"] or state["question"]
    context = _format_context(state["retrieved_chunks"], state["chunk_sources"])

    llm = ChatNVIDIA(model=config.MODEL_GENERATOR, temperature=0.0)
    chain = calculator_extract_prompt | llm

    response = chain.invoke({"question": question, "chunks": context})
    extraction = parse_calculation(response.content)

    if extraction["operation"] == "insufficient_data" or not extraction["values"]:
        answer = (
            "I found relevant chunks, but couldn't extract the specific "
            "numbers needed to perform this calculation. Try rephrasing "
            "with more specific terms."
        )
        return {"answer": answer}

    missing = [v for v in extraction["values"] if "(not found in retrieved chunks)" in v["label"]]
    found = [v for v in extraction["values"] if v not in missing]

    # Exclude placeholders from max/min so a 0 never wins — but still disclose them
    compute_values = found if (extraction["operation"] in ("max", "min") and missing) else extraction["values"]

    if not compute_values:
        return {"answer": "Couldn't find the needed figures for any company in the retrieved chunks."}

    try:
        result = compute(extraction["operation"], compute_values)
        labels = ", ".join(f"{v['label']} = {v['value']:,}" for v in compute_values)
        op_name = extraction["operation"].replace("_", " ")

        if extraction["operation"] in ("max", "min"):
            answer = f"Comparing {labels}, the {op_name} is {result['label']} at {result['value']:,}."
        else:
            answer = f"Using {labels}, the {op_name} is {result}."

        if missing:
            missing_names = ", ".join(v["label"].replace(" (not found in retrieved chunks)", "") for v in missing)
            answer += f" Note: figures for {missing_names} were not found in the retrieved chunks and are excluded from this comparison."
    except (ValueError, IndexError, KeyError, ZeroDivisionError) as e:
        answer = f"Could not complete the calculation: {e}"

    if not state.get("cache_hit") and state.get("relevant") == "yes" and "Could not complete" not in answer and "Couldn't find" not in answer:
        vs = get_vectorstore()
        emb = vs.embeddings.embed_query(question)
        put_cache(
            route=state["route"],
            companies=state["companies_mentioned"],
            query_embedding=emb,
            resolved_question=question,
            retrieved_chunks=state["retrieved_chunks"],
            chunk_sources=state["chunk_sources"],
            grade_result=state["relevant"]
        )

    return {"answer": answer}


hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a hallucination detector for financial answers. Given an "
     "answer and the source chunks it was generated from, verify that "
     "every specific number, percentage, date, and financial figure in "
     "the answer is directly traceable to the source chunks. "
     "First, briefly reason through each specific figure in the answer "
     "and whether it appears in the source chunks. Then, on a new final "
     "line by itself, write exactly one word: 'grounded' if every "
     "specific claim in the answer exists in the chunks, or "
     "'not_grounded' if the answer contains any figure not present in "
     "the chunks."),
    ("human", "Answer: {answer}\nSource chunks: {chunks}"),
])


def hallucination_check_node(state: GraphState) -> dict:
    answer = state["answer"]
    context = _format_context(state["retrieved_chunks"], state["chunk_sources"])

    llm = ChatNVIDIA(model=config.MODEL_HALLUC, temperature=0.0)
    chain = hallucination_prompt | llm

    response = chain.invoke({"answer": answer, "chunks": context})
    grounded = parse_hallucination(response.content)

    if grounded == "grounded":
        return {"grounded": grounded, "final_answer": answer}

    # not grounded — increment the shared retry counter (PRD 12.2)
    return {"grounded": grounded, "retry_count": state["retry_count"] + 1}


def grade_exhausted_warning_node(state: GraphState) -> dict:
    """
    Runs only when Grade said 'no' but retries are exhausted (PRD 12.1
    row 1). Lets the pipeline continue to Generate/Calculator with the
    best available chunks, but records a warning rather than silently
    proceeding as if retrieval succeeded.
    """
    return {
        "error_message": (
            "Answer generated from best available context. Retrieval "
            "confidence was low — verify with source document."
        )
    }


def hallucination_exhausted_node(state: GraphState) -> dict:
    """
    Runs when Hallucination Check has failed MAX_RETRY times (PRD 12.1
    row 2). Returns an honest failure message as final_answer instead of
    the unverified/possibly-hallucinated answer.
    """
    message = (
        "Could not generate a verified answer for this question. The "
        "model was unable to produce an answer grounded in the source "
        "documents. Try rephrasing or asking a more specific question."
    )
    return {"final_answer": message, "error_message": message}

def _format_context(retrieved_chunks: list[str], chunk_sources: list[dict]) -> str:
    """
    Build a labeled context string for Generate, grouping each chunk under
    an explicit company/section header. Prevents cross-company number
    contamination in the LLM's answer (see PRD Ch 19 risk table).
    """
    labeled_sections = []
    for chunk_text, source in zip(retrieved_chunks, chunk_sources):
        company = source.get("company", "UNKNOWN COMPANY")
        section = source.get("section_name", "UNKNOWN SECTION")
        header = f"=== {company.upper()} — {section} ==="
        labeled_sections.append(f"{header}\n{chunk_text}")
    return "\n\n".join(labeled_sections)




def rewrite_node(state: GraphState) -> dict:
    llm = ChatNVIDIA(
        model=config.MODEL_REWRITE,
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
        temperature=0.0
    )

    chain = rewrite_prompt | llm
    response = chain.invoke({"question": state["question"]})

    rewritten = response.content.strip()

    return {
        "rewritten_question": rewritten,
        "retry_count": state["retry_count"] + 1
    }


def grade_node(state: GraphState) -> dict:
    llm = ChatNVIDIA(
        model=config.MODEL_GRADER,
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
        temperature=0.0
    )

    chunks_text = "\n\n---\n\n".join(state["retrieved_chunks"])
    active_question = state["rewritten_question"] or state["question"]

    companies = state.get("companies_mentioned") or []
    if companies == ["all"]:
        companies_list = get_all_full_names()
    else:
        companies_list = [SHORT_TO_FULL.get(c, c) for c in companies]
    companies_str = ", ".join(companies_list) if companies_list else "the company in question"

    chain = grade_prompt | llm
    response = chain.invoke({
        "question": active_question,
        "chunks": chunks_text,
        "companies": companies_str
    })

    relevant = parse_grade(response.content)

    return {"relevant": relevant}

def router_node(state: GraphState) -> dict:
    llm = ChatNVIDIA(
        model=config.MODEL_ROUTER,
        api_key=config.NVIDIA_API_KEY,
        base_url=config.NVIDIA_BASE_URL,
        temperature=0.0
    )

    route_chain = router_prompt | llm
    route_response = route_chain.invoke({"question": state["question"]})
    route = parse_route(route_response.content)

    company_chain = company_extraction_prompt | llm
    company_response = company_chain.invoke({"question": state["question"]})
    companies_mentioned = parse_companies(company_response.content)

    return {"route": route, "companies_mentioned": companies_mentioned}

def _chunks_to_state(docs: list) -> tuple[list[str], list[dict]]:
    """Convert Chroma Document objects to parallel (texts, sources) lists."""
    texts = [doc.page_content for doc in docs]
    sources = [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "company": doc.metadata.get("company"),
            "ticker": doc.metadata.get("ticker"),
            "section_name": doc.metadata.get("section_name"),
            "chunk_type": doc.metadata.get("chunk_type"),
            "table_name": doc.metadata.get("table_name"),
        }
        for doc in docs
    ]
    return texts, sources


_RRF_K = 60  # standard RRF constant (Ch 21.2)


def _rrf_merge(
    vector_docs: list,
    bm25_results: list[dict],
    final_k: int,
) -> tuple[list[str], list[dict]]:
    """
    Merge per-company vector and BM25 results via Reciprocal Rank Fusion.

    RRF_score(chunk) = 1/(60 + rank_in_vector) + 1/(60 + rank_in_bm25)

    A chunk surfaced by only one method still receives a score — it isn't
    dropped for lack of overlap.  The merged list is truncated to `final_k`.

    Args:
        vector_docs:  Chroma Document objects, already ranked (index 0 = best).
        bm25_results: Output of bm25_query(), already ranked (rank key is
                      1-based, but we use list position for RRF).
        final_k:      How many chunks to return after merging.

    Returns:
        (texts, sources) parallel lists ready to extend into GraphState.
    """
    # Index vector docs by chunk_id for dedup and score accumulation
    # chunk_id is the canonical key; text + metadata travel with it.
    scores: dict[str, float] = {}
    texts_map: dict[str, str] = {}
    sources_map: dict[str, dict] = {}

    # --- vector contributions (1-based rank = list index + 1) ---
    for rank_0, doc in enumerate(vector_docs):
        cid = doc.metadata.get("chunk_id", f"vec_{rank_0}")
        rrf = 1.0 / (_RRF_K + rank_0 + 1)
        scores[cid] = scores.get(cid, 0.0) + rrf
        texts_map[cid] = doc.page_content
        sources_map[cid] = {
            "chunk_id":    doc.metadata.get("chunk_id"),
            "company":     doc.metadata.get("company"),
            "ticker":      doc.metadata.get("ticker"),
            "section_name": doc.metadata.get("section_name"),
            "chunk_type":  doc.metadata.get("chunk_type"),
            "table_name":  doc.metadata.get("table_name"),
        }

    # --- BM25 contributions (result["rank"] is already 1-based) ---
    for result in bm25_results:
        cid = result["metadata"].get("chunk_id", result["id"])
        rrf = 1.0 / (_RRF_K + result["rank"])
        scores[cid] = scores.get(cid, 0.0) + rrf
        if cid not in texts_map:
            texts_map[cid] = result["text"]
            meta = result["metadata"]
            sources_map[cid] = {
                "chunk_id":    meta.get("chunk_id"),
                "company":     meta.get("company"),
                "ticker":      meta.get("ticker"),
                "section_name": meta.get("section_name"),
                "chunk_type":  meta.get("chunk_type"),
                "table_name":  meta.get("table_name"),
            }

    # Sort descending by RRF score, truncate to final_k
    top_ids = sorted(scores, key=lambda c: scores[c], reverse=True)[:final_k]

    texts   = [texts_map[c]   for c in top_ids]
    sources = [sources_map[c] for c in top_ids]
    return texts, sources


def cache_lookup_node(state: GraphState) -> dict:
    route = state["route"]
    companies = state["companies_mentioned"]
    question = state["rewritten_question"] or state["question"]
    
    if route not in ["retrieve", "calculate"]:
        return {"cache_hit": False}
        
    vs = get_vectorstore()
    query_embedding = vs.embeddings.embed_query(question)
    
    cache_result = get_cache(route, companies, query_embedding)
    if cache_result:
        retrieved_chunks, chunk_sources, grade_result = cache_result
        print(f"CACHE HIT: {route} {companies}")
        return {
            "cache_hit": True,
            "retrieved_chunks": retrieved_chunks,
            "chunk_sources": chunk_sources,
            "relevant": grade_result
        }
        
    return {"cache_hit": False}


def retrieve_node(state: GraphState) -> dict:
    """
    Hybrid retrieval: vector search (Chroma) + BM25 search, merged per company
    via Reciprocal Rank Fusion.  Final chunk counts are unchanged from the
    pure-vector version (Ch 09.2 / Ch 21.4):
      - single company : 5 chunks
      - two-company    : 4 chunks per company  (8 total)
      - ["all"]        : 4 chunks per company  (36 total)
    """
    vectorstore = get_vectorstore()
    query = state["rewritten_question"] or state["question"]
    companies = state["companies_mentioned"]

    all_texts:   list[str]  = []
    all_sources: list[dict] = []

    if companies == ["all"]:
        for full_name in get_all_full_names():
            vec_docs = vectorstore.similarity_search(
                query, k=4, filter={"company": full_name}
            )
            bm25_res = bm25_query(query, full_name, top_k=20)
            texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=4)
            all_texts.extend(texts)
            all_sources.extend(sources)

    elif len(companies) == 1:
        full_name = SHORT_TO_FULL.get(companies[0])
        if full_name:
            vec_docs = vectorstore.similarity_search(
                query, k=5, filter={"company": full_name}
            )
            bm25_res = bm25_query(query, full_name, top_k=20)
            texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=5)
            all_texts.extend(texts)
            all_sources.extend(sources)

    else:  # two or more specific companies
        for short_name in companies:
            full_name = SHORT_TO_FULL.get(short_name)
            if full_name:
                vec_docs = vectorstore.similarity_search(
                    query, k=4, filter={"company": full_name}
                )
                bm25_res = bm25_query(query, full_name, top_k=20)
                texts, sources = _rrf_merge(vec_docs, bm25_res, final_k=4)
                all_texts.extend(texts)
                all_sources.extend(sources)

    return {"retrieved_chunks": all_texts, "chunk_sources": all_sources}

