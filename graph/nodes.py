import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

import config
from tools.output_parsers import parse_route, parse_companies
from graph.state import create_initial_state, GraphState
from tools.vectorstore import get_vectorstore
from tools.company_names import SHORT_TO_FULL, get_all_full_names
from tools.output_parsers import parse_route, parse_companies, parse_grade

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
     "If the question names one or more of these companies (by name, ticker, or "
     "clear reference), return a JSON array of exactly those company names, "
     "using the exact spelling from the list above.\n"
     "If the question does not mention any specific company from the list "
     "(e.g. it asks about \"which company\" or makes no company reference at all), "
     "return [\"all\"].\n\n"
     "Examples:\n"
     "Q: What was Apple's revenue? -> [\"Apple\"]\n"
     "Q: Compare Apple and Microsoft's margins -> [\"Apple\", \"Microsoft\"]\n"
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
     "First, briefly reason about whether the excerpts contain the SPECIFIC "
     "fact, figure, or explanation needed to answer the question AS WRITTEN. "
     "Excerpts that only mention the topic in passing (e.g. a table of "
     "contents listing a section name) do not count as containing the answer.\n\n"
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
     "Cite which company and section each figure comes from."),
    ("human", "Question: {question}\nContext: {context}"),
])

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

    llm = ChatNVIDIA(model=config.MODEL_GENERATE, temperature=0.0)
    chain = direct_answer_prompt | llm

    response = chain.invoke({"question": question})
    answer = response.content.strip()

    return {"answer": answer, "final_answer": answer}

def hallucination_check_node(state: GraphState) -> dict:
    answer = state["answer"]
    chunks = "\n\n".join(state["retrieved_chunks"])

    llm = ChatNVIDIA(model=config.MODEL_HALLUCINATION, temperature=0.0)
    chain = hallucination_prompt | llm

    response = chain.invoke({"answer": answer, "chunks": chunks})
    grounded = parse_hallucination(response.content)

    update = {"grounded": grounded}
    if grounded == "grounded":
        update["final_answer"] = answer
    return update

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


def generate_node(state: GraphState) -> dict:
    question = state["rewritten_question"] or state["question"]
    context = _format_context(state["retrieved_chunks"], state["chunk_sources"])

    llm = ChatNVIDIA(model=config.MODEL_GENERATE, temperature=0.0)
    chain = generate_prompt | llm

    response = chain.invoke({"question": question, "context": context})
    answer = response.content.strip()

    return {"answer": answer}

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

    chain = grade_prompt | llm
    response = chain.invoke({
        "question": active_question,
        "chunks": chunks_text
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


def retrieve_node(state: GraphState) -> dict:
    vectorstore = get_vectorstore()
    query = state["rewritten_question"] or state["question"]
    companies = state["companies_mentioned"]

    all_docs = []

    if companies == ["all"]:
        for full_name in get_all_full_names():
            docs = vectorstore.similarity_search(query, k=2, filter={"company": full_name})
            all_docs.extend(docs)

    elif len(companies) == 1:
        full_name = SHORT_TO_FULL.get(companies[0])
        if full_name:
            all_docs = vectorstore.similarity_search(query, k=5, filter={"company": full_name})

    else:
        for short_name in companies:
            full_name = SHORT_TO_FULL.get(short_name)
            if full_name:
                docs = vectorstore.similarity_search(query, k=4, filter={"company": full_name})
                all_docs.extend(docs)

    texts, sources = _chunks_to_state(all_docs)

    return {"retrieved_chunks": texts, "chunk_sources": sources}

if __name__ == "__main__":
    print("=== Router tests ===")
    test_questions = [
        "What does EBITDA mean?",
        "Apple revenue?",
        "Highest gross margin?",
        "What was Apple's R&D expense in 2024?",
        "Which company had the highest revenue?",
    ]

    for q in test_questions:
        state = create_initial_state(q)
        result = router_node(state)
        print(f"question={q!r} -> route={result['route']!r} companies={result['companies_mentioned']!r}")

    print("\n=== Retrieve test: single company ===")
    single_state = create_initial_state("What was Apple's total revenue in fiscal year 2024?")
    single_state["companies_mentioned"] = ["Apple"]
    single_result = retrieve_node(single_state)
    print(f"Retrieved {len(single_result['retrieved_chunks'])} chunks")
    for src in single_result["chunk_sources"]:
        print(f"  {src}")

    print("\n=== Retrieve test: all companies ===")
    all_state = create_initial_state("Which company had the highest revenue?")
    all_state["companies_mentioned"] = ["all"]
    all_result = retrieve_node(all_state)
    print(f"Retrieved {len(all_result['retrieved_chunks'])} chunks")
    tickers_seen = [src["ticker"] for src in all_result["chunk_sources"]]
    print(f"Tickers present: {sorted(set(tickers_seen))}")
    print(f"Chunk count per ticker: { {t: tickers_seen.count(t) for t in set(tickers_seen)} }")

    print("\n=== Retrieve + Grade test: good match ===")
    good_state = create_initial_state("What was Apple's total revenue in fiscal year 2024?")
    good_state["companies_mentioned"] = ["Apple"]
    good_state.update(retrieve_node(good_state))
    good_state.update(grade_node(good_state))
    print(f"relevant={good_state['relevant']!r} (expecting 'yes')")

    print("\n=== Retrieve + Grade test: bad match ===")
    bad_state = create_initial_state("What is Apple's stock ticker symbol on the Nasdaq exchange?")
    bad_state["companies_mentioned"] = ["Tesla"]
    bad_state.update(retrieve_node(bad_state))
    bad_state.update(grade_node(bad_state))
    print(f"relevant={bad_state['relevant']!r} (expecting 'no' — Tesla chunks can't answer an Apple ticker question)")

    print("\n=== Retrieve + Grade test: second good match (different company/metric) ===")
    good_state2 = create_initial_state("What was NVIDIA's R&D expense?")
    good_state2["companies_mentioned"] = ["NVIDIA"]
    good_state2.update(retrieve_node(good_state2))
    good_state2.update(grade_node(good_state2))
    print(f"relevant={good_state2['relevant']!r} (expecting 'yes')")

    print("\n=== Retrieve + Grade test: second bad match (real company, wrong topic) ===")
    bad_state2 = create_initial_state("What is Apple's CEO's annual salary?")
    bad_state2["companies_mentioned"] = ["Apple"]
    bad_state2.update(retrieve_node(bad_state2))
    bad_state2.update(grade_node(bad_state2))
    print(f"relevant={bad_state2['relevant']!r} (expecting 'no' — executive compensation isn't in Item 8 financials we retrieve)")
    
    print("\n=== Rewrite test: oddly-phrased question triggers retry ===")
    vague_state = create_initial_state("How much cheddar did Apple rake in last year?")
    vague_state["companies_mentioned"] = ["Apple"]
    vague_state.update(retrieve_node(vague_state))
    vague_state.update(grade_node(vague_state))
    print(f"Original question: {vague_state['question']!r}")
    print(f"First-pass relevant: {vague_state['relevant']!r}")

    if vague_state["relevant"] == "no":
        vague_state.update(rewrite_node(vague_state))
        print(f"Rewritten question: {vague_state['rewritten_question']!r}")
        print(f"retry_count: {vague_state['retry_count']}")

        vague_state.update(retrieve_node(vague_state))
        print("Second-pass retrieved chunks:")
        for src in vague_state["chunk_sources"]:
            print(f"  {src}")
        vague_state.update(grade_node(vague_state))
        print(f"Second-pass relevant (after rewrite): {vague_state['relevant']!r}")
    else:
        print("First pass already succeeded — even slang retrieved correctly, try something else")