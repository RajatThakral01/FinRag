import time
from tools.context_resolver import resolve_context

cases = [
    {
        "id": "R1",
        "desc": "Pronoun 'their' -> single company",
        "history": [{"turn_number": 1, "raw_question": "What was Apple's revenue?", "final_answer": "Apple's revenue was $383 billion.", "companies": ["Apple"]}],
        "question": "What about their R&D expense?"
    },
    {
        "id": "R2",
        "desc": "the company -> single company",
        "history": [{"turn_number": 1, "raw_question": "What was NVIDIA's revenue?", "final_answer": "NVIDIA's revenue was $60 billion.", "companies": ["NVIDIA"]}],
        "question": "How much did the company spend on R&D?"
    },
    {
        "id": "M1",
        "desc": "Metric-only follow-up",
        "history": [{"turn_number": 1, "raw_question": "What was Apple's revenue?", "final_answer": "Apple's revenue was $383 billion.", "companies": ["Apple"]}],
        "question": "And net income?"
    },
    {
        "id": "N1",
        "desc": "New company, no comparison signal",
        "history": [{"turn_number": 1, "raw_question": "What was Apple's revenue?", "final_answer": "Apple's revenue was $383 billion.", "companies": ["Apple"]}],
        "question": "What about Tesla?"
    },
    {
        "id": "C1",
        "desc": "Explicit comparison, already self-contained",
        "history": [],
        "question": "How does Tesla's revenue compare to Apple's in 2024?"
    },
    {
        "id": "B1",
        "desc": "both their after two-company history",
        "history": [{"turn_number": 1, "raw_question": "Compare Apple and Microsoft's revenue.", "final_answer": "Apple revenue was X and Microsoft was Y.", "companies": ["Apple", "Microsoft"]}],
        "question": "How do both their gross margins compare?"
    },
    {
        "id": "D1",
        "desc": "Definition / direct route",
        "history": [],
        "question": "What does gross margin mean?"
    },
    {
        "id": "Y1",
        "desc": "Implicit year shift",
        "history": [{"turn_number": 1, "raw_question": "What was Apple's revenue in 2024?", "final_answer": "Apple's revenue in 2024 was $383 billion.", "companies": ["Apple"]}],
        "question": "What about 2023?"
    }
]

def run_tests():
    for case in cases:
        print(f"\n--- Case {case['id']}: {case['desc']} ---")
        print(f"History: {case['history']}")
        print(f"Question: {case['question']}")
        start = time.time()
        try:
            resolved = resolve_context(case['question'], case['history'])
            print(f"Resolved: '{resolved}'")
        except Exception as e:
            print(f"ERROR: {e}")
        print(f"Time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    run_tests()
