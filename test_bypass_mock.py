import json
from tools.retrieval_cache import put_cache, get_cache

def test_bypass():
    companies_json = json.dumps(sorted(["Apple"]))
    print(f"Bypass 'general':")
    put_cache("retrieve", ["Apple"], "general", [0.1]*1536, "q", ["c1"], [{"s": 1}], "yes")
    print(f"  Lookup general: {get_cache('retrieve', companies_json, 'general', [0.1]*1536)}")
    
    print(f"Bypass 'revenue_sales':")
    put_cache("retrieve", ["Apple"], "revenue_sales", [0.1]*1536, "q", ["c1"], [{"s": 1}], "yes")
    print(f"  Lookup revenue: {'HIT' if get_cache('retrieve', companies_json, 'revenue_sales', [0.1]*1536) else 'MISS'}")

test_bypass()
