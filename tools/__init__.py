# tools/__init__.py
# Backward-compatibility shim using sys.modules aliasing.
#
# Python resolves "from tools.session_store import X" by looking for either:
#   a) a file at tools/session_store.py, OR
#   b) a key "tools.session_store" in sys.modules
#
# Since the files have moved into sub-packages, we register aliases here so
# all existing import statements in api.py, graph/nodes.py, graph/graph.py, etc.
# continue to work without any changes.

import sys
import tools.retrieval.vectorstore as _vectorstore
import tools.retrieval.bm25_index as _bm25_index
import tools.retrieval.calculator as _calculator
import tools.retrieval.output_parsers as _output_parsers
import tools.retrieval.company_names as _company_names
import tools.session.session_store as _session_store
import tools.session.context_resolver as _context_resolver
import tools.session.retrieval_cache as _retrieval_cache

# Register flat-name aliases so "from tools.X import Y" resolves correctly
sys.modules.setdefault("tools.vectorstore", _vectorstore)
sys.modules.setdefault("tools.bm25_index", _bm25_index)
sys.modules.setdefault("tools.calculator", _calculator)
sys.modules.setdefault("tools.output_parsers", _output_parsers)
sys.modules.setdefault("tools.company_names", _company_names)
sys.modules.setdefault("tools.session_store", _session_store)
sys.modules.setdefault("tools.context_resolver", _context_resolver)
sys.modules.setdefault("tools.retrieval_cache", _retrieval_cache)

