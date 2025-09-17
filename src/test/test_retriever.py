from retrieval.retriever import Retriever


def test_add_documents_and_search_basic():
    retriever = Retriever()
    docs = ["apple", "banana", "car"]
    retriever.add_documents(docs)
    # Search for "fruit" should return "apple" and "banana" (closest vectors)
    results = retriever.search("fruit", top_k=2)
    assert set(results) == {"apple", "banana"}


def test_search_returns_top_k():
    retriever = Retriever()
    docs = ["apple", "banana", "car", "vehicle"]
    retriever.add_documents(docs)
    results = retriever.search("car", top_k=1)
    assert results == ["car"]


def test_add_documents_extends_docs():
    retriever = Retriever()
    retriever.add_documents(["apple"])
    retriever.add_documents(["banana"])
    assert retriever.docs == ["apple", "banana"]


def test_search_with_no_documents():
    retriever = Retriever()
    results = retriever.search("fruit", top_k=1)
    assert results == []


def test_search_top_k_greater_than_docs():
    retriever = Retriever()
    retriever.add_documents(["apple"])
    results = retriever.search("apple", top_k=5)
    assert results == ["apple"]
