from jd_extractor import extract_structured_jd


def test_ai_product_engineering_jd_keeps_all_named_requirement_skills():
    """Requirement-section skills should not be lost just because they are prose."""
    jd = """
    Position: AI Engineer — GenAI Product Engineering
    Required Skills & Qualifications:
    2-4 years of experience in software engineering, AI engineering, machine learning engineering,
    or backend/application engineering with AI exposure.
    Strong Python skills and comfort working with APIs, services, and backend workflows.
    Hands-on experience building or integrating LLM-based applications.
    Familiarity with RAG concepts such as chunking, embeddings, vector search, and retrieval quality.
    Experience with frameworks or tools such as LangChain, LlamaIndex, FastAPI, or similar application-layer AI tooling.
    Understanding of how to evaluate AI output quality and improve it through experimentation.
    Preferred Qualifications:
    Exposure to vector databases such as FAISS, Pinecone, Weaviate, or Chroma.
    Experience with prompt evaluation, guardrails, output validation, or LLM testing frameworks.
    Familiarity with Docker, cloud deployment, logging, and production monitoring.
    Exposure to OCR, document parsing, or unstructured data workflows.
    """

    result = extract_structured_jd(jd)

    assert {
        "Python", "APIs", "Backend development", "LLMs", "LLM applications", "RAG",
        "Chunking", "Embeddings", "Vector search", "Retrieval quality", "LangChain",
        "LlamaIndex", "FastAPI", "Model evaluation", "LLM evaluation",
    } <= set(result["mandatory_skills"])
    assert {
        "Vector databases", "FAISS", "Pinecone", "Weaviate", "Chroma", "Prompt engineering",
        "Guardrails", "Output validation", "LLM testing", "Docker", "Cloud deployment",
        "Logging", "Monitoring", "OCR", "Document parsing", "Unstructured data",
    } <= set(result["preferred_technical_skills"])
    assert result["min_years_experience"] == 2
    assert result["max_years_experience"] == 4
