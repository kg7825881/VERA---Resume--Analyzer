"""Canonical skill names used by deterministic policy and evidence matching.

This intentionally contains only recruiter-approved equivalences, rather than
fuzzy synonyms.  It prevents spelling and vendor-prefix variation from
lowering a candidate's score while keeping distinct skills distinct.
"""
from __future__ import annotations

import re


_ALIASES = {
    "Java": ("java", "java se", "java ee", "JAVA"),
    "Kafka": ("kafka", "apache kafka"),
    "API design": (
        "api design", "api development", "rest api", "rest apis",
        "restful api", "restful apis", "restful services", "rest services",
    ),
    "Spring Boot": ("spring boot",),
    # Singular/plural wording and a framework-qualified phrase are still
    # explicit evidence of the named reliability mechanism.  This does not
    # make the broader resiliency capability equivalent to a circuit breaker.
    "Circuit breakers": ("circuit breaker", "circuit breakers", "resilience4j circuit breaker"),
    "Code review": ("code review", "code reviews"),
    "SQL": ("sql",),
    "Microservices": ("microservices", "microservice architecture"),
    "Distributed systems": ("distributed systems", "distributed system"),
    "AWS": ("aws", "amazon web services"),
    "Google Cloud": ("google cloud", "gcp", "google cloud platform"),
    "GenAI": ("genai", "generative ai", "generative artificial intelligence"),
    "LLMs": ("llm", "llms", "large language model", "large language models"),
    "RAG": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "MCP": ("mcp", "model context protocol"),
    "CI/CD": ("ci/cd", "cicd", "continuous integration continuous delivery"),
    "Infrastructure as code": ("infrastructure as code", "iac"),
    # Vendor-capitalization variants are the same hosting platform.  GitHub
    # Copilot intentionally remains a separate capability and is not listed
    # here as an alias for GitHub.
    "GitHub": ("github",),
    "GitLab": ("gitlab",),
    # These are wording variants of a capability, not claims that a distinct
    # product was used.  They let a JD's compact label match an explicitly
    # listed resume skill such as "Document Chunking" or "AI Evaluation".
    "Backend development": ("backend development", "backend engineering", "backend services"),
    "Embeddings": ("embeddings", "embedding model", "embedding models", "vector embeddings"),
    "Vector databases": ("vector database", "vector databases", "vector db", "vector dbs"),
    "Vector search": ("vector search", "similarity search", "nearest-neighbor search"),
    "Chunking": ("chunking", "document chunking", "text chunking"),
    "Model evaluation": ("model evaluation", "ai evaluation", "model assessment"),
    "LLM evaluation": ("llm evaluation", "llm-as-a-judge", "llm as a judge"),
    "Retrieval quality": ("retrieval quality", "retrieval evaluation", "retrieval precision", "context relevance"),
    # Recruiter-reviewed equivalents for broad capabilities. These are not
    # fuzzy guesses, so each phrase is safe as direct resume evidence.
    # Only spelling, abbreviation, and truly interchangeable variants belong
    # in this exact-alias registry. Related practices (observability, OCR,
    # model training, UI work, and so on) are intentionally handled by the
    # reference policy as yellow 80% evidence instead.
    "Monitoring": ("monitoring",),
    "Document parsing": ("document parsing",),
    "Unstructured data": ("unstructured data",),
    "Document processing": ("document processing",),
    "Document-heavy data": ("document-heavy data", "document heavy data"),
    "Machine learning": ("machine learning", "ml"),
    "Logging": ("logging",),
    "Full-stack development": ("full-stack development", "full stack development", "full-stack", "full stack"),
    "Frontend development": ("frontend development", "front-end development"),
    # Broad, role-independent capability labels. These aliases describe the
    # same practice; individual tools remain distinct named evidence.
    "Data engineering": ("data engineering",),
    "Data pipelines": ("data pipelines", "data pipeline", "pipeline development", "data workflow pipelines"),
    # ETL and ELT are deliberately treated as an approved interchangeable
    # transformation-pipeline reference.  A resume that names either has
    # explicitly evidenced the same hiring capability; it is not merely a
    # neighbouring capability-group signal.
    "ETL": (
        "etl", "extract transform load", "extract-transform-load",
        "elt", "extract load transform", "extract-load-transform",
    ),
    "Data transformation logic": ("data transformation logic",),
    "Structured and semi-structured data": ("structured and semi-structured data", "structured and semi structured data"),
    "Alerting": ("alerting", "alerts", "alert management", "alerting pipelines"),
    "Embedding pipelines": ("embedding pipelines", "embedding pipeline", "embedding workflow"),
    "Retrieval datasets": ("retrieval datasets", "retrieval dataset", "retrieval corpus", "retrieval corpora"),
    "Data modeling": ("data modeling", "data modelling"),
    "Schema design": ("schema design",),
    "Data quality": ("data quality",),
    "Data validation": ("data validation",),
    "Data governance": ("data governance",),
    "Data lineage": ("data lineage",),
    "Data warehouse": ("data warehouse", "data warehousing"),
    "Lakehouse": ("lakehouse", "data lakehouse"),
    "Business analysis": ("business analysis", "business analyst", "product analysis", "product analyst", "systems analysis", "systems analyst"),
    "Requirements elicitation": ("requirements elicitation", "requirements gathering", "requirements analysis", "business requirements"),
    "Agile delivery": ("agile delivery", "agile product delivery", "scrum delivery", "iterative delivery"),
    "Backlog management": ("backlog management", "product backlog", "backlog refinement", "backlog grooming"),
    "Process modeling": ("process modeling", "process modelling", "business process modeling", "business process modelling"),
    "Workflow design": ("workflow design", "workflow modeling", "workflow modelling", "workflow analysis"),
    "Stakeholder management": ("stakeholder management", "stakeholder engagement", "client stakeholder management"),
    "DevOps": ("devops", "dev ops", "devsecops", "platform engineering", "site reliability engineering", "sre"),
    "Kubernetes": ("kubernetes", "k8s", "eks", "aks", "gke"),
    "GitOps": ("gitops", "git ops"),
    "Deployment automation": ("deployment automation", "automated deployments", "release automation", "deployment orchestration"),
    "Cloud cost optimization": ("cloud cost optimization", "cloud cost management", "finops", "cloud cost efficiency"),
    "Disaster recovery": ("disaster recovery", "dr planning", "business continuity", "continuity planning"),
    "Platform security": ("platform security", "infrastructure security", "cloud security", "devsecops"),
}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", (value or "").casefold()).strip()


_CANONICAL_BY_ALIAS = {
    _key(alias): canonical
    for canonical, aliases in _ALIASES.items()
    for alias in (canonical, *aliases)
}


def canonical_skill(value: str) -> str:
    """Return the approved canonical label, or a cleaned original label."""
    clean = " ".join((value or "").split()).strip()
    return _CANONICAL_BY_ALIAS.get(_key(clean), clean)


def skills_equivalent(left: str, right: str) -> bool:
    """Whether two labels are the same approved skill after canonicalization."""
    return bool(left and right) and canonical_skill(left).casefold() == canonical_skill(right).casefold()


def explicit_skill_in_values(required: str, values: list[str] | set[str] | tuple[str, ...]) -> str | None:
    """Return explicitly listed evidence for a compact JD skill label.

    Resume parsers preserve useful qualifiers (for example, ``REST APIs`` and
    ``Large Language Models (LLM)``).  Matching a complete approved alias as
    a word-bounded phrase keeps that evidence explicit without permitting
    unsafe substring matches such as ``Java`` inside ``JavaScript``.
    """
    canonical = canonical_skill(required)
    aliases = (canonical, *_ALIASES.get(canonical, ()))
    for value in values or ():
        if not isinstance(value, str) or not value.strip():
            continue
        normalized_value = _key(value)
        if canonical_skill(value).casefold() == canonical.casefold():
            return value
        padded_value = f" {normalized_value} "
        for alias in aliases:
            normalized_alias = _key(alias)
            if normalized_alias and f" {normalized_alias} " in padded_value:
                return value
    return None
