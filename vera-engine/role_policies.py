"""Versioned, conservative role defaults for JD requirements.

Defaults only label requirements that the JD extractor already found. They
never inject an absent skill, and a recruiter override always wins.
"""
from __future__ import annotations

from dataclasses import dataclass


ROLE_POLICY_VERSION = "1.1"


@dataclass(frozen=True)
class DefaultGroup:
    labels: tuple[str, ...]
    alternative_group: str = ""
    non_negotiable: bool = True


_POLICIES: tuple[tuple[tuple[str, ...], tuple[DefaultGroup, ...]], ...] = (
    (("java", "lead"), (
        # These are the compact, recruiter-reviewable core for this role.
        # Everything else extracted from a Required section remains visible,
        # but is calibrated as preferred until a recruiter promotes it.
        DefaultGroup(("Java",), non_negotiable=True),
        DefaultGroup(("Spring Boot",), non_negotiable=True),
        DefaultGroup(("API design", "REST APIs", "Microservices"), "java-service-architecture", False),
        DefaultGroup(("Kafka",), non_negotiable=False), DefaultGroup(("Distributed systems",), non_negotiable=False),
        DefaultGroup(("SQL",), non_negotiable=False),
    )),
    (("ai", "genai"), (
        DefaultGroup(("Python",)), DefaultGroup(("LLMs", "GenAI"), "ai-llm-development"),
        DefaultGroup(("RAG", "Retrieval quality"), "ai-retrieval"),
        DefaultGroup(("Prompt engineering", "Model evaluation", "LLM evaluation"), "ai-evaluation"),
        DefaultGroup(("APIs", "Backend development"), "ai-product-integration"),
    )),
    (("data", "engineer"), (
        DefaultGroup(("Python",)), DefaultGroup(("SQL",)),
        DefaultGroup(("ETL", "ELT"), "data-etl-elt"), DefaultGroup(("Data modeling",)),
        DefaultGroup(("Spark", "Distributed systems"), "data-distributed-processing"),
        DefaultGroup(("Airflow", "Dagster"), "data-orchestration"),
    )),
    (("business", "analyst"), (
        DefaultGroup(("Communication",)), DefaultGroup(("SQL", "Analytics"), "ba-data-analysis"),
    )),
    (("devops",), (
        DefaultGroup(("Docker", "Kubernetes"), "devops-containers"),
        DefaultGroup(("Infrastructure as Code",)),
        DefaultGroup(("Monitoring", "Observability"), "devops-observability"),
    )),
)


def default_groups_for_role(role_title: str) -> tuple[DefaultGroup, ...]:
    """Return defaults only for the first clearly matching role family."""
    words = (role_title or "").casefold()
    for required_words, groups in _POLICIES:
        if all(word in words for word in required_words):
            return groups
    return ()
