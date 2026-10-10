"""Reusable recruiter-facing capability groups for related JD skills.

The JD still records every explicitly requested tool.  Groups only control how
related tools contribute to a score: evidence for one member establishes the
capability, while the evidence view retains the named tools for review.
"""
from __future__ import annotations

import re

from skill_aliases import canonical_skill, explicit_skill_in_values


CAPABILITY_GROUP_VERSION = "1.8"

# Some group members are named vendor/tooling products rather than umbrella
# capabilities. Generic work in a capability (for example, observability,
# workflow automation, ML, or testing) must never be used to claim hands-on
# experience with a particular product. Approved deterministic relationships
# can still award yellow related credit before this guard is consulted.
_STRICT_NAMED_TOOL_MEMBERS = {
    "observability": {
        "prometheus", "grafana", "loki", "elk/opensearch", "opentelemetry",
    },
    "cloud-native-platform": {"docker", "kubernetes", "helm", "openshift", "istio"},
    "infrastructure-delivery": {
        "terraform", "ansible", "pulumi", "jenkins", "argo cd", "github actions",
        "gitlab ci", "circleci",
    },
    "big-data-processing": {
        "spark", "pyspark", "databricks", "hadoop", "hive", "flink", "beam", "delta lake",
    },
    "analytics-bi": {"power bi", "tableau", "looker"},
    "data-orchestration": {"airflow", "dagster"},
    "machine-learning-engineering": {
        "scikit-learn", "tensorflow", "pytorch", "mlflow", "sagemaker",
    },
    "rag-retrieval-systems": {
        "pinecone", "weaviate", "milvus", "faiss", "chroma", "langchain", "llamaindex",
    },
    "agentic-ai-orchestration": {"langgraph", "crew ai", "autogen"},
    "ai-developer-productivity": {
        "github copilot", "chatgpt", "claude", "cursor", "codewhisperer", "tabnine",
    },
    "testing-quality-engineering": {"selenium", "cypress", "playwright", "pytest"},
}

_GROUPS = {
    "spring-ecosystem": {
        "spring boot", "spring framework", "spring data jpa", "spring discovery",
        "spring cloud gateway", "spring batch", "hibernate",
    },
    "service-architecture": {
        "api design", "rest apis", "apis", "microservices", "distributed systems",
        "backend development",
    },
    "event-streaming-messaging": {
        "kafka", "aws sqs", "rabbitmq", "activemq", "event-driven architecture", "retry strategies",
    },
    "data-persistence": {
        "sql", "nosql", "oracle", "mysql", "postgresql", "mongodb", "redis",
    },
    "observability": {
        "monitoring", "alerting", "observability", "prometheus", "grafana", "loki",
        "elk/opensearch", "opentelemetry", "logging",
    },
    "aws-cloud-platform": {
        "aws", "amazon web services", "ec2", "s3", "lambda", "ecs", "eks", "rds", "dynamodb",
        "cloudformation", "iam", "vpc", "api gateway", "cloudwatch", "eventbridge", "step functions",
        "aws sqs", "aws sns", "aws glue", "redshift",
    },
    "azure-cloud-platform": {
        "azure", "microsoft azure", "azure functions", "azure devops", "azure kubernetes service",
        "azure data factory", "azure databricks", "azure synapse",
    },
    "google-cloud-platform": {
        "google cloud", "gcp", "google kubernetes engine", "bigquery", "cloud run", "cloud functions",
        "pub/sub", "google cloud storage", "vertex ai",
    },
    "cloud-native-platform": {
        "cloud deployment", "cloud-native architecture", "containerization", "docker", "kubernetes", "helm",
        "openshift", "service mesh", "istio",
    },
    "infrastructure-delivery": {
        "infrastructure as code", "terraform", "ansible", "pulumi", "jenkins", "ci/cd", "gitops", "argo cd",
        "github actions", "gitlab ci", "circleci", "build pipelines", "release management",
    },
    "reliability-engineering": {
        "circuit breakers", "resiliency patterns", "fault tolerance", "configuration management",
    },
    "engineering-practices": {
        "design patterns", "solid principles", "scalability", "performance engineering", "code review",
        "junit", "mockito", "maven", "gradle",
    },
    "ai-developer-productivity": {"github copilot", "chatgpt", "claude", "cursor", "codewhisperer", "tabnine"},
    "data-transformation": {
        "etl", "elt", "data transformation logic", "data modeling", "schema design",
        "data warehouse", "lakehouse", "dbt",
    },
    "big-data-processing": {
        "spark", "pyspark", "databricks", "hadoop", "hive", "flink", "beam", "delta lake",
        "data lake", "batch processing", "stream processing",
    },
    "analytics-bi": {
        "power bi", "tableau", "looker", "data visualization", "analytics", "reporting",
    },
    "data-orchestration": {"airflow", "dagster", "workflow automation"},
    "data-quality-governance": {
        "data validation", "deduplication", "consistency checks", "data quality control",
        "data quality checks", "data lineage", "data governance",
    },
    "insurance-claims-domain": {
        "property and casualty insurance", "p&c insurance", "insurance", "claims", "claims processing",
        "claims operations", "subrogation", "recovery operations", "auto claims", "property claims",
        "guidewire claimcenter", "duck creek", "origami risk",
    },
    "machine-learning-engineering": {
        "machine learning", "ml", "model training", "model serving", "feature engineering", "scikit-learn",
        "tensorflow", "pytorch", "mlflow", "sagemaker",
    },
    "llm-genai-development": {
        "llms", "large language models", "genai", "generative ai", "llm applications", "openai api",
        "azure openai", "bedrock", "hugging face", "transformers", "fine tuning", "prompt engineering",
    },
    "rag-retrieval-systems": {
        "rag", "retrieval augmented generation", "embeddings", "vector search", "vector databases",
        "semantic search", "hybrid search", "reranking", "pinecone", "weaviate", "milvus", "faiss", "chroma",
        "langchain", "llamaindex",
    },
    "agentic-ai-orchestration": {
        "ai agents", "agentic workflows", "tool calling", "function calling", "model context protocol",
        "mcp", "multi-agent systems", "langgraph", "crew ai", "autogen",
    },
    "llm-quality-safety": {
        "model evaluation", "prompt engineering", "llm evaluation", "retrieval quality", "guardrails",
        "output validation", "llm testing", "hallucination evaluation", "red teaming", "responsible ai",
    },
    "application-security": {
        "application security", "secure coding", "oauth", "openid connect", "jwt", "iam", "secrets management",
        "vulnerability management", "threat modeling", "owasp", "security testing",
    },
    "testing-quality-engineering": {
        "unit testing", "integration testing", "test automation", "selenium", "cypress", "playwright", "pytest",
        "quality assurance", "tdd", "bdd",
    },
}

_GROUP_BY_SKILL = {
    canonical_skill(skill).casefold(): group
    for group, skills in _GROUPS.items()
    for skill in skills
}


def capability_group_for_skill(skill: str) -> str:
    """Return the approved score group for a known skill, else an empty string."""
    normalized = canonical_skill(skill).casefold()
    if normalized.startswith("spring "):
        return "spring-ecosystem"
    return _GROUP_BY_SKILL.get(normalized, "")


def requires_named_tool_evidence(skill: str) -> bool:
    """Return whether a non-exact match needs an explicitly named tool.

    This protects product-specific requirements from semantic overreach. It
    intentionally does not reject a deterministic related rule: callers run
    exact and approved related evidence first, then use this guard only before
    asking a semantic model to infer a remaining match.
    """
    normalized = canonical_skill(skill).casefold()
    group = capability_group_for_skill(normalized)
    return normalized in _STRICT_NAMED_TOOL_MEMBERS.get(group, set())


def _approved_anchor_in_values(anchor: str, values: set[str]) -> bool:
    """Find an approved multiword anchor in extracted skill or source text."""
    normalized_anchor = re.sub(r"[^a-z0-9+#]+", " ", anchor.casefold()).strip()
    if not normalized_anchor:
        return False
    for value in values:
        normalized_value = re.sub(r"[^a-z0-9+#]+", " ", str(value).casefold()).strip()
        if f" {normalized_anchor} " in f" {normalized_value} ":
            return True
    return False


def partial_capability_evidence(required_skill: str, candidate_skills: set[str]) -> str | None:
    """Return a recruiter-safe reason for a related, non-exact skill match.

    This is deliberately narrower than score grouping.  A group prevents
    duplicate scoring for a related capability; it must not claim that every
    tool in that group was used.  The caller awards a fixed partial credit and
    keeps the named requirement visibly inferred.
    """
    required = canonical_skill(required_skill).casefold()
    candidates = {canonical_skill(skill).casefold() for skill in candidate_skills}

    required_group = capability_group_for_skill(required)
    strict_members = _STRICT_NAMED_TOOL_MEMBERS.get(required_group, set())
    if required in strict_members:
        sibling_products = strict_members - {required}
        if sibling_products & candidates or any(
            _approved_anchor_in_values(product, candidate_skills)
            for product in sibling_products
        ):
            return "Related named-tool capability-group evidence; the requested product is not explicitly evidenced."
        return None

    # Spring products share concepts, but practical use of one product is not
    # evidence of hands-on use of another product such as Gateway or Discovery.
    if required.startswith("spring ") and any(skill.startswith("spring ") for skill in candidates):
        return "Related Spring ecosystem evidence; the named product is not explicitly evidenced."

    partial_anchors = {
        # Docker/Kubernetes are direct evidence of the containerization
        # capability, while remaining distinct named tools.
        "containerization": {"docker", "kubernetes", "helm", "openshift"},
        # API design is exact when REST/API-design aliases normalize to it.
        # A generic plural API requirement can still receive partial credit
        # from concrete API-design evidence without becoming an exact match.
        "apis": {"api design", "openapi", "swagger", "restapi",},
        "api gateway": {"api design", "microservices"},
        # Version-control fluency does not prove use of GitHub, but it is a
        # useful interview follow-up when a JD names the hosting platform.
        "github": {"git", "gitlab"},
        # AWS familiarity is related to a named AWS service.  Keep it partial
        # until the service itself is mentioned in the resume.
        "aws sqs": {"aws", "amazon web services"},
        # These are one-way, concrete references to a broad engineering
        # practice.  They intentionally remain partial evidence: naming
        # Eureka does not claim hands-on Spring Cloud Gateway experience, for
        # example, and naming Resilience4j does not make every reliability
        # practice a direct match.
        "spring discovery": {"eureka", "netflix eureka"},
        "spring cloud gateway": {"api gateway", "spring gateway"},
        "configuration management": {"config server", "spring cloud config"},
        "resiliency patterns": {"resilience4j", "circuit breaker", "retry mechanisms", "retry mechanism"},
        "fault tolerance": {"resilience4j", "circuit breaker", "retry mechanisms", "retry mechanism"},
        "performance engineering": {"performance optimization", "performance tuning", "load testing", "throughput optimization"},
        "code review": {"code reviews", "peer review", "peer reviews"},
        "llm applications": {"generative ai", "genai", "nlp", "prompt engineering", "rag", "retrieval augmented generation", "langchain", "langgraph", "ai agents", "agentic ai", "model integration"},
        "model evaluation": {"rag testing", "rag evaluation", "evaluation", "evaluation framework", "benchmarking", "llm testing", "hallucination detection", "guardrails", "ragas", "deepeval", "langsmith"},
        "llm evaluation": {"rag testing", "rag evaluation", "evaluation", "evaluation framework", "benchmarking", "llm testing", "hallucination detection", "guardrails", "ragas", "deepeval", "langsmith"},
        "llm testing": {"rag testing", "rag evaluation", "evaluation", "evaluation framework", "benchmarking", "hallucination detection", "prompt testing", "quality assurance", "guardrails", "model evaluation", "llm evaluation"},
        "document parsing": {"ocr", "information extraction", "document intelligence", "pdf parsing", "entity recognition", "data extraction", "pymupdf", "pdf text extraction"},
        "workflow automation": {"process automation", "orchestration", "pipelines", "integrations", "rpa", "task automation", "process orchestration"},
        "full-stack development": {"frontend development", "backend development", "react", "next.js", "angular", "vue"},
    }
    anchors = partial_anchors.get(required, set())
    if anchors & candidates or any(_approved_anchor_in_values(anchor, candidate_skills) for anchor in anchors):
        return "Related capability evidence; the named tool or practice is not explicitly evidenced."

    # Non-product capability siblings may be useful related evidence. Strict
    # named products returned above are deliberately excluded from this broad
    # fallback, so a generic group never invents a vendor/tool claim.
    candidate_groups = {capability_group_for_skill(skill) for skill in candidates}
    if required_group and required_group in candidate_groups:
        return "Related capability-group evidence; the named tool or practice is not explicitly evidenced."

    group_members = _GROUPS.get(required_group, set())
    if required_group and any(
        _approved_anchor_in_values(member, candidate_skills)
        for member in group_members
        if canonical_skill(member).casefold() != required
    ):
        return "Related capability-group evidence; the named tool or practice is not explicitly evidenced."
    return None


def explicit_capability_evidence(required_skill: str, candidate_skills: set[str]) -> str | None:
    """Return a reason when named tools explicitly prove a broader capability.

    These are one-way relationships: Docker proves containerization, but a
    generic containerization claim must never be rewritten as Docker.  This
    keeps tool-level evidence truthful while recognizing established umbrella
    capabilities across all roles.
    """
    required = canonical_skill(required_skill).casefold()
    candidates = {canonical_skill(skill).casefold() for skill in candidate_skills}
    direct_anchors = {
        "containerization": {"docker", "kubernetes", "helm", "openshift"},
        "nosql": {
            "mongodb", "mongodb (nosql)", "aerospike", "cassandra", "dynamodb",
            "couchbase", "cosmos db", "redis", "neo4j",
        },
        # These named relational databases directly establish the broader SQL
        # capability.  The mapping is intentionally one-way: SQL experience
        # does not claim hands-on use of any particular database product.
        "sql": {
            "mysql", "postgres", "postgresql", "oracle", "oracle db", "mariadb",
            "sql server", "microsoft sql server", "sqlite", "db2",
        },
        # Approved concrete evidence for broad capabilities. These are
        # intentionally one-way: they make the umbrella requirement yellow,
        # but do not assert use of a named vendor/tool in its capability group.
        "ocr": {"document digitization", "text extraction", "image to text", "image-to-text", "document intelligence"},
        "data modeling": {
            "schema design", "dimensional modeling", "dimensional modelling", "er diagrams",
            "entity relationship diagrams", "data architecture", "star schema", "data model",
        },
        "data warehouse": {
            "etl", "elt", "data lake", "data mart", "olap", "snowflake", "bigquery", "redshift",
            "synapse", "databricks sql", "delta lake",
        },
        "data validation": {"validation rules", "integrity checks", "error detection", "reconciliation"},
        "deduplication": {"record matching", "entity resolution", "duplicate detection", "data cleansing"},
        "consistency checks": {"data integrity", "cross field validation", "cross-field validation", "referential integrity"},
        "data quality control": {"data profiling", "completeness", "accuracy", "standardization", "anomaly detection"},
        "data quality checks": {"data profiling", "completeness", "accuracy", "standardization", "anomaly detection"},
        "analytics": {"reporting", "dashboards", "business intelligence", "bi", "insights", "kpi analysis", "data visualization"},
        "backend development": {"fastapi", "flask", "django", "spring boot", "express", "node.js", "rest apis", "api design", "rest services"},
        "data pipelines": {"etl", "elt", "airflow", "dagster", "data ingestion", "data integration", "data transformation"},
        "data quality": {"data validation", "deduplication", "consistency checks", "data quality checks", "great expectations", "dbt tests", "data profiling", "completeness", "accuracy", "standardization", "anomaly detection"},
        "requirements elicitation": {"discovery workshops", "stakeholder interviews", "requirements gathering", "business requirements", "user stories"},
        "agile delivery": {"scrum", "kanban", "sprint planning", "backlog refinement", "backlog grooming", "sprint reviews", "retrospectives"},
        "deployment automation": {"ci/cd", "gitops", "argo cd", "flux cd", "github actions", "gitlab ci", "jenkins", "spinnaker"},
        "infrastructure as code": {"terraform", "ansible", "pulumi", "cloudformation", "arm templates", "bicep"},
        "monitoring": {"prometheus", "grafana", "datadog", "cloudwatch", "cloudwatch metrics", "opentelemetry", "distributed tracing"},
        "logging": {"elk", "elasticsearch", "opensearch", "splunk", "cloudwatch logs", "loguru", "structured logging"},
        "document processing": {"ocr", "pymupdf", "pdfplumber", "apache tika", "unstructured.io", "unstructured"},
    }
    anchors = direct_anchors.get(required, set())
    if anchors & candidates or any(_approved_anchor_in_values(anchor, candidate_skills) for anchor in anchors):
        return "Explicit named technology evidence establishes this broader capability."
    return None
