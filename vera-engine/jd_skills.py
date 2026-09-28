"""Deterministic JD skill normalisation and extraction helpers.

This module deliberately contains no model calls.  The extractor supplies text from a
known JD section and these functions return stable, first-seen ordered labels.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


_FILLER = re.compile(
    r"^(?:strong|good|solid|hands[- ]on|proven|demonstrated|working)\s+"
    r"(?:experience|knowledge|understanding|skills?|proficiency|expertise)?\s*"
    r"(?:with|in|of|using|building)?\s*",
    re.I,
)

# Canonical labels are intentionally explicit: adding a term is a reviewable code
# change instead of an LLM silently expanding a hard requirement.
SKILL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("SQL", r"\bsql\b"), ("Python", r"\bpython\b"),
    ("ETL", r"\betl\b"), ("ELT", r"\belt\b"),
    ("Data modeling", r"\bdata model(?:ing)?\b"),
    ("Schema design", r"\bschema design\b"),
    ("Data transformation logic", r"\b(?:data )?transformation logic\b"),
    ("Structured and semi-structured data", r"\bstructured(?:\s+and\s+|/)semi[- ]structured data\b"),
    ("Data warehouse", r"\bdata warehouse\b"), ("Lakehouse", r"\blakehouse\b"),
    ("Data validation", r"\bdata validation\b"), ("Deduplication", r"\bdeduplicat\w*\b"),
    ("Consistency checks", r"\bconsistency checks?\b"),
    ("Data quality control", r"\bdata quality control\b"),
    ("Data quality checks", r"\bdata quality checks?\b"),
    ("Monitoring", r"\bmonitoring\b"), ("Alerting", r"\balert(?:ing|s)?\b"),
    ("Spark", r"\b(?:apache )?spark\b"), ("dbt", r"\bdbt\b"),
    ("Airflow", r"\bairflow\b"), ("Dagster", r"\bdagster\b"),
    ("Kafka", r"\bkafka\b"), ("OCR", r"\bocr\b|optical character recognition"),
    ("Vector databases", r"\bvector databases?\b"),
    ("Embedding pipelines", r"\bembedding pipelines?\b"),
    ("Retrieval datasets", r"\bretrieval datasets?\b"),
    ("Data lineage", r"\bdata lineage\b"), ("Data governance", r"\bdata governance\b"),
    ("Access control", r"\baccess control\b"),
    ("Java", r"\bjava\b"), ("Spring Boot", r"\bspring boot\b"),
    ("Core Java", r"\bcore java\b"), ("Java EE", r"\b(?:java ee|j2ee)\b"),
    ("Spring Framework", r"\bspring framework\b|\bspring mvc\b"),
    ("Spring Data JPA", r"\bspring data jpa\b"), ("Hibernate", r"\bhibernate\b"),
    ("Spring Discovery", r"\bspring discovery\b"),
    ("Spring Cloud Gateway", r"\b(?:spring )?cloud gateway\b"),
    # This JD abbreviates the Spring ecosystem list as "Spring Discovery, config,
    # cloud gateway, batch"; retain its intended Spring Batch requirement.
    ("Spring Batch", r"\b(?:spring )?batch\b"),
    ("API design", r"\bapi design\b"),
    ("Circuit breakers", r"\bcircuit breakers?\b"),
    ("Resiliency patterns", r"\bresilienc(?:y|e) patterns?\b"),
    ("Fault tolerance", r"\bfault\s+tolerance\b"),
    ("Configuration management", r"\bconfiguration management\b"),
    ("JUnit", r"\bjunit\b"), ("Mockito", r"\bmockito\b"),
    ("Maven", r"\bmaven\b"), ("Gradle", r"\bgradle\b"),
    ("Jenkins", r"\bjenkins\b"), ("CI/CD", r"\bci\s*/\s*cd\b|\bcontinuous (?:integration|delivery|deployment)\b"),
    ("AWS", r"\baws\b|\bamazon web services\b"), ("Azure", r"\bazure\b"),
    ("Google Cloud", r"\b(?:google cloud|gcp)\b"),
    ("AWS SQS", r"\baws sqs\b|\bsimple queue service\b"),
    ("Retry strategies", r"\bretry strategies?\b"),
    ("Oracle", r"\boracle\b"), ("MySQL", r"\bmysql\b"),
    ("NoSQL", r"\bno\s*sql\b"),
    ("Redis", r"\bredis\b"), ("ActiveMQ", r"\bactivemq\b"),
    ("RabbitMQ", r"\brabbitmq\b"), ("SOAP", r"\bsoap\b"),
    ("JSON", r"\bjson\b"), ("XML", r"\bxml\b"),
    ("Design patterns", r"\bdesign patterns?\b"),
    ("SOLID principles", r"\bsolid principles?\b"),
    ("Scalability", r"\bscalability\b"),
    ("Performance engineering", r"\bperformance engineering\b"),
    ("Code review", r"\bcode reviews?\b"),
    ("Containerization", r"\bcontaineri[sz]ation\b"),
    ("Microservices", r"\bmicroservices?\b"), ("REST APIs", r"\brest(?:ful)? APIs?\b"),
    ("Distributed systems", r"\bdistributed systems?\b"),
    ("Event-driven architecture", r"\bevent[- ]driven architecture\b"),
    ("Cloud-native architecture", r"\bcloud[- ]native architecture\b"),
    ("DevOps", r"\bdevops\b"), ("Docker", r"\bdocker\b"),
    ("Kubernetes", r"\bkubernetes\b"), ("Helm", r"\bhelm charts?\b"),
    ("Istio", r"\bistio\b"), ("Linkerd", r"\blinkerd\b"),
    ("Infrastructure as Code", r"\b(?:infrastructure as code|iac)\b"),
    ("Argo CD", r"\bargo cd\b"), ("GitOps", r"\bgitops\b"),
    ("Blue-Green deployments", r"\bblue[- ]green\b"), ("Canary deployments", r"\bcanary\b"),
    ("Prometheus", r"\bprometheus\b"), ("Grafana", r"\bgrafana\b"),
    ("Loki", r"\bloki\b"), ("ELK/OpenSearch", r"\b(?:elk|opensearch)\b"),
    ("OpenTelemetry", r"\bopen telemetry\b"), ("GitHub Copilot", r"\bgithub copilot\b"),
    ("ChatGPT", r"\bchatgpt\b"),
    ("Claude", r"\bclaude\b"), ("Cursor", r"\bcursor\b"),
    # Shared baseline dictionary: these labels can appear in either a JD or a
    # candidate's Skills, Experience, or Projects section.
    ("FastAPI", r"\bfastapi\b"), ("Django", r"\bdjango\b"),
    ("APIs", r"\b(?:api|apis|application programming interfaces)\b"),
    ("Backend development", r"\b(?:backend|back-end)(?:/application)?(?:\s+(?:engineering|development|workflows?))?\b"),
    ("Git", r"\bgit\b"), ("GitLab", r"\bgitlab\b"),
    # Do not infer the hosting platform solely from the Copilot product name.
    ("GitHub", r"\bgithub\b(?!\s+copilot\b)"),
    ("Databricks", r"\b(?:azure )?databricks\b"),
    ("Azure OpenAI", r"\bazure openai\b"), ("Azure AI Search", r"\bazure ai search\b"),
    ("PostgreSQL", r"\b(?:postgres|postgresql)\b"),
    ("MongoDB", r"\bmongodb\b"), ("Snowflake", r"\bsnowflake\b"),
    ("TensorFlow", r"\btensorflow\b"), ("PyTorch", r"\bpytorch\b"),
    ("Pandas", r"\bpandas\b"), ("NumPy", r"\bnumpy\b"),
    ("Agile", r"\bagile\b"), ("SaaS", r"\bsaas\b"),
    ("Guidewire ClaimCenter", r"\bguidewire claimcenter\b"),
    ("Duck Creek", r"\bduck creek\b"), ("Origami Risk", r"\borigami risk\b"),
    ("Machine learning", r"\bmachine learning\b"), ("Analytics", r"\banalytics\b"),
    ("Intelligent automation", r"\bintelligent automation\b"),
    # AI/data concepts that are explicitly named requirements, not free-text duties.
    ("Machine learning", r"\b(?:machine learning|\bml\b)"),
    ("LLMs", r"\b(?:llms?|large language models?)\b"),
    ("RAG", r"\brag\b|retrieval[- ]augmented generation"),
    ("GenAI", r"\bgenai\b|generative ai"), ("Model evaluation", r"\b(?:model )?evaluat(?:e|ion)\b"),
    ("Prompt engineering", r"\bprompt (?:engineering|evaluation|quality)\b"),
    ("LLM evaluation", r"\b(?:llm|ai) (?:evaluation|testing|output quality)\b"),
    ("LLM applications", r"\bllm[- ]based applications?\b|\bllm applications?\b"),
    ("LangChain", r"\blangchain\b"), ("LlamaIndex", r"\bllamaindex\b"),
    ("Chunking", r"\b(?:document |text )?chunking\b"),
    ("Embeddings", r"\b(?:vector )?embeddings?\b"),
    ("Vector search", r"\bvector search\b"),
    ("Retrieval quality", r"\bretrieval quality\b"),
    ("Guardrails", r"\bguardrails?\b"),
    ("Output validation", r"\b(?:output|response) validation\b"),
    ("LLM testing", r"\bllm testing\b"),
    ("Debugging", r"\bdebugging\b"),
    ("Logging", r"\blogging\b"),
    ("Cloud deployment", r"\bcloud deployment\b"),
    ("Workflow automation", r"\bworkflow automation\b"),
    ("Document AI", r"\bdocument ai\b"),
    ("Document parsing", r"\bdocument parsing\b"),
    ("Frontend development", r"\bfrontend\b"),
    ("Full-stack development", r"\bfull[- ]stack(?: product delivery| development)?\b"),
    ("FAISS", r"\bfaiss\b"), ("Pinecone", r"\bpinecone\b"),
    ("Weaviate", r"\bweaviate\b"), ("Chroma", r"\bchroma\b"),
    ("Unstructured data", r"\bunstructured data\b"),
    # Kubernetes and delivery vocabulary
    ("Networking", r"\bnetworking\b"), ("Ingress controllers", r"\bingress controllers?\b"),
    ("Storage classes", r"\bstorage classes?\b"), ("Autoscaling", r"\bautoscaling\b"),
    ("RBAC", r"\brbac\b"), ("Namespaces", r"\bnamespaces\b"),
    ("Kubernetes Operators", r"\bkubernetes operators?\b"), ("Service mesh", r"\bservice mesh\b"),
    ("Infrastructure provisioning", r"\binfrastructure provisioning\b"),
    ("Lifecycle management", r"\blifecycle management\b"), ("Rollbacks", r"\brollbacks?\b"),
    ("Release orchestration", r"\brelease orchestration\b"),
    ("Kubernetes deployment automation", r"\bkubernetes deployment automation\b"),
    ("Build automation", r"\bbuild automation\b"), ("Release automation", r"\brelease automation\b"),
    ("Artifact management", r"\bartifact management\b"),
    ("Secure software supply chain", r"\bsecure software supply chain\b"),
    ("Pipeline optimization", r"\bpipeline optimization\b"),
    ("Cloud-agnostic architecture", r"\bcloud[- ]agnostic solutions?\b"),
    ("Observability", r"\bobservability\b"), ("Security", r"\bsecurity\b"),
)
DOMAIN_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Insurance", r"\binsurance\b"), ("Property & Casualty", r"\bproperty (?:&|and) casualty\b"),
    ("Claims", r"\bclaims?\b"), ("Auto claims", r"\bauto claims?\b"),
    ("Property claims", r"\bproperty claims?\b"), ("Recovery operations", r"\brecovery operations\b"),
    ("Subrogation", r"\bsubrogation\b"), ("Fintech", r"\bfintech\b"),
    ("Banking", r"\bbanking\b"), ("Healthcare", r"\bhealthcare\b"),
    ("Retail", r"\bretail\b"), ("Telecom", r"\btelecom\b"),
    ("Payments", r"\bpayments?\b"),
)
ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Document processing", r"\bdocument (?:processing|ingestion|parsing|normalization)\b"),
    ("Metadata extraction", r"\bmetadata extraction\b"),
    ("Data traceability", r"\b(?:data )?traceability\b"),
    ("Cross-functional collaboration", r"\bcross[- ]functional collaboration\b"),
    ("Document-heavy data", r"\bdocument[- ]heavy\b"),
    ("Executive workshops", r"\bexecutive.*workshops?\b"),
    ("Client-facing workshops", r"\bclient[- ]facing workshops?\b"),
    ("Communication", r"\bstrong communication\b"),
)

# Explicit titles mentioned in a Required section are stored separately from skills.
# They define acceptable candidate-title targets for downstream title matching.
TARGET_TITLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Business Analyst", r"\bbusiness analyst\b"), ("Product Analyst", r"\bproduct analyst\b"),
    ("Systems Analyst", r"\bsystems analyst\b"),
    ("Insurance Technology Consultant", r"\binsurance technology consultant\b"),
    ("Data Engineer", r"\bdata engineer\b"), ("Analytics Engineer", r"\banalytics engineer\b"),
    ("Data Platform Engineer", r"\bdata platform (?:engineer|roles?)\b"),
    ("DevOps Engineer", r"\bdevops engineer\b"),
    ("Java Lead Engineer", r"\b(?:sr\.?\s*)?java lead engineer\b"),
)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen, output = set(), []
    for item in items:
        value = (item or "").strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def find_catalog_items(text: str, catalog: tuple[tuple[str, str], ...]) -> list[str]:
    """Return catalog labels in catalog order, independent of formatting noise."""
    return _dedupe(label for label, pattern in catalog if re.search(pattern, text or "", re.I))


def atomize_skill_line(raw_line: str) -> list[str]:
    """Safely split a simple comma/and/or skills list without inventing fragments."""
    line = (raw_line or "").strip().strip("•*- \t")
    if not line or re.search(r"\b(?:years? of experience|bachelor|master|degree|certification)\b", line, re.I):
        return []
    line = _FILLER.sub("", line).rstrip(".")
    if not re.search(r"[,;]|\b(?:and|or)\b", line, re.I):
        return [line] if len(line.split()) <= 6 else []
    parts = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", line, flags=re.I)
    return _dedupe(_FILLER.sub("", part).strip(" .") for part in parts if part.strip())


def atomize_skill_list(raw_items: Iterable[str]) -> list[str]:
    return _dedupe(atom for item in raw_items or [] for atom in atomize_skill_line(item))


_SOFT_TERMS = re.compile(r"\b(?:communication|collaborat|leadership|stakeholder|teamwork|mentoring|presentation)\b", re.I)
_TRAILING_CONTEXT = re.compile(r"\s+(?:skills?|experience|knowledge|expertise|proficiency|pipelines?|workflows?)$", re.I)


def extract_requirement_items(lines: Iterable[str]) -> tuple[list[str], list[str]]:
    """Generic deterministic extraction from explicit Required/Preferred bullets."""
    skills, roles = [], []
    for raw in lines:
        line = (raw or "").strip("•*- \t").rstrip(".")
        if not line or re.search(r"\b(?:years? of experience|bachelor|master|degree|certification)\b", line, re.I):
            continue
        for part in re.split(r"\s*(?:;|,|\band\b|\bor\b)\s*", line, flags=re.I):
            item = _TRAILING_CONTEXT.sub("", _FILLER.sub("", part).strip()).strip(" .")
            if item and len(item.split()) <= 8:
                (roles if _SOFT_TERMS.search(item) else skills).append(item)
    return _dedupe(skills), _dedupe(roles)
