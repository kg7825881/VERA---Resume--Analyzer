from extractor import extract_structured_evidence


def test_resume_extraction_is_repeatable_and_records_skill_evidence():
    resume = """
    JANE DOE
    Professional Summary
    Data Engineer with 5+ years of experience.
    Skills and Technologies
    Python, SQL, OCR
    Work Experience
    Acme Corp, Data Engineer
    Jan 2021 - Present
    Built Apache Airflow DAGs and PySpark pipelines.
    Projects
    Claims automation platform
    Used Docker and Kubernetes.
    Education
    B.Tech in Computer Science
    Certifications
    AWS Certified Developer
    """

    first = extract_structured_evidence(resume)
    second = extract_structured_evidence(resume)

    assert first == second
    assert first["candidate_name"] == "JANE DOE"
    assert first["current_role_title_from_summary"] == "Data Engineer"
    assert {"Python", "SQL", "OCR", "Apache Airflow DAGs", "PySpark", "Docker", "Kubernetes"} <= set(first["skills"])
    assert first["experience"][0]["start_date_raw"] == "Jan 2021"
    assert first["experience"][0]["end_date_raw"] == "Present"
    assert any(item["skill"] == "PySpark" and item["source_section"] == "Experience"
               for item in first["skill_evidence"])
    assert any(item["skill"] == "Docker" and item["source_section"] == "Projects"
               for item in first["skill_evidence"])


def test_resume_specific_skills_do_not_need_to_exist_in_a_shared_dictionary():
    resume = """
    ALEX RAY
    Core Competencies
    Rust, Terraform, Playwright, Salesforce, Tableau
    Professional Experience
    Example Labs | Platform Engineer
    2022 - Present
    Tools and Technologies Used - Terraform, Argo Workflows, Grafana
    Projects
    Customer portal
    Technology Stack: TypeScript, React, Supabase
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    # None of these is required to be preconfigured in the JD skill catalogue.
    assert {"Rust", "Terraform", "Playwright", "Salesforce", "Tableau", "Argo Workflows",
            "Grafana", "TypeScript", "React", "Supabase"} <= skills
    assert any(item["skill"] == "Supabase" and item["source_section"] == "Projects"
               for item in extracted["skill_evidence"])


def test_wrapped_skills_are_not_mistaken_for_projects_and_dates_stay_separate():
    resume = """
    KAJAL NIMJE
    Work Experience
    Glint India Technologies Pvt. Ltd. – Product & Business Consultant (Project Based)May 2024 – Present
    Education & Certifications
    PGDM – IT & Marketing
    Skills & Tools
    Product Management: PRD, Roadmaps, Agile/Scrum,
    Automation, LangGraph, Phidata, CrewAI,
    AutoGen, Agent Orchestration
    Generative AI: OpenAI, LangChain, Vector Databases
    PAINT BY NUMBER – CONTOUR ANNOTATION
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert {"PRD", "Roadmaps", "Agile/Scrum", "Automation", "LangGraph", "Phidata",
            "CrewAI", "AutoGen", "Agent Orchestration", "OpenAI", "LangChain",
            "Vector Databases"} <= skills
    assert extracted["experience"][0]["start_date_raw"] == "May 2024"
    assert extracted["experience"][0]["end_date_raw"] == "Present"
    assert extracted["education"] == [{"degree_level": "PGDM", "field": "", "institution": ""}]
    assert extracted["projects"] == [{"title": "PAINT BY NUMBER – CONTOUR ANNOTATION", "technologies_used": []}]


def test_explicit_tools_in_experience_and_projects_are_added_to_global_skills():
    resume = """
    SAM DOE
    Work Experience
    Acme | Data Engineer
    Jan 2022 - Present
    Tools and Techniques Used – Python, Azure Document Intelligence,
    GPT-4o post-processing, Python APIs
    Projects
    Invoice automation platform
    Technology Stack: FastAPI, PostgreSQL, React
    Tech: EasyOCR, Azure Document Intelligence
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert {"Python", "Azure Document Intelligence", "GPT-4o post-processing", "Python APIs",
            "FastAPI", "PostgreSQL", "React", "EasyOCR", "Azure Document Intelligence"} <= skills
    assert any(item["skill"] == "Azure Document Intelligence" and item["source_section"] == "Experience"
               for item in extracted["skill_evidence"])
    assert any(item["skill"] == "PostgreSQL" and item["source_section"] == "Projects"
               for item in extracted["skill_evidence"])


def test_stacked_pdf_list_markers_do_not_hide_experience_tools():
    resume = """
    MORGAN DOE
    Work Experience
    Example Labs | AI Engineer
    Jan 2023 - Present
    - • Tools and Techniques Used – Python, ETL, LangChain, RAG, LLM, Vector Store, ML Ops
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert {"Python", "ETL", "LangChain", "RAG", "LLM", "Vector Store", "ML Ops"} <= skills
    assert {"Python", "ETL", "LangChain", "RAG", "LLM", "Vector Store", "ML Ops"} <= set(
        extracted["experience"][0]["technologies_used"]
    )


def test_wrapped_multiword_skills_and_plus_delimiters_remain_clean():
    resume = """
    TAYLOR DOE
    Technical Skills
    OCR/Docs:
    Azure
    Document
    Intelligence,
    extraction + LLM
    validation, PyMuPDF4LLM
    Misc.:
    Deep
    Learning, Machine
    Learning, Computer Vision, Statistical
    Analysis, Processing Large Datasets
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert {"Azure Document Intelligence", "extraction", "LLM validation", "PyMuPDF4LLM",
            "Deep Learning", "Machine Learning", "Computer Vision", "Statistical Analysis",
            "Processing Large Datasets"} <= skills
    assert "Deep" not in skills
    assert "Learning" not in skills
    assert "+ LLM validation" not in skills


def test_multi_page_style_groups_and_experience_prose_are_kept_as_skills():
    resume = """
    ULKA JAMBHULKAR
    AI Architect
    Work Experience
    Oct 2023 - Present
    Chetu, Inc.
    Built an AI service using Python, OpenCV, scikit-image, PIL, and KMeans clustering.
    Technical Skills
    Agentic AI: LangGraph, CrewAI, AutoGen
    AI Technical Lead with 5+ years of experience delivering AI solutions.
    Certifications
    Data Engineering & Pipelines: Pandas, NumPy, SQL, RAG pipeline design,
    vector indexing using FAISS, Pinecone, and ChromaDB.
    Model Deployment: FastAPI, Flask, CI/CD pipelines, AWS
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert {"Agentic AI", "LangGraph", "CrewAI", "AutoGen", "Python", "OpenCV",
            "scikit-image", "PIL", "KMeans clustering", "Data Engineering & Pipelines",
            "Pandas", "NumPy", "SQL", "RAG pipeline design", "FAISS", "Pinecone",
            "ChromaDB", "FastAPI", "Flask", "CI/CD pipelines", "AWS"} <= skills
    assert "AI Technical Lead with 5" not in skills
    assert extracted["experience"][0]["company"] == "Chetu, Inc."


def test_column_bullet_lists_stay_separate_and_labelled_groups_keep_following_bullets():
    resume = """
    PREETI DOE
    Technical Skills
    Python, SQL
    Exploratory Data Analysis
    Orange Data Mining
    Pandas, NumPy
    Scikit-learn, TensorFlow, Keras
    Matplotlib, Seaborn
    Certifications
    Data Visualization:
    Tableau
    Power BI
    2025-2026
    candidate@example.com
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert {"Python", "SQL", "Exploratory Data Analysis", "Orange Data Mining", "Pandas",
            "NumPy", "Scikit-learn", "TensorFlow", "Keras", "Matplotlib", "Seaborn",
            "Data Visualization", "Tableau", "Power BI"} <= skills
    assert "SQL Exploratory Data Analysis" not in skills
    assert "Tableau Power BI" not in skills


def test_summary_skills_and_decimal_experience_preserve_resume_wording():
    resume = """
    GIRIJA PRASAD YADAV
    Summary
    Data Engineer with 2.5+ years of experience operating ETL/ELT pipelines
    (Azure Data Factory, Databricks, ADLS Gen2). Skilled in Medallion Architecture,
    Delta Lake, dimensional data modeling, advanced SQL, and data governance.
    Technical Skills
    Big Data: Apache Spark, PySpark, Parquet
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert extracted["stated_years_experience_from_summary"] == 2.5
    assert {"ETL/ELT pipelines", "Azure Data Factory", "Databricks", "ADLS Gen2",
            "Medallion Architecture", "Delta Lake", "dimensional data modeling",
            "advanced SQL", "data governance", "Apache Spark", "PySpark", "Parquet"} <= skills
    assert "Spark" not in skills


def test_current_role_uses_latest_employment_title_not_summary_title():
    resume = """
    ALEX RAY
    Summary
    Product Manager with 8+ years of experience in financial products.
    Work Experience
    BoBCARD Ltd. | Product & Portfolio Manager – Credit Cards
    Sep 2022 - Feb 2024
    Glint India Technologies Pvt. Ltd. | Product & Business Consultant
    May 2024 - Present
    """

    extracted = extract_structured_evidence(resume)

    assert extracted["latest_experience_role_title"] == "Product & Business Consultant"
    assert extracted["current_role_title_from_summary"] == "Product & Business Consultant"
    assert extracted["current_role_title_resolution"] == "latest_experience"


def test_latest_employment_title_wins_when_summary_conflicts():
    resume = """
    TAYLOR DOE
    Summary
    Data Engineer with 5+ years of experience.
    Work Experience
    Example Studio | Product Designer
    Jan 2020 - Dec 2021
    """

    extracted = extract_structured_evidence(resume)

    assert extracted["current_role_title_from_summary"] == "Product Designer"
    assert extracted["current_role_title_resolution"] == "latest_experience"


def test_wrapped_cloud_platform_category_keeps_each_exact_platform_and_service():
    resume = """
    RUPENDRA SHEKHAWAT
    Technical Expertise
    Cloud Platforms: Microsoft Azure (Data Factory, Databricks, Delta Lake, Data Lake
    Storage), AWS, Google Cloud Platform
    Data Technologies: Azure Data Factory, Databricks, Apache Spark, PySpark, Apache
    Airflow, Apache NiFi
    """

    extracted = extract_structured_evidence(resume)
    skills = set(extracted["skills"])

    assert {"Microsoft Azure", "Data Factory", "Databricks", "Delta Lake", "Data Lake Storage",
            "AWS", "Google Cloud Platform", "Azure Data Factory", "Apache Spark", "PySpark",
            "Apache Airflow", "Apache NiFi"} <= skills
    assert "Cloud Platforms" not in skills
    assert "Data Technologies" not in skills
