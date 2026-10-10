from source_bundle import build_source_bundle, source_hash


def test_source_bundle_has_stable_content_addressed_id_and_normalized_markdown():
    first = build_source_bundle(
        document_id="document-a", markdown="\ufeff# Ada\r\nPython\r\n", file_name="ada.docx",
        extraction_method="docx_markdown", extraction_warnings=["ocr skipped"],
    )
    second = build_source_bundle(
        document_id="document-b", markdown="# Ada\nPython\n", file_name="copy.docx",
        extraction_method="docx_markdown",
    )

    assert first["source_id"] == second["source_id"]
    assert first["source_hash"] == source_hash("# Ada\nPython\n")
    assert first["markdown"] == "# Ada\nPython\n"
    assert first["document_id"] == "document-a"
    assert first["extraction_warnings"] == ["ocr skipped"]


def test_db_persists_source_bundle_fields(tmp_path, monkeypatch):
    import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "vera.db"))
    db.init_db()
    bundle = build_source_bundle(
        document_id="resume-doc", markdown="# Ada\nPython", file_name="ada.docx",
        extraction_method="docx_markdown",
    )
    db.insert_resume({
        "candidate_id": "candidate-1", "document_id": bundle["document_id"],
        "candidate_name": "Ada", "skills": ["Python"], **{
            key: bundle[key] for key in ("source_id", "source_hash")
        },
        "source_markdown": bundle["markdown"], "source_bundle_version": bundle["bundle_version"],
    })

    stored = db.get_resume_by_candidate_id("candidate-1")
    assert stored["source_id"] == bundle["source_id"]
    assert stored["source_hash"] == bundle["source_hash"]
    assert stored["source_markdown"] == "# Ada\nPython"
