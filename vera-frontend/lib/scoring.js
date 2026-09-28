// lib/scoring.js
// Presentation helpers that derive everything from the real score records
// returned by GET /results/{role_id} — mirrors backend weights and categories.

// Max points per category mirroring scorer.py's WEIGHTS * 100
export const CATEGORY_MAX = {
  mandatory_skills: 30,
  relevant_experience: 20,
  education: 25,
  soft_skills: 10,
  job_title_match: 10,
  preferred_skills: 5,
};

export const CATEGORY_LABELS = {
  mandatory_skills: "Mandatory Skills",
  relevant_experience: "Experience",
  education: "Education",
  soft_skills: "Role-Specific Requirements",
  job_title_match: "Job Title Match",
  preferred_skills: "Preferred Skills",
};

// Shared 3-state status → accent color, used by the Evidence panel's sub-sections.
export const STATUS_COLOR = {
  matched: "var(--g)",
  weak_match: "#f5bd62",
  missing: "var(--r)",
};

export const STATUS_ICON = {
  matched: "✓",
  weak_match: "~",
  missing: "×",
};

export function statusFor(finalScore) {
  // Candidate status is deliberately based only on the final Job Fit score.
  // The mandatory-skills gate remains visible as evidence, but does not change
  // this recruiter-facing label.
  if (finalScore >= 85) return { key: "excellent", label: "Excellent" };
  if (finalScore >= 80) return { key: "strong", label: "Strong" };
  if (finalScore >= 75) return { key: "review", label: "Review" };
  return { key: "low", label: "Reject" };
}

export function initials(name) {
  if (!name) return "?";
  const parts = name.trim().split(/\s+/);
  return parts
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() || "")
    .join("");
}

/** The backend is the source of truth for the configurable mandatory-skills gate. */
export function passesMandatory(record) {
  return !record.hard_gate_failed;
}

/**
 * Rank candidates by their final Job Fit score.  Mandatory-gate status is
 * retained on each record for filters and labels, but must not silently move a
 * higher-scoring candidate below a lower-scoring candidate in the main list.
 */
export function rankAll(ranked, excluded) {
  const byScore = (a, b) => b.final_score - a.final_score;
  return [...(ranked || []), ...(excluded || [])]
    .sort(byScore)
    .map((r, i) => ({ ...r, rank: i + 1 }));
}

/** Top few matched skills across categories, for a compact "top evidence" table cell. */
export function topEvidence(record, limit = 3) {
  const cs = record.category_scores || {};
  const pool = [
    ...(cs.mandatory_skills?.matched || []),
    ...(cs.preferred_skills?.matched || []),
  ];
  return pool.slice(0, limit).join(" · ");
}

/** Every matched skill across skill categories, deduplicated. */
export function allMatchedSkills(record) {
  const cs = record.category_scores || {};
  const pool = [
    ...(cs.mandatory_skills?.matched || []),
    ...(cs.preferred_skills?.matched || []),
  ];
  return [...new Set(pool)];
}

/** The candidate's most recent role (first entry in their experience list), or null. */
export function mostRecentRole(record) {
  const experience = record.experience || [];
  return experience.length > 0 ? experience[0] : null;
}

/** Flat list across all skill-based categories for backward compatibility. */
export function evidenceList(record) {
  const cs = record.category_scores || {};
  const cats = [
    ["mandatory_skills", "Mandatory"],
    ["preferred_skills", "Preferred"],
    ["soft_skills", "Role-Specific Requirements"],
  ];
  const items = [];
  for (const [key, label] of cats) {
    for (const skill of cs[key]?.matched || []) {
      items.push({ skill, matched: true, category: label });
    }
    for (const skill of cs[key]?.missing || []) {
      items.push({ skill, matched: false, category: label });
    }
  }
  return items;
}

function mapSkillRow(r) {
  return {
    label: r.skill,
    status: r.status, // "matched" | "weak_match" | "missing"
    detail: r.detail || (r.matched_against ? `matched against: ${r.matched_against}` : null),
    matchType: r.match_type, // "exact" | "evidence" | "none"
    citations: r.citations || [],
  };
}

/**
 * Structured evidence for the Candidate Detail page's Evidence panel
 */
export function evidenceSections(record) {
  const ev = record.evidence || {};

  const skillSections = [
    { key: "mandatory_technical_skills", label: "Mandatory Technical Skills", items: (ev.mandatory_technical_skills || ev.mandatory_skills || []).map(mapSkillRow) },
    { key: "mandatory_domain_requirements", label: "Mandatory Domain Requirements", items: (ev.mandatory_domain_requirements || []).map(mapSkillRow) },
    { key: "mandatory_role_specific_requirements", label: "Mandatory Role-Specific Requirements", items: (ev.mandatory_role_specific_requirements || []).map(mapSkillRow) },
    { key: "soft_skills", label: "Role-Specific Requirements", items: (ev.soft_skills || []).map(mapSkillRow) },
    { key: "preferred_skills", label: "Preferred Skills", items: (ev.preferred_skills || []).map(mapSkillRow) },
  ].filter((section) => section.items.length > 0);

  const experienceEv = ev.experience || {};
  const experienceRoles = (experienceEv.roles || []).map((r) => ({
    label: [r.title, r.company].filter(Boolean).join(" @ ") || "Role",
    detail: r.duration,
    status: r.status,
    similarity: r.relevance_similarity,
  }));

  let educationItems = [];
  if (ev.education?.[0]?.status === "not_required") {
    educationItems = [{ label: "No education requirement in this job description", status: "matched", detail: "Not scored as a candidate qualification" }];
  } else if (ev.education && ev.education.length > 0) {
    educationItems = ev.education.map((e) => ({
      label: [e.required_degree_level, e.required_field].filter(Boolean).join(" in ") || e.skill || e.requirement || "Education requirement",
      status: e.status, // "matched" or "missing"
      detail: e.status === "matched" ? "Satisfied by candidate degree" : "Not found in resume",
    }));
  }

  return {
    skillSections,
    experience: { years: experienceEv.years || null, roles: experienceRoles },
    jobTitle: ev.job_title || null,
    education: educationItems,
    additionalSkills: ev.additional_candidate_skills || [],
    additionalSkillsTotal: ev.additional_candidate_skills_total || 0,
  };
}

export function formatEducation(education) {
  if (!education || education.length === 0) return "—";
  return education.map((e) => [e.degree_level, e.field].filter(Boolean).join(" — ")).join(", ");
}

/** Grounded, template-built explanation of a candidate's rank. */
export function explainRank(record, roleTitle) {
  const cs = record.category_scores || {};
  const missingMandatory = cs.mandatory_skills?.gate_missing || cs.mandatory_skills?.missing || [];
  const core = cs.mandatory_skills || {};
  const evidence = record.evidence || {};
  const title = evidence.job_title;
  const lines = [
    `#${record.rank} of this batch · ${record.final_score}% fit for ${roleTitle || "this role"}.`,
  ];

  if (record.hard_gate_failed) {
    lines.push(record.hard_gate_reason || "A mandatory requirement was not met.");
    return lines;
  }

  const experience = evidence.experience?.years;
  const strengths = [];
  const matchedCore = (core.matched || []).slice(0, 4);
  const preferred = (cs.preferred_skills?.matched || []).slice(0, 2);
  if (matchedCore.length) strengths.push(matchedCore.join(", "));
  if (preferred.length) strengths.push(preferred.join(", "));
  if (experience?.total_years_experience != null) {
    strengths.push(`${experience.total_years_experience} years' experience`);
  }
  if (title?.best_match?.title) {
    strengths.push(title.best_match.title);
  }
  if (strengths.length) lines.push(`Strengths: ${strengths.join(" · ")}.`);
  const gaps = [
    ...missingMandatory.slice(0, 2),
    ...(cs.preferred_skills?.missing || []).slice(0, 2),
  ];
  lines.push(gaps.length ? `Gaps to verify: ${[...new Set(gaps)].join(", ")}.` : "No material gaps found in the scored requirements.");
  return lines.slice(0, 3);
}

function uniqueLabels(items) {
  const seen = new Set();
  return items.filter((item) => {
    const value = typeof item === "string" ? item.trim() : "";
    const key = value.toLowerCase();
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/**
 * Build the full recruiter-facing candidate narrative from the persisted
 * assessment. This is deliberately deterministic: every statement maps to a
 * score, evidence status, or explicitly extracted resume fact.
 */
export function detailedCandidateSummary(record, roleTitle) {
  const evidence = record.evidence || {};
  const scores = record.category_scores || {};
  const sections = evidenceSections(record);
  const requirementRows = [
    ...(evidence.mandatory_technical_skills || evidence.mandatory_skills || []),
    ...(evidence.mandatory_domain_requirements || []),
    ...(evidence.mandatory_role_specific_requirements || []),
    ...(evidence.soft_skills || []),
    ...(evidence.preferred_skills || []),
  ];
  const labelsFor = (status) => uniqueLabels(
    requirementRows.filter((row) => row?.status === status).map((row) => row.skill || row.requirement)
  );
  const directStrengths = labelsFor("matched");
  const relatedStrengths = labelsFor("weak_match");
  const missingRequirements = labelsFor("missing");
  const titleEvidence = sections.jobTitle;
  const currentRole = titleEvidence?.best_match?.title
    || record.current_role_title_from_summary
    || record.experience?.find((entry) => entry?.title)?.title;
  const experience = sections.experience?.years;
  const titleScore = Number(scores.job_title_match?.score || 0);
  const lines = [
    `Overall fit: ranked #${record.rank} in this batch with a ${record.final_score}% fit for ${roleTitle || "the selected role"}.`,
  ];

  if (directStrengths.length) {
    lines.push(`Directly evidenced strengths: ${directStrengths.join(", ")}.`);
  }
  if (relatedStrengths.length) {
    lines.push(`Related or inferred evidence (partial credit; verify depth in interview): ${relatedStrengths.join(", ")}.`);
  }

  const gaps = [...missingRequirements];
  if (titleScore === 0 && currentRole) gaps.push("job-title alignment");
  if (experience?.status === "missing") gaps.push("minimum experience");
  if (record.hard_gate_failed) {
    lines.push(`Eligibility concern: ${record.hard_gate_reason || "mandatory-skill coverage did not meet the required threshold"}.`);
  }
  if (gaps.length) lines.push(`Gaps or interview checks: ${uniqueLabels(gaps).join(", ")}.`);
  else lines.push("Gaps or interview checks: no missing scored requirement was recorded.");

  const additional = uniqueLabels(sections.additionalSkills || []);
  if (additional.length) {
    const remainder = Math.max(0, (sections.additionalSkillsTotal || additional.length) - additional.length);
    lines.push(
      `Additional candidate skills: ${additional.join(", ")}.`
      + (remainder ? ` ${remainder} more additional extracted skill${remainder === 1 ? "" : "s"} are available in the evidence record.` : "")
    );
  }
  return lines;
}
