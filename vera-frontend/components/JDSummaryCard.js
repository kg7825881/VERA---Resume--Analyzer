export default function JDSummaryCard({ jd }) {
    if (!jd) return null;
  
    const groups = [
      { key: "mandatory-technical", label: "Mandatory Technical Skills", items: jd.mandatory_skills || [], kind: "mandatory" },
      { key: "mandatory-domain", label: "Mandatory Domain Requirements", items: jd.mandatory_domain_requirements || [], kind: "mandatory" },
      { key: "mandatory-role", label: "Mandatory Role-Specific Requirements", items: jd.mandatory_role_specific_requirements || [], kind: "mandatory" },
      { key: "preferred-technical", label: "Preferred Technical Skills", items: jd.preferred_technical_skills || [], kind: "preferred" },
      { key: "role-requirements", label: "Role-Specific Requirements", items: jd.soft_preferred_skills || [], kind: "preferred" },
      { key: "domain-experience", label: "Domain Experience", items: jd.industry_keywords || [], kind: "domain" },
    ].filter((group) => group.items.length > 0);
  
    const eduReq = (jd.education_requirements || [])
      .map((e) => [e.degree_level, e.field].filter(Boolean).join(" in "))
      .filter(Boolean)
      .join(" or ");
  
    const summaryLine = [
      jd.min_years_experience > 0 ? `${jd.min_years_experience}+ yrs experience` : null,
      eduReq || null,
    ]
      .filter(Boolean)
      .join(" · ");
  
    return (
      <div className="panel" style={{ marginTop: 14 }}>
        <div className="body">
          <h3 style={{ margin: "0 0 12px" }}>{jd.role_title || "Role"}</h3>
  
          {groups.map((group) => (
            <div key={group.key} style={{ marginBottom: 12 }}>
              <small style={{ display: "block", color: "var(--m)", fontWeight: 800, letterSpacing: ".05em", marginBottom: 6 }}>
                {group.label}
              </small>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {group.items.map((item) => (
                  <span
                    key={`${group.key}-${item}`}
                    className={group.kind === "mandatory" ? "tag" : "tag pref"}
                    style={group.kind === "mandatory" ? { background: "var(--b)", color: "#fff", fontWeight: 600 } : undefined}
                  >
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
  
          {summaryLine && <div style={{ color: "var(--m)", fontSize: 13 }}>{summaryLine}</div>}
        </div>
      </div>
    );
  }
  
