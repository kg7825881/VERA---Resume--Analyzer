// Client-side exports for the recruiter-ready shortlist. No server-side files
// or external services are needed, which keeps the feature available in both
// local and browser-semantic screening modes.

function escapeXml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;");
}

function safeFilePart(value) {
  return String(value || "shortlist")
    .trim()
    .replace(/[^a-z0-9]+/gi, "-")
    .replace(/(^-|-$)/g, "")
    .toLowerCase() || "shortlist";
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

export function shortlistText({ roleTitle, candidates }) {
  const heading = `VERA qualified candidates — ${roleTitle || "Selected role"}`;
  const criteria = "Included when Job Fit is at least 65% or the candidate passes the mandatory-skill threshold.";
  const rows = (candidates || []).flatMap((candidate) => [
    candidate.candidate_name || "Unnamed candidate",
    ...candidate.summary.map((item) => `• ${item}`),
    "",
  ]);
  return [heading, criteria, "", ...rows].join("\n").trim();
}

// A compact, standards-compliant DOCX writer. The archive is intentionally
// uncompressed so it can be generated without a third-party library.
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function u16(value) { return [value & 255, (value >>> 8) & 255]; }
function u32(value) { return [value & 255, (value >>> 8) & 255, (value >>> 16) & 255, (value >>> 24) & 255]; }

function zipStore(entries) {
  const encoder = new TextEncoder();
  const output = [];
  const directory = [];
  let offset = 0;
  for (const [name, content] of entries) {
    const nameBytes = encoder.encode(name);
    const body = encoder.encode(content);
    const crc = crc32(body);
    const local = [0x50, 0x4b, 0x03, 0x04, ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(crc), ...u32(body.length), ...u32(body.length), ...u16(nameBytes.length), ...u16(0), ...nameBytes, ...body];
    output.push(...local);
    directory.push([0x50, 0x4b, 0x01, 0x02, ...u16(20), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(crc), ...u32(body.length), ...u32(body.length), ...u16(nameBytes.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(0), ...u32(offset), ...nameBytes]);
    offset += local.length;
  }
  const directoryStart = offset;
  for (const item of directory) output.push(...item);
  const directorySize = output.length - directoryStart;
  output.push(0x50, 0x4b, 0x05, 0x06, ...u16(0), ...u16(0), ...u16(entries.length), ...u16(entries.length), ...u32(directorySize), ...u32(directoryStart), ...u16(0));
  return new Uint8Array(output);
}

export function downloadShortlistDocx({ roleTitle, candidates }) {
  const candidateXml = (candidates || []).map((candidate) => {
    const title = `${candidate.candidate_name || "Unnamed candidate"} — ${candidate.final_score}% Job Fit`;
    const bullets = candidate.summary.map((line) => `<w:p><w:pPr><w:pStyle w:val="ListParagraph"/></w:pPr><w:r><w:t xml:space="preserve">${escapeXml(line)}</w:t></w:r></w:p>`).join("");
    return `<w:p><w:pPr><w:spacing w:before="240"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>${escapeXml(title)}</w:t></w:r></w:p>${bullets}`;
  }).join("");
  const documentXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:rPr><w:b/><w:sz w:val="32"/></w:rPr><w:t>VERA qualified candidates</w:t></w:r></w:p><w:p><w:r><w:t>${escapeXml(roleTitle || "Selected role")}</w:t></w:r></w:p><w:p><w:r><w:t>Included when Job Fit is at least 65% or the candidate passes the mandatory-skill threshold.</w:t></w:r></w:p>${candidateXml}<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>`;
  const entries = [
    ["[Content_Types].xml", "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/></Types>"],
    ["_rels/.rels", "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>"],
    ["word/document.xml", documentXml],
  ];
  download(new Blob([zipStore(entries)], { type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" }), `${safeFilePart(roleTitle)}-qualified-candidates.docx`);
}

// Browser print is used for PDF so the operating system can create a real PDF
// with reliable pagination and selectable text, without adding a heavyweight PDF dependency.
export function downloadShortlistPdf({ roleTitle, candidates }) {
  const blocks = (candidates || []).map((candidate) => `<section><h2>${escapeXml(candidate.candidate_name || "Unnamed candidate")} <span>${escapeXml(candidate.final_score)}% Job Fit</span></h2><ul>${candidate.summary.map((line) => `<li>${escapeXml(line)}</li>`).join("")}</ul></section>`).join("");
  const page = window.open("", "_blank");
  if (!page) throw new Error("Allow pop-ups to export the shortlist as a PDF.");
  page.opener = null;
  page.document.write(`<!doctype html><html><head><title>VERA qualified candidates</title><style>body{font-family:Arial,sans-serif;color:#122033;margin:42px;line-height:1.45}h1{margin:0 0 4px}p{color:#4e6478}section{break-inside:avoid;border-top:1px solid #d6dfe7;margin-top:24px;padding-top:14px}h2{font-size:17px;margin:0}h2 span{font-size:13px;color:#16745e;font-weight:normal}li{margin:7px 0}</style></head><body><h1>VERA qualified candidates</h1><p>${escapeXml(roleTitle || "Selected role")} · Included when Job Fit is at least 65% or the candidate passes the mandatory-skill threshold.</p>${blocks}</body></html>`);
  page.document.close();
  page.focus();
  setTimeout(() => page.print(), 250);
}
