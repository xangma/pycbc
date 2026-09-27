"use strict";

const $ = id => document.getElementById(id);
const state = {report: null, step: null, arm: null, detector: "all", stage: null,
               row: null, verdict: "fail", tab: "differences"};

function el(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = String(value);
  return node;
}
function clear(node) { node.replaceChildren(); }
function nice(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(5);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
function badge(status) {
  return el("span", `badge ${status === "pass" ? "pass" : status === "fail" ? "fail" : "missing"}`, status);
}
function stepBadge(status) {
  return badge(status === "passed" ? "pass" : status === "failed" ? "fail" : status);
}
function step() { return state.report.steps.find(x => x.id === state.step); }
function rows() {
  return state.report.comparisons.filter(x => x.step === state.step && x.arm === state.arm &&
    (state.detector === "all" || x.detector === state.detector || !x.detector));
}
function saveState() {
  const params = new URLSearchParams({step: state.step || "", arm: state.arm || "",
    stage: state.stage || "", tab: state.tab, verdict: state.verdict,
    detector: state.detector});
  history.replaceState(null, "", `#${params}`);
}
function setStep(id) {
  state.step = id;
  const arms = Object.keys(step().science);
  state.arm = arms.includes(state.arm) ? state.arm : (arms[0] || null);
  state.detector = "all";
  state.stage = null;
  state.row = null;
  render();
}
function renderNavigation() {
  const target = $("steps"); clear(target);
  for (const item of state.report.steps) {
    const button = el("button", `step ${item.id === state.step ? "active" : ""}`);
    button.type = "button";
    const title = el("span", "stepname", item.executable);
    title.append(stepBadge(item.status));
    const meta = el("span", "stepmeta", `${item.templates} templates · ${item.mode} · ${item.kind}`);
    button.append(title, meta);
    button.addEventListener("click", () => setStep(item.id));
    target.append(button);
  }
}
function renderHero() {
  const item = step();
  $("title").textContent = `${item.executable} · ${item.templates} templates`;
  $("subtitle").textContent = `${item.kind} / ${item.mode}. ${item.provisional ? "Only a progress checkpoint is available; evidence and timings are provisional." : item.receipt ? "Campaign receipt available." : "Campaign was not run or its receipt is missing."}`;
  const overall = $("overall"); clear(overall);
  overall.append(stepBadge(item.status));
  $("topmeta").textContent = `${state.report.source_artifact} · suite ${state.report.suite_status}`;
}
function renderControls() {
  const arm = $("arm"); clear(arm);
  const names = Object.keys(step().science);
  if (!names.length) names.push("not run");
  for (const name of names) {
    const option = el("option", "", name); option.value = name; arm.append(option);
  }
  arm.value = state.arm || "not run";
  const detector = $("detector"); clear(detector);
  const detectors = [...new Set(rows().map(x => x.detector).filter(Boolean))];
  for (const name of ["all", ...detectors]) {
    const option = el("option", "", name === "all" ? "All detectors" : name);
    option.value = name; detector.append(option);
  }
  detector.value = state.detector;
  $("verdict").value = state.verdict;
  const science = step().science[state.arm];
  const note = science ? `Strict qualification: ${science.passed === true ? "passed" : science.passed === false ? "failed" : "unavailable"}.` : "Science was not run.";
  const inspected = rows().filter(x => x.examples);
  const available = inspected.filter(x => x.examples.status === "available").length;
  const detail = inspected.length
    ? `${available} of ${inspected.length} failing datasets have exact-value coordinates in this report.`
    : "No coordinate-level examples were captured for this selection.";
  $("coverage").textContent = `${note} ${state.report.coverage_note} ${detail} Empty stage cards mean no observation was recorded, not agreement.`;
}
function renderStages() {
  const target = $("stages"); clear(target);
  const all = rows();
  for (const [index, stage] of state.report.stages.entries()) {
    const subset = all.filter(x => x.stage === stage.id);
    const failed = subset.filter(x => x.verdict === "fail" || x.verdict === "missing").length;
    const status = failed ? "fail" : subset.length ? "pass" : "unobserved";
    const button = el("button", `stage ${stage.id} ${status} ${state.stage === stage.id ? "active" : ""}`);
    button.type = "button";
    button.append(el("span", "num", String(index + 1).padStart(2, "0")),
                  el("span", "name", stage.label),
                  el("span", "state", status === "unobserved" ? "No recorded tap" : `${subset.length} fields · ${failed} ${failed === 1 ? "issue" : "issues"}`));
    button.addEventListener("click", () => {state.stage = stage.id; state.row = null; render();});
    target.append(button);
  }
  const first = state.report.stages.find(s => all.some(x => x.stage === s.id && x.verdict === "fail"));
  const missing = all.some(x => x.verdict === "missing");
  $("stagehint").textContent = first ? `Earliest observed failing stage: ${first.label}`
    : missing ? "No observed failing stage; coverage is incomplete" : "No observed failing stage";
}
function visibleRows() {
  return rows().filter(x => (!state.stage || x.stage === state.stage) &&
    (state.verdict === "all" || x.verdict !== "pass"));
}
function renderFields() {
  const stage = state.report.stages.find(x => x.id === state.stage);
  $("fieldtitle").textContent = stage ? stage.label : "Recorded fields";
  const target = $("fields"); clear(target);
  const data = visibleRows();
  $("fieldcount").textContent = `${data.length} shown`;
  if (!data.length) target.append(el("div", "empty", "No recorded fields match these filters."));
  if (!data.includes(state.row)) state.row = data[0] || null;
  data.forEach(row => {
    const button = el("button", `field ${row === state.row ? "active" : ""}`);
    button.type = "button";
    button.append(badge(row.verdict), el("span", "fieldname", row.path));
    const changed = row.metric.failed_elements !== undefined ? `${row.metric.failed_elements} changed` : row.kind;
    const suffix = row.scope.includes("block") ? `${row.scope} · ${changed}` : changed;
    button.append(el("span", "fieldsmall", suffix));
    button.addEventListener("click", () => {state.row = row; renderFields(); renderDetail();});
    target.append(button);
  });
  renderDetail();
}
function kv(target, key, value) {
  const row = el("div", "kv"); row.append(el("span", "", key), el("span", "", nice(value))); target.append(row);
}
function renderDetail() {
  const target = $("detail"); clear(target);
  if (!state.row) {
    $("detailtitle").textContent = "Choose a field";
    target.append(el("div", "empty", "Select a recorded field to inspect its strict comparator result."));
    return;
  }
  const row = state.row, metric = row.metric;
  $("detailtitle").textContent = row.path;
  target.append(badge(row.verdict));
  const grid = el("div", "statgrid");
  for (const [label, value] of [
    ["Failed elements", metric.failed_elements], ["Maximum absolute Δ", metric.max_absolute_difference],
    ["Maximum relative Δ", metric.max_relative_difference], ["Exact values", metric.exact]
  ]) {
    if (value === undefined) continue;
    const stat = el("div", "stat"); stat.append(el("label", "", label), el("strong", "", nice(value))); grid.append(stat);
  }
  if (grid.childNodes.length) target.append(grid);
  kv(target, "Scope", row.scope);
  kv(target, "Detector", row.detector || "all / shared");
  kv(target, "Comparison", row.kind);
  if (row.reference_file) kv(target, "Reference HDF", row.reference_file);
  if (row.candidate_file) kv(target, "Candidate HDF", row.candidate_file);
  const stage = state.report.stages.find(x => x.id === row.stage);
  if (stage) {
    const programPath = step().executable === "pycbc_inspiral" ? stage.inspiral_source : stage.live_source;
    const candidatePath = state.arm?.startsWith("jax") && !["segment", "selection"].includes(stage.id)
      ? stage.jax_source : programPath;
    for (const [role, path] of [["reference", programPath], ["candidate", candidatePath]]) {
      const href = sourceLink(role, path);
      if (!href) continue;
      const link = el("a", "small", `Inspect ${role} stage source · ${path}`);
      link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
      target.append(link, el("br"));
    }
  }
  for (const [key, value] of Object.entries(metric)) {
    if (["failed_elements", "max_absolute_difference", "max_relative_difference", "exact"].includes(key)) continue;
    kv(target, key.replaceAll("_", " "), value);
  }
  if (row.examples) {
    const examples = row.examples;
    const heading = el("h3", "", "Exact value examples"); target.append(heading);
    kv(target, "Detail status", examples.status);
    if (examples.exact_different_elements !== undefined) kv(target, "Exact differences", examples.exact_different_elements);
    if (examples.first) kv(target, "First exact difference", examples.first);
    if (examples.worst_absolute) kv(target, "Largest absolute difference", examples.worst_absolute);
    target.append(el("p", "small", "These coordinates show exact value changes. The acceptance verdict and failed-element count above come from the existing strict comparator."));
  }
  const explanation = row.explanation;
  const cause = el("div", "notice", explanation
    ? `${explanation.status}: ${explanation.claim}`
    : "Cause: unresolved in this receipt. A failing downstream field does not establish its source. Reproduce with stage captures or link a controlled replay before assigning an explanation.");
  target.append(cause);
  for (const item of explanation?.evidence || []) {
    const href = item.href;
    if (!/^https?:\/\//.test(href) && (!/^[\w./-]+$/.test(href) || href.includes(".."))) continue;
    const link = el("a", "", item.label);
    link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
    target.append(link, el("br"));
  }
  if (!row.reference_file || !row.candidate_file) {
    target.append(el("p", "small", "Paired raw HDF paths were not recorded here; exact value coordinates are unavailable in this report."));
  } else if (!row.examples) {
    target.append(el("p", "small", "Raw HDF paths are recorded in the campaign. This receipt view preserves their names; coordinate-level inspection requires the raw files and a targeted replay."));
  }
}
function renderTimings() {
  const target = $("timingcontent"); clear(target);
  const data = state.report.timings.filter(x => x.step === state.step);
  if (!data.length) {
    target.append(el("div", "empty", "No timing repetitions were recorded for this step. The suite stopped at the science gate or did not reach this step."));
    return;
  }
  const max = Math.max(...data.map(x => Number(x.wall_seconds) || 0), 1);
  for (const item of data) {
    const card = el("div", "timingrow");
    const head = el("div", "timinghead");
    head.append(el("strong", "", `${item.arm} · run ${item.replicate}`),
      el("span", `badge ${item.publishable ? "pass" : "missing"}`, item.publishable ? "publishable" : "diagnostic"));
    card.append(head, el("div", "small", `${item.scope} · process wall ${nice(item.wall_seconds)} s`));
    if (item.wall_seconds !== null && item.wall_seconds !== undefined) {
      const bar = el("div", "timingbar"), fill = el("span");
      fill.style.width = `${100 * item.wall_seconds / max}%`; bar.append(fill); card.append(bar);
    }
    if (item.calc_seconds !== null && item.calc_seconds !== undefined) card.append(el("div", "small", `Legacy calculation estimate: ${nice(item.calc_seconds)} s`));
    if (Object.keys(item.stages || {}).length) {
      const breakdown = el("details", "breakdown");
      breakdown.append(el("summary", "", "Recorded stage estimates"));
      for (const [stage, seconds] of Object.entries(item.stages)) kv(breakdown, stage.replaceAll("_", " "), `${nice(seconds)} s`);
      card.append(breakdown);
    }
    target.append(card);
  }
}
function sourceLink(role, path) {
  const snapshot = state.report.source_snapshots?.[role]?.files?.[path];
  if (snapshot) return snapshot.path;
  const source = state.report.sources[role] || {};
  if (!source.revision || !path) return null;
  const base = state.report.source_web_urls?.[role];
  return base ? `${base}/blob/${encodeURIComponent(source.revision)}/${path}` : null;
}
function renderSources() {
  const target = $("sourcecontent"); clear(target);
  for (const role of ["reference", "candidate"]) {
    const source = state.report.sources[role] || {};
    const card = el("div", "sourcecard");
    card.append(el("h3", "", `${role === "reference" ? "Original CPU" : "Candidate"} source`));
    kv(card, "Revision", source.revision || "unavailable");
    kv(card, "Dirty working tree", source.dirty);
    kv(card, "Tracked diff SHA256", source.tracked_diff_sha256);
    kv(card, "Untracked files SHA256", source.untracked_files_sha256);
    kv(card, "Source files SHA256", source.files_sha256);
    const patch = state.report.source_patches?.[role];
    if (patch) {
      const link = el("a", "", `View captured ${role} patch`);
      link.href = patch.path; link.target = "_blank"; link.rel = "noopener";
      card.append(link);
    }
    if (source.dirty) card.append(el("p", "small", "Commit links below show the base revision. The captured patch and untracked-file hashes are required to reconstruct this exact run."));
    kv(card, "Code snapshot", state.report.source_snapshots?.[role]?.status || "unavailable");
    target.append(card);
  }
  const card = el("div", "sourcecard");
  card.append(el("h3", "", "Stage source map"), el("p", "", "These are semantic entry points. A file link alone does not prove that a particular operation caused a difference."));
  for (const stage of state.report.stages) {
    const path = step().executable === "pycbc_inspiral" ? stage.inspiral_source : stage.live_source;
    const candidate = state.arm?.startsWith("jax")
      ? (["segment", "selection"].includes(stage.id) ? path : stage.jax_source)
      : path;
    const row = el("div", "kv"); row.append(el("span", "", stage.label));
    const links = el("span");
    for (const [role, file] of [["reference", path], ["candidate", candidate]]) {
      const href = sourceLink(role, file);
      if (href) {
        const exact = state.report.source_snapshots?.[role]?.files?.[file];
        const link = el("a", "", `${role}: ${file}${exact ? " · captured" : " · base revision"}`);
        link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
        links.append(link, el("br"));
      }
    }
    row.append(links); card.append(row);
  }
  target.append(card);
  const info = el("div", "sourcecard");
  info.append(el("h3", "", "Input and command record"));
  kv(info, "Suite SHA256", state.report.source_sha256);
  for (const [key, value] of Object.entries(state.report.runtime || {})) kv(info, key.replaceAll("_", " "), value);
  for (const [key, value] of Object.entries(state.report.input_hashes || {})) kv(info, key, value);
  kv(info, "Step command", step().command);
  kv(info, "Campaign receipt", step().receipt || "not available");
  kv(info, "Input contract", step().input_contract);
  kv(info, "Workload", step().workload);
  target.append(info);
}
function renderTabs() {
  document.querySelectorAll(".tab").forEach(button => {
    const active = button.dataset.tab === state.tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  for (const name of ["differences", "timings", "sources"]) $(name).classList.toggle("hidden", name !== state.tab);
}
function render() {
  renderNavigation(); renderHero(); renderControls(); renderStages(); renderFields();
  renderTimings(); renderSources(); renderTabs(); saveState();
}
async function main() {
  try {
    if (window.PYCBC_DIVERGENCE_REPORT) {
      state.report = window.PYCBC_DIVERGENCE_REPORT;
    } else {
      const response = await fetch("data.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.report = await response.json();
    }
    const saved = new URLSearchParams(location.hash.slice(1));
    state.step = state.report.steps.some(x => x.id === saved.get("step")) ? saved.get("step") : state.report.steps[0]?.id;
    state.arm = saved.get("arm"); state.stage = saved.get("stage");
    state.verdict = saved.get("verdict") === "all" ? "all" : "fail";
    state.tab = ["differences", "timings", "sources"].includes(saved.get("tab")) ? saved.get("tab") : "differences";
    state.detector = saved.get("detector") || "all";
    if (!state.step) throw new Error("No benchmark steps in receipt");
    const arms = Object.keys(step().science);
    if (!arms.includes(state.arm)) state.arm = arms.find(x => step().science[x].passed === false) || arms[0] || null;
    $("arm").addEventListener("change", event => {state.arm = event.target.value; state.stage = null; state.row = null; render();});
    $("detector").addEventListener("change", event => {state.detector = event.target.value; state.row = null; render();});
    $("verdict").addEventListener("change", event => {state.verdict = event.target.value; state.row = null; render();});
    document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {state.tab = button.dataset.tab; render();}));
    render();
  } catch (error) {
    $("title").textContent = "Unable to load report";
    $("subtitle").textContent = `${error.message}. Rebuild the report or serve this directory with python -m http.server.`;
  }
}
main();
