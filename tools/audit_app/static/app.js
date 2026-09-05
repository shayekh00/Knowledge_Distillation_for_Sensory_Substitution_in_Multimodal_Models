// VQA-SUNRGBD-v2 audit tool — frontend.
// No build step, no framework: one file, plain DOM + fetch.

const state = {
  annotatorId: "",
  items: [],          // full audit set, from /api/items (gold answer included)
  filtered: [],        // items after type/unanswered filters
  responses: {},        // question_id -> saved response, for the current annotator
  currentIndex: 0,
};

const el = (id) => document.getElementById(id);

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${url}: ${body}`);
  }
  return response.json();
}

// ── Bootstrap ────────────────────────────────────────────────────────────

const DEFAULT_SOLO_ANNOTATOR_ID = "solo";

async function init() {
  // Defaults to the declared single reviewer id and is remembered per browser.
  state.annotatorId = localStorage.getItem("vqa_audit_annotator_id") || DEFAULT_SOLO_ANNOTATOR_ID;
  localStorage.setItem("vqa_audit_annotator_id", state.annotatorId);
  el("annotator-input").value = state.annotatorId;

  const status = await fetchJSON("/api/status");
  if (!status.items_loaded) {
    el("status-banner").textContent = status.load_error;
    el("status-banner").classList.remove("hidden");
    return;
  }

  const { items, model_hints_loaded } = await fetchJSON("/api/items");
  state.items = items;
  state.modelHintsLoaded = Boolean(model_hints_loaded);
  populateTypeFilter(items);
  el("app").classList.remove("hidden");
  if (!state.modelHintsLoaded) el("sort-order").value = "sample";
  renderModelSummary();

  el("annotator-input").addEventListener("input", onAnnotatorChanged);
  el("type-filter").addEventListener("change", applyFilters);
  el("only-unanswered").addEventListener("change", applyFilters);
  el("sort-order").addEventListener("change", applyFilters);
  el("model-reasoning-toggle").addEventListener("click", (event) => {
    event.preventDefault();
    el("model-hint-reasoning").classList.toggle("hidden");
  });
  el("prev-btn").addEventListener("click", () => moveTo(state.currentIndex - 1));
  el("next-btn").addEventListener("click", () => moveTo(state.currentIndex + 1));
  el("save-correction-btn").addEventListener("click", saveCorrection);
  el("own-answer-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") saveCorrection();
  });
  document.querySelectorAll("button.verdict[data-verdict]").forEach((button) => {
    button.addEventListener("click", () => onVerdictClicked(button.dataset.verdict));
  });
  el("use-model-btn").addEventListener("click", saveModelAnswerAsCorrection);
  document.addEventListener("keydown", onGlobalKeydown);
  el("stats-toggle").addEventListener("click", openStats);
  el("stats-close").addEventListener("click", () => el("stats-panel").classList.add("hidden"));

  if (state.annotatorId) {
    await loadAnnotatorState();
  } else {
    applyFilters();
  }
}

function populateTypeFilter(items) {
  const types = [...new Set(items.map((item) => item.question_type))].sort();
  const select = el("type-filter");
  for (const type of types) {
    const option = document.createElement("option");
    option.value = type;
    option.textContent = type;
    select.appendChild(option);
  }
}

async function onAnnotatorChanged() {
  state.annotatorId = el("annotator-input").value.trim();
  localStorage.setItem("vqa_audit_annotator_id", state.annotatorId);
  await loadAnnotatorState();
}

async function loadAnnotatorState() {
  if (!state.annotatorId) {
    state.responses = {};
    applyFilters();
    return;
  }
  state.responses = await fetchJSON(`/api/responses?annotator=${encodeURIComponent(state.annotatorId)}`);
  await refreshProgress();
  applyFilters();
}

// ── Filtering / navigation ──────────────────────────────────────────────

// Ordering only — the sample itself is never filtered by model agreement,
// so the reported gold-error rate stays an unbiased estimate over the full
// stratified sample (§8.3). Disagreements simply come first, model failures
// last (nothing for a human to adjudicate in those).
const MODEL_STATUS_PRIORITY = { disagrees: 0, agrees: 1, unavailable: 2 };

function applyFilters() {
  const typeFilter = el("type-filter").value;
  const onlyUnanswered = el("only-unanswered").checked;
  const filtered = state.items.filter((item) => {
    if (typeFilter && item.question_type !== typeFilter) return false;
    if (onlyUnanswered && state.responses[item.question_id]) return false;
    return true;
  });

  if (el("sort-order").value === "disagreements") {
    filtered.sort((a, b) =>
      (MODEL_STATUS_PRIORITY[a.model_status] ?? 2) - (MODEL_STATUS_PRIORITY[b.model_status] ?? 2));
  }
  state.filtered = filtered;
  moveTo(0);
}

function renderModelHint(item) {
  const wrap = el("model-hint");
  const useModelButton = el("use-model-btn");
  useModelButton.classList.toggle(
    "hidden", !(item.model_status === "disagrees" && item.model_answer));
  if (item.model_answer) useModelButton.title = `Save "${item.model_answer}" as the correction`;

  if (!state.modelHintsLoaded || !item.model_status || item.model_status === "unavailable") {
    wrap.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");
  const disagrees = item.model_status === "disagrees";
  el("model-hint-badge").textContent = disagrees ? "model disagrees" : "model agrees";
  el("model-hint-badge").className = disagrees ? "badge disagrees" : "badge agrees";
  el("model-hint-answer").textContent = `model said "${item.model_answer}" — a hint, not ground truth`;
  el("model-hint-reasoning").textContent = item.model_reasoning || "(no reasoning returned)";
  el("model-hint-reasoning").classList.add("hidden");
}

async function renderModelSummary() {
  const box = el("model-summary");
  if (!box) return;
  const summary = await fetchJSON("/api/model_summary");
  if (!summary.available) {
    box.innerHTML = "";
    return;
  }
  const rows = Object.entries(summary.by_type).sort()
    .map(([type, counts]) => {
      const rate = counts.agreement_rate === null ? "—" : `${(counts.agreement_rate * 100).toFixed(0)}%`;
      return `<div><span>${type}</span><span>${rate}</span></div>`;
    }).join("");
  box.innerHTML = `<div class="model-summary-title">Model agrees with gold</div>${rows}
    <p class="hint">Low agreement on depth-derived types mostly reflects the
    model guessing depth from one RGB frame — gold there comes from measured
    depth. Treat those disagreements as weak signal.</p>`;
}

function moveTo(index) {
  if (state.filtered.length === 0) {
    renderEmptyItem();
    return;
  }
  state.currentIndex = Math.max(0, Math.min(index, state.filtered.length - 1));
  renderCurrentItem();
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Bolds the object name(s) the question is actually about (item.highlight_words,
// computed server-side from that question's evidence — see audit_items.py's
// extract_highlight_words) so a reviewer can spot them at a glance. Longer
// phrases are matched first so "trash can" isn't split by a bare "can".
function questionHtmlWithHighlights(question, highlightWords) {
  const escapedQuestion = escapeHtml(question);
  if (!highlightWords || highlightWords.length === 0) return escapedQuestion;
  const alternation = [...highlightWords]
    .sort((a, b) => b.length - a.length)
    .map((word) => escapeRegExp(escapeHtml(word)))
    .join("|");
  const pattern = new RegExp(`\\b(${alternation})\\b`, "gi");
  return escapedQuestion.replace(pattern, "<strong>$1</strong>");
}

function renderEmptyItem() {
  el("item-position").textContent = "no items match this filter";
  el("question-text").textContent = "";
  el("rgb-image").removeAttribute("src");
  clearOverlay();
  el("gold-block").classList.add("hidden");
}

// ── Rendering one item ───────────────────────────────────────────────────

async function renderCurrentItem() {
  const item = state.filtered[state.currentIndex];
  el("gold-block").classList.remove("hidden");
  el("item-position").textContent =
    `item ${state.currentIndex + 1} of ${state.filtered.length} (${item.question_type})`;
  el("meta-type").textContent = `type: ${item.question_type}`;
  el("meta-sensor").textContent = `sensor: ${item.sensor ?? "?"}`;
  el("meta-scene").textContent = `scene: ${item.scene_type ?? "?"}`;
  el("meta-qid").textContent = `id: ${item.question_id}`;
  el("question-text").innerHTML = questionHtmlWithHighlights(item.question, item.highlight_words);
  el("gold-answer-text").textContent = item.answer;
  renderModelHint(item);

  el("image-loading").classList.remove("hidden");
  el("rgb-image").src = `/api/image/${item.image_id}`;
  loadOverlay(item);

  const existing = state.responses[item.question_id];
  el("notes-input").value = existing ? existing.notes : "";

  document.querySelectorAll("button.verdict").forEach((button) => {
    button.classList.toggle("selected", existing && button.dataset.verdict === existing.verdict);
  });

  if (existing && existing.verdict === "incorrect") {
    openCorrectionBox(existing.own_answer_raw || existing.own_answer);
    renderSavedIndicator(existing);
  } else {
    closeCorrectionBox();
    el("saved-indicator").classList.add("hidden");
  }
}

function openCorrectionBox(prefillValue) {
  el("correction-block").classList.remove("hidden");
  el("own-answer-input").value = prefillValue || "";
  el("own-answer-input").focus();
}

function closeCorrectionBox() {
  el("correction-block").classList.add("hidden");
  el("own-answer-input").value = "";
}

function renderSavedIndicator(saved) {
  const indicator = el("saved-indicator");
  indicator.classList.remove("hidden");
  if (saved.verdict !== "incorrect") {
    indicator.innerHTML = `Saved: <strong>${saved.verdict}</strong>`;
    return;
  }
  const correctionText = saved.own_answer
    ? (saved.was_corrected
        ? `Saved: <strong>${saved.own_answer}</strong> (auto-corrected from "${saved.own_answer_raw}")`
        : `Saved: <strong>${saved.own_answer}</strong>`)
    : "Saved (no correction typed)";
  indicator.innerHTML = `${correctionText} · <a href="#" id="edit-correction-link">Edit</a>`;
  el("edit-correction-link").addEventListener("click", (event) => {
    event.preventDefault();
    openCorrectionBox(saved.own_answer_raw || saved.own_answer);
  });
}

function loadOverlay(item) {
  const objectsParam = (item.evidence_object_indices || []).join(",");
  fetchJSON(`/api/polygons/${item.image_id}?objects=${objectsParam}`)
    .then((data) => {
      el("rgb-image").onload = () => {
        el("image-loading").classList.add("hidden");
        drawOverlay(data);
      };
      if (el("rgb-image").complete) drawOverlay(data);
    })
    .catch(() => clearOverlay());
}

function clearOverlay() {
  const canvas = el("overlay-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawOverlay(data) {
  const img = el("rgb-image");
  const canvas = el("overlay-canvas");
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  const scaleX = img.clientWidth / data.image_width;
  const scaleY = img.clientHeight / data.image_height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#39ff6a";
  ctx.fillStyle = "rgba(57, 255, 106, 0.18)";
  ctx.font = "13px sans-serif";

  for (const polygon of data.polygons) {
    if (polygon.x.length < 3) continue;
    ctx.beginPath();
    ctx.moveTo(polygon.x[0] * scaleX, polygon.y[0] * scaleY);
    for (let i = 1; i < polygon.x.length; i++) {
      ctx.lineTo(polygon.x[i] * scaleX, polygon.y[i] * scaleY);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#39ff6a";
    ctx.fillText(polygon.name, polygon.x[0] * scaleX, Math.max(12, polygon.y[0] * scaleY - 6));
    ctx.fillStyle = "rgba(57, 255, 106, 0.18)";
  }
}

// ── Answering ────────────────────────────────────────────────────────────

// "Incorrect" only opens the correction box — it does not save by itself,
// so the annotator gets a chance to type what they think is right first
// (and to re-edit later: see renderSavedIndicator's "Edit" link).
// "Correct"/"Ambiguous" need no correction and save immediately.
function onVerdictClicked(verdict) {
  if (state.filtered.length === 0) return;
  if (verdict === "incorrect") {
    const item = state.filtered[state.currentIndex];
    const existing = state.responses[item.question_id];
    openCorrectionBox(existing && existing.verdict === "incorrect" ? (existing.own_answer_raw || existing.own_answer) : "");
    document.querySelectorAll("button.verdict").forEach((button) => {
      button.classList.toggle("selected", button.dataset.verdict === "incorrect");
    });
    return;
  }
  closeCorrectionBox();
  saveResponse(verdict, "");
}

function saveCorrection() {
  if (state.filtered.length === 0) return;
  saveResponse("incorrect", el("own-answer-input").value);
}

// One-click shortcut for "gold is wrong and the model got it right": records
// verdict=incorrect with the model's answer as the correction, so it does not
// have to be retyped. Offered only when the model actually disagrees with gold
// — proposing the model's answer while it matches gold would be self-
// contradictory. Like every other button here, this writes only to the audit
// log; it does not edit the released dataset.
function saveModelAnswerAsCorrection() {
  if (state.filtered.length === 0) return;
  const item = state.filtered[state.currentIndex];
  if (!item.model_answer || item.model_status !== "disagrees") return;
  saveResponse("incorrect", item.model_answer);
}

async function saveResponse(verdict, ownAnswer) {
  if (!state.annotatorId) {
    alert("Enter an annotator ID first.");
    return;
  }
  const item = state.filtered[state.currentIndex];
  const body = {
    question_id: item.question_id,
    annotator_id: state.annotatorId,
    own_answer: ownAnswer,
    verdict,
    notes: el("notes-input").value,
  };
  const result = await fetchJSON("/api/response", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.responses[item.question_id] = {
    question_id: item.question_id,
    annotator_id: state.annotatorId,
    verdict,
    notes: body.notes,
    answered_at_utc: new Date().toISOString(),
    ...result.saved,
  };
  document.querySelectorAll("button.verdict").forEach((button) => {
    button.classList.toggle("selected", button.dataset.verdict === verdict);
  });
  renderSavedIndicator(state.responses[item.question_id]);
  if (verdict === "incorrect") el("own-answer-input").value = result.saved.own_answer;
  await refreshProgress();

  const onlyUnanswered = el("only-unanswered").checked;
  setTimeout(() => {
    if (onlyUnanswered) applyFilters();
    else moveTo(state.currentIndex + 1);
  }, 250);
}

function onGlobalKeydown(event) {
  if (document.activeElement === el("own-answer-input") || document.activeElement === el("notes-input")) return;
  if (event.key === "1") onVerdictClicked("correct");
  if (event.key === "2") onVerdictClicked("incorrect");
  if (event.key === "3") onVerdictClicked("ambiguous");
  if (event.key === "4") saveModelAnswerAsCorrection();
}

// ── Progress ─────────────────────────────────────────────────────────────

async function refreshProgress() {
  if (!state.annotatorId) {
    el("progress-text").textContent = "— / —";
    el("progress-bar-fill").style.width = "0%";
    el("type-progress-list").innerHTML = "";
    return;
  }
  const progress = await fetchJSON(`/api/progress?annotator=${encodeURIComponent(state.annotatorId)}`);
  el("progress-text").textContent = `${progress.answered} / ${progress.total} answered`;
  const pct = progress.total ? (100 * progress.answered) / progress.total : 0;
  el("progress-bar-fill").style.width = `${pct}%`;

  const list = el("type-progress-list");
  list.innerHTML = "";
  for (const [type, counts] of Object.entries(progress.by_type).sort()) {
    const row = document.createElement("div");
    row.innerHTML = `<span>${type}</span><span>${counts.answered}/${counts.total}</span>`;
    list.appendChild(row);
  }
}

// ── Stats panel ──────────────────────────────────────────────────────────

async function openStats() {
  const stats = await fetchJSON("/api/stats");
  const tbody = document.querySelector("#stats-table tbody");
  tbody.innerHTML = "";
  const pct = (value) => (value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`);
  for (const row of stats.types) {
    const tr = document.createElement("tr");
    const acceptCell = row.meets_acceptance === null ? "—" : (row.meets_acceptance ? "yes" : "no");
    tr.innerHTML = `<td>${row.question_type}</td><td>${row.n_sampled}</td><td>${row.n_verdicts}</td>
      <td>${pct(row.gold_accuracy)}</td><td>${pct(row.human_accuracy_vs_gold)}</td>
      <td>${pct(row.ambiguous_share)}</td><td>${acceptCell}</td>`;
    tbody.appendChild(tr);
  }
  el("stats-panel").classList.remove("hidden");
}

init().catch((error) => {
  el("status-banner").textContent = `Failed to start: ${error.message}`;
  el("status-banner").classList.remove("hidden");
});
