// Demo UI. The chat lives in the browser; the assistant is called through /api/suggest.

const byId = (id) => document.getElementById(id);

const dom = {
  company: byId("company"),
  status: byId("status"),
  statusText: byId("status-text"),
  clientName: byId("client-name"),
  chatName: byId("chat-name"),
  chatAvatar: byId("chat-avatar"),
  examples: byId("examples"),
  reset: byId("reset"),
  feed: byId("feed"),
  composer: byId("composer"),
  composerInput: byId("composer-input"),
  assistant: byId("assistant"),
};

const state = {
  meta: null,
  messages: [], // { author: "client" | "manager", text, at: Date }
  phase: "idle", // idle | loading | ready | error | sent
  suggestion: null,
  error: null,
  controller: null,
  startedAt: 0,
};

const TYPE_LABELS = { product: "Товар", policy: "Условия магазина", promo: "Акция" };
const STATUS_POLL_MS = 15_000;
const timeFormat = new Intl.DateTimeFormat("ru-RU", { hour: "2-digit", minute: "2-digit" });
const priceFormat = new Intl.NumberFormat("ru-RU", {
  style: "currency",
  currency: "RUB",
  maximumFractionDigits: 0,
});

/** Minimal element builder. Children are inserted as text nodes, never parsed as HTML. */
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "value") node.value = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const clientName = () => dom.clientName.value.trim();

// ---------- Data ----------

async function loadMeta() {
  try {
    const response = await fetch("api/meta");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.meta = await response.json();
  } catch {
    dom.company.textContent = "Не удалось загрузить настройки";
    return;
  }
  const { company, examples } = state.meta;
  dom.company.textContent = `${company.name} — ${company.description}`;
  document.title = `${company.name} · Sales Reply Assistant`;
  dom.examples.replaceChildren(
    ...examples.map((example) =>
      h(
        "button",
        { class: "example", type: "button", onclick: () => sendClientMessage(example.text) },
        h("span", { class: "example__title" }, example.title),
        h("span", { class: "example__text" }, example.text),
      ),
    ),
  );
}

async function refreshStatus() {
  let status;
  try {
    status = await (await fetch("api/status")).json();
  } catch {
    status = { reachable: false, detail: "Сервер приложения недоступен" };
  }
  const model = state.meta?.chat_model ?? "модель";
  const where = state.meta?.local_inference ? "локально" : "облако";
  let view;
  if (!status.reachable) {
    view = ["error", "Сервер моделей недоступен"];
  } else if (status.chat_model_available === false || status.embedding_model_available === false) {
    view = ["warning", `${model}: модель не найдена на сервере`];
  } else {
    view = ["ok", `${model} · ${where}`];
  }
  [dom.status.dataset.state, dom.statusText.textContent] = view;
  dom.status.title = status.detail ?? "";
}

async function requestSuggestion() {
  state.controller?.abort();
  const controller = new AbortController();
  Object.assign(state, {
    controller,
    phase: "loading",
    suggestion: null,
    error: null,
    startedAt: performance.now(),
  });
  renderAssistant();
  const ticker = setInterval(renderElapsed, 100);

  try {
    const response = await fetch("api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: state.messages.map(({ author, text }) => ({ author, text })),
        client_name: clientName() || null,
      }),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorText(response.status, data));
    Object.assign(state, { phase: "ready", suggestion: data });
  } catch (error) {
    if (error.name === "AbortError") return; // a newer request has replaced this one
    Object.assign(state, { phase: "error", error: error.message });
  } finally {
    clearInterval(ticker);
    if (state.controller === controller) state.controller = null;
  }
  renderAssistant();
}

function errorText(status, data) {
  if (typeof data.detail === "string") return data.detail;
  if (status === 422) return "Некорректный запрос: последнее сообщение должно быть от клиента.";
  return `Ошибка сервера (${status}).`;
}

// ---------- Actions ----------

function sendClientMessage(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  state.messages.push({ author: "client", text: trimmed, at: new Date() });
  renderFeed();
  requestSuggestion();
}

function sendManagerReply(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  state.messages.push({ author: "manager", text: trimmed, at: new Date() });
  Object.assign(state, { phase: "sent", suggestion: null });
  renderFeed();
  renderAssistant();
}

function resetDialog() {
  state.controller?.abort();
  Object.assign(state, { messages: [], phase: "idle", suggestion: null, error: null });
  renderFeed();
  renderAssistant();
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    flashLabel(button, "Скопировано");
  } catch {
    flashLabel(button, "Не удалось скопировать");
  }
}

function flashLabel(button, label) {
  const original = button.textContent;
  button.textContent = label;
  setTimeout(() => (button.textContent = original), 1500);
}

// ---------- Rendering ----------

function renderFeed() {
  const name = clientName() || "Клиент";
  dom.chatName.textContent = name;
  dom.chatAvatar.textContent = name[0].toUpperCase();

  if (!state.messages.length) {
    dom.feed.replaceChildren(
      h(
        "div",
        { class: "chat__empty" },
        "Диалог пуст. Выберите сценарий слева или напишите сообщение от имени клиента.",
      ),
    );
    return;
  }
  dom.feed.replaceChildren(
    ...state.messages.map((message) =>
      h(
        "div",
        { class: `message message--${message.author}` },
        h("div", { class: "message__bubble" }, message.text),
        h(
          "div",
          { class: "message__meta" },
          `${message.author === "client" ? name : "Менеджер"} · ${timeFormat.format(message.at)}`,
        ),
      ),
    ),
  );
  dom.feed.scrollTop = dom.feed.scrollHeight;
}

function renderAssistant() {
  const views = {
    idle: () =>
      placeholder("Когда клиент напишет, здесь появится черновик ответа и подсказка по допродаже."),
    sent: () => placeholder("Ответ отправлен. Жду следующее сообщение клиента."),
    loading: renderLoading,
    error: renderError,
    ready: () => renderSuggestion(state.suggestion),
  };
  dom.assistant.replaceChildren(...views[state.phase]());
}

function placeholder(text) {
  return [
    h(
      "div",
      { class: "placeholder" },
      h("div", { class: "placeholder__icon", "aria-hidden": "true" }, "✦"),
      h("p", {}, text),
    ),
  ];
}

function renderLoading() {
  return [
    h(
      "div",
      { class: "loading", role: "status" },
      h("span", { class: "spinner", "aria-hidden": "true" }),
      h(
        "div",
        {},
        h("div", {}, "Ищу в базе знаний и готовлю черновик…"),
        h("div", { class: "small", id: "elapsed" }, "0.0 с"),
      ),
    ),
    ...[92, 78, 85, 60].map((width) => h("div", { class: "skeleton", style: `width: ${width}%` })),
  ];
}

function renderElapsed() {
  const node = byId("elapsed");
  if (node) node.textContent = `${((performance.now() - state.startedAt) / 1000).toFixed(1)} с`;
}

function renderError() {
  return [
    alertBox("danger", "Не получилось подготовить ответ", state.error),
    h(
      "div",
      { class: "card__actions" },
      h(
        "button",
        { class: "button button--primary", type: "button", onclick: requestSuggestion },
        "Повторить",
      ),
    ),
  ];
}

function renderSuggestion(suggestion) {
  const draft = h("textarea", {
    class: "input draft",
    "aria-label": "Черновик ответа клиенту",
    value: suggestion.reply,
  });

  return [
    h(
      "div",
      { class: "intent" },
      h("span", { class: "intent__label" }, "Суть запроса"),
      h("span", { class: "intent__text" }, suggestion.client_intent),
    ),
    suggestion.needs_attention &&
      alertBox(
        "warning",
        "Нужно внимание менеджера",
        suggestion.attention_reason ?? "Проверьте ответ перед отправкой.",
      ),
    ...suggestion.compliance_warnings.map((warning) =>
      alertBox("danger", "Проверьте формулировку", warning),
    ),
    h(
      "section",
      { class: "card" },
      h(
        "div",
        { class: "card__head" },
        h("h3", { class: "card__title" }, "Ответ клиенту"),
        h("span", { class: "muted small" }, "можно отредактировать"),
      ),
      draft,
      h(
        "div",
        { class: "card__actions" },
        h(
          "button",
          {
            class: "button button--primary",
            type: "button",
            onclick: () => sendManagerReply(draft.value),
          },
          "Отправить клиенту",
        ),
        h(
          "button",
          {
            class: "button button--ghost",
            type: "button",
            onclick: (event) => copyText(draft.value, event.currentTarget),
          },
          "Скопировать",
        ),
        h(
          "button",
          { class: "button button--ghost", type: "button", onclick: requestSuggestion },
          "Другой вариант",
        ),
      ),
    ),
    renderUpsell(suggestion.upsell, draft),
    suggestion.sources.length > 0 &&
      h(
        "section",
        { class: "card" },
        h("h3", { class: "card__title" }, "Источники из базы знаний"),
        h(
          "div",
          { class: "chips" },
          suggestion.sources.map((source) =>
            h("span", { class: "chip", title: TYPE_LABELS[source.type] }, source.title),
          ),
        ),
      ),
    h(
      "div",
      { class: "footer-meta" },
      h("span", {}, suggestion.model),
      h("span", {}, `${(suggestion.latency_ms / 1000).toFixed(1)} с`),
    ),
  ].filter(Boolean);
}

function renderUpsell(upsell, draft) {
  if (!upsell.should_offer) {
    return h(
      "section",
      { class: "card card--hold" },
      h("h3", { class: "card__title" }, "Сейчас не допродавать"),
      h("p", { class: "card__text muted" }, upsell.hint),
    );
  }

  const addPitch = (event) => {
    draft.value = `${draft.value.trim()}\n\n${upsell.pitch}`;
    event.currentTarget.disabled = true;
    event.currentTarget.textContent = "Фраза добавлена в ответ";
  };

  return h(
    "section",
    { class: "card card--upsell" },
    h("h3", { class: "card__title" }, "Можно предложить"),
    upsell.products.length > 0 &&
      h(
        "div",
        { class: "chips" },
        upsell.products.map((product) =>
          h(
            "span",
            { class: "chip" },
            product.title,
            product.price_rub && h("span", { class: "chip__price" }, priceFormat.format(product.price_rub)),
          ),
        ),
      ),
    h("p", { class: "card__text" }, upsell.hint),
    upsell.pitch && h("blockquote", { class: "pitch" }, upsell.pitch),
    upsell.pitch &&
      h(
        "div",
        { class: "card__actions" },
        h(
          "button",
          { class: "button button--ghost button--small", type: "button", onclick: addPitch },
          "Добавить фразу в ответ",
        ),
      ),
  );
}

function alertBox(kind, title, text) {
  return h(
    "div",
    { class: `alert alert--${kind}`, role: "alert" },
    h("span", { class: "alert__title" }, title),
    h("span", {}, text),
  );
}

// ---------- Wiring ----------

dom.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendClientMessage(dom.composerInput.value);
  dom.composerInput.value = "";
});

dom.composerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    dom.composer.requestSubmit();
  }
});

dom.clientName.addEventListener("input", renderFeed);
dom.reset.addEventListener("click", resetDialog);

renderFeed();
renderAssistant();
await loadMeta();
await refreshStatus();
setInterval(refreshStatus, STATUS_POLL_MS);
