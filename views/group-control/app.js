const bridge = window.AstrBotPluginView;
const $ = (id) => document.getElementById(id);
const selected = new Set();
let data = { groups: [], platforms: [], warnings: [] };
let busy = false;
const key = (row) => JSON.stringify([row.platform_id, row.group_id]);

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}

function visibleRows() {
  const query = $("search").value.trim().toLowerCase();
  return data.groups.filter((r) => `${r.name} ${r.group_id} ${r.platform_id}`.toLowerCase().includes(query));
}

function batchState() {
  const eligible = visibleRows().filter((r) => r.available);
  const count = eligible.filter((r) => selected.has(key(r))).length;
  $("select-all").checked = eligible.length > 0 && count === eligible.length;
  $("select-all").indeterminate = count > 0 && count < eligible.length;
  $("select-all").disabled = busy || !eligible.length;
  for (const id of ["enable-selected", "disable-selected"]) $(id).disabled = busy || !selected.size;
}

function element(tag, className, text) {
  const el = document.createElement(tag);
  el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function render() {
  const rows = visibleRows();
  const container = $("groups");
  container.replaceChildren();
  $("count").textContent = `${data.groups.length} 个群 · ${data.groups.filter((r) => r.enabled).length} 个允许发言`;
  if (!rows.length) container.append(element("div", "empty", data.groups.length ? "没有匹配的群聊" : "暂无群聊，点击刷新或在下方添加群号。"));
  for (const row of rows) {
    const item = element("div", "group");
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = selected.has(key(row));
    check.disabled = busy || !row.available;
    check.setAttribute("aria-label", `选择群 ${row.group_id}`);
    check.onchange = () => { check.checked ? selected.add(key(row)) : selected.delete(key(row)); batchState(); };
    const identity = element("div", "identity");
    identity.append(element("div", "name", row.name || `群 ${row.group_id}`));
    identity.append(element("div", "meta", `${row.group_id} · ${row.platform_id}`));
    const flags = [];
    if (!row.available) flags.push("平台未加载，连接后可操作");
    if (!row.native_enabled) flags.push("原生消息处理已关闭，请先在「自定义规则」恢复一次");
    if (row.owner_paused) flags.push("主人暂停仍生效，开启总开关后依然只回应主人");
    if (flags.length) identity.append(element("div", "flags", flags.join("；")));
    const toggle = element("button", "switch");
    toggle.type = "button";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-label", `群 ${row.group_id} 允许发言`);
    toggle.setAttribute("aria-checked", String(row.enabled));
    toggle.disabled = busy || !row.available;
    toggle.append(element("span", "", row.enabled ? "允许发言" : "已关闭"), element("span", "track"));
    toggle.onclick = () => switchGroups([row], !row.enabled);
    item.append(check, identity, toggle);
    container.append(item);
  }
  $("refresh").disabled = busy;
  $("add-button").disabled = busy || !data.platforms.length;
  batchState();
}

function accept(snapshot) {
  data = snapshot;
  const available = new Set(data.groups.filter((r) => r.available).map(key));
  for (const k of selected) if (!available.has(k)) selected.delete(k);
  const previous = $("platform").value;
  $("platform").replaceChildren(...data.platforms.map((p) => new Option(p.id, p.id)));
  if (data.platforms.some((p) => p.id === previous)) $("platform").value = previous;
}

async function run(operation, message) {
  if (busy) return;
  busy = true;
  render();
  notice("正在处理…");
  try {
    accept(await operation());
    notice([message, ...data.warnings].filter(Boolean).join("\n"), data.warnings.length > 0);
  } catch (error) {
    notice(`未能确认操作结果：${error.message}。请刷新核对。`, true);
  } finally {
    busy = false;
    render();
  }
}

async function switchGroups(rows, enabled) {
  await run(() => bridge.apiPost("groups/switch", {
    enabled, groups: rows.map((r) => ({ platform_id: r.platform_id, group_id: r.group_id })),
  }), `${rows.length} 个群已${enabled ? "允许发言" : "关闭发言"}，已保存。`);
}

$("search").oninput = render;
$("select-all").onchange = () => {
  for (const row of visibleRows().filter((r) => r.available)) {
    $("select-all").checked ? selected.add(key(row)) : selected.delete(key(row));
  }
  render();
};
$("enable-selected").onclick = () => switchGroups(data.groups.filter((r) => selected.has(key(r))), true);
$("disable-selected").onclick = () => switchGroups(data.groups.filter((r) => selected.has(key(r))), false);
$("refresh").onclick = () => run(() => bridge.apiPost("groups/refresh", {}), "群列表已刷新。");
$("add-group").onsubmit = (event) => {
  event.preventDefault();
  switchGroups([{ platform_id: $("platform").value, group_id: $("group-id").value.trim() }], false);
};

try {
  await bridge.ready();
  await run(() => bridge.apiGet("groups"), "开关状态已加载。");
} catch (error) {
  notice(`请从 AstrBot 的「群聊控制」视图打开面板。${error.message}`, true);
}
