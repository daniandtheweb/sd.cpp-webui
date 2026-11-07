"""sd.cpp-webui - UI component for the list of installed models"""

import gradio as gr

# Its value is a JSON text {folder, models, message, ok}; models are
# {name, size, modified, bytes, time}. A click on a button of a row sends
# {action, name, new_name} to Python, which answers with a new value.
CSS = """
.model-list { width: 100%; }
.model-list .ml-bar { display: flex; gap: 12px; align-items: center;
                      margin-bottom: 8px; }
.model-list .ml-bar input { flex: 1; padding: 6px 10px; border-radius: 6px;
    border: 1px solid var(--border-color-primary);
    background: var(--input-background-fill);
    color: var(--body-text-color); }
.model-list .ml-count { opacity: 0.7; white-space: nowrap; }
.model-list .ml-message { margin-bottom: 8px; padding: 6px 10px;
    border-radius: 6px; background: var(--background-fill-secondary); }
.model-list .ml-message.err { color: var(--error-text-color, #c0392b); }
.model-list table { width: 100%; border-collapse: collapse; }
.model-list th, .model-list td { text-align: left; padding: 6px 10px;
    border-bottom: 1px solid var(--border-color-primary); }
.model-list th { cursor: pointer; user-select: none; }
.model-list td.ml-num { white-space: nowrap; opacity: 0.8; }
.model-list td.ml-name { word-break: break-all; }
.model-list .ml-dir { opacity: 0.55; }
.model-list .ml-actions { white-space: nowrap; text-align: right; }
.model-list button { padding: 3px 10px; margin-left: 6px; cursor: pointer;
    border-radius: 6px; border: 1px solid var(--border-color-primary);
    background: var(--button-secondary-background-fill);
    color: var(--button-secondary-text-color); }
.model-list button.danger { background: var(--button-cancel-background-fill,
    #c0392b); color: var(--button-cancel-text-color, #fff); }
.model-list td.ml-name input { width: 70%; padding: 3px 8px;
    border-radius: 6px; border: 1px solid var(--border-color-primary);
    background: var(--input-background-fill);
    color: var(--body-text-color); }
.model-list .ml-empty { padding: 16px; opacity: 0.7; }
"""

JS = """
const root = element.querySelector('.model-list') || element;
const COLUMNS = [['name', 'Name'], ['bytes', 'Size'], ['time', 'Modified']];
const CONFIRM_MS = 4000;

let data = { folder: '', models: [], message: '', ok: true };
let filter = '';
let order = { key: 'name', down: false };
let renaming = null;      // name of the model being renamed
let armed = null;         // name whose Delete was clicked once
let timer = null;

let rows = null;
let count = null;
let note = null;
let heads = {};

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function send(action, name, newName) {
  armed = null;
  renaming = null;
  trigger('click', { action, name, new_name: newName || '' });
}

function build() {
  if (rows && root.contains(rows)) return;
  root.innerHTML = '';
  const bar = make('div', 'ml-bar');
  const box = make('input');
  box.type = 'search';
  box.placeholder = 'Filter by name';
  box.addEventListener('input', () => {
    filter = box.value.toLowerCase();
    renderRows();
  });
  count = make('span', 'ml-count');
  bar.append(box, count);
  note = make('div', 'ml-message');
  const table = make('table');
  const head = make('tr');
  for (const [key, label] of COLUMNS) {
    const cell = make('th', '', label);
    cell.addEventListener('click', () => {
      order = { key, down: order.key === key ? !order.down : false };
      renderRows();
    });
    heads[key] = cell;
    head.appendChild(cell);
  }
  head.appendChild(make('th'));
  const top = make('thead');
  top.appendChild(head);
  rows = make('tbody');
  table.append(top, rows);
  root.append(bar, note, table);
}

function split(name) {
  const slash = name.lastIndexOf('/');
  const dot = name.lastIndexOf('.');
  return {
    dir: name.slice(0, slash + 1),
    stem: name.slice(slash + 1, dot > slash ? dot : name.length),
    ext: dot > slash ? name.slice(dot) : '',
  };
}

function nameCell(model) {
  const cell = make('td', 'ml-name');
  const { dir, stem, ext } = split(model.name);
  if (dir) cell.appendChild(make('span', 'ml-dir', dir));
  if (renaming !== model.name) {
    cell.append(stem + ext);
    return cell;
  }
  const field = make('input');
  field.value = stem;
  field.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') send('rename', model.name, field.value);
    if (event.key === 'Escape') { renaming = null; renderRows(); }
  });
  cell.append(field, ext);
  setTimeout(() => { field.focus(); field.select(); }, 0);
  return cell;
}

function button(label, action, className) {
  const node = make('button', className, label);
  node.type = 'button';
  node.addEventListener('click', action);
  return node;
}

function actionsCell(model) {
  const cell = make('td', 'ml-actions');
  if (renaming === model.name) {
    const field = () => rows.querySelector('input');
    cell.append(
      button('Save', () => send('rename', model.name, field().value)),
      button('Cancel', () => { renaming = null; renderRows(); }));
    return cell;
  }
  const sure = armed === model.name;
  cell.append(
    button('Rename', () => {
      armed = null;
      renaming = model.name;
      renderRows();
    }),
    button(sure ? 'Click again to delete' : 'Delete', () => {
      if (armed === model.name) return send('remove', model.name);
      armed = model.name;
      clearTimeout(timer);
      timer = setTimeout(() => { armed = null; renderRows(); }, CONFIRM_MS);
      renderRows();
    }, sure ? 'danger' : ''));
  return cell;
}

function renderRows() {
  build();
  const shown = data.models
    .filter((m) => m.name.toLowerCase().includes(filter))
    .sort((a, b) => {
      const x = a[order.key];
      const y = b[order.key];
      const way = typeof x === 'string' ? x.localeCompare(y) : x - y;
      return order.down ? -way : way;
    });
  for (const [key, label] of COLUMNS) {
    heads[key].textContent = label
      + (order.key === key ? (order.down ? ' ▼' : ' ▲') : '');
  }
  count.textContent = `${shown.length} of ${data.models.length} models`;
  note.textContent = data.message;
  note.className = 'ml-message' + (data.ok ? '' : ' err');
  note.style.display = data.message ? '' : 'none';
  rows.innerHTML = '';
  if (!shown.length) {
    const cell = make('td', 'ml-empty', data.models.length
      ? 'No model matches the filter.'
      : 'No models here: ' + (data.folder || 'no folder is set'));
    cell.colSpan = 4;
    const row = make('tr');
    row.appendChild(cell);
    rows.appendChild(row);
    return;
  }
  for (const model of shown) {
    const row = make('tr');
    row.append(nameCell(model),
      make('td', 'ml-num', model.size),
      make('td', 'ml-num', model.modified),
      actionsCell(model));
    rows.appendChild(row);
  }
}

function refresh() {
  try {
    const value = JSON.parse(props.value);
    if (value && Array.isArray(value.models)) data = value;
  } catch (e) { /* keep what is shown */ }
  renaming = null;
  armed = null;
  renderRows();
}

watch('value', refresh);
refresh();
"""


def create_model_list(value: str) -> gr.HTML:
    """The list of models: filter, sort, rename and delete in each row."""
    return gr.HTML(
        value,
        html_template='<div class="model-list"></div>',
        css_template=CSS,
        js_on_load=JS,
        label="Models"
    )
