// The pose editor: the body of gr.HTML's js_on_load (see editor.py).
//
// props.value is a JSON text {w, h, photo, box, people}: the size of the
// canvas, a data URL of the photo (or '') with the box it fills (two
// corners, 0..1 of the canvas), and the people, each a list of the 18
// OpenPose points as [x, y] in 0..1 (or null when the point is missing).
// The editor changes props.value when a gesture ends; Python reads it when
// the Save button is pressed. props.limbs and props.colors come from
// skeleton.py, so the editor draws what the saved image will show.
//
// The skeletons are listed next to the canvas: click one to select it,
// and each has its own size slider and delete button; "Add" makes a new
// one.

const LIMBS = props.limbs;
const COLORS = props.colors.map((c) => `rgb(${c[0]},${c[1]},${c[2]})`);

// A standing person: x from the body's center, y from its top, in units of
// the body height. Same order as OpenPose's points.
const TEMPLATE = [
  [0, 0.08], [0, 0.18], [-0.10, 0.20], [-0.14, 0.34], [-0.15, 0.47],
  [0.10, 0.20], [0.14, 0.34], [0.15, 0.47], [-0.06, 0.50], [-0.07, 0.73],
  [-0.07, 0.96], [0.06, 0.50], [0.07, 0.73], [0.07, 0.96], [-0.02, 0.06],
  [0.02, 0.06], [-0.05, 0.08], [0.05, 0.08],
];
const BODY_HEIGHT = 0.8;     // of the image, for a new person
const GRAB = 12;             // pixels around a point that grab it
const MAX_SIDE = 640;        // of the canvas

// <pure>
// Geometry that doesn't touch the page (tested with node, see
// test_pose.py). People are lists of [x, y] in 0..1 or null.
const MIN_SIDE = 8;
const MAX_CANVAS = 8192;

// A size typed in a box: a whole number from MIN_SIDE up, or null while it
// is not one yet (the user may be in the middle of typing).
function canvasSide(text) {
  const value = Math.round(Number(text));
  if (!Number.isFinite(value) || value < MIN_SIDE) return null;
  return Math.min(MAX_CANVAS, value);
}

// A point on a canvas of another size: it keeps its place in pixels
// relative to the center, so a resize never stretches anything.
function movePoint(p, oldW, oldH, newW, newH) {
  return [
    (p[0] * oldW + (newW - oldW) / 2) / newW,
    (p[1] * oldH + (newH - oldH) / 2) / newH,
  ];
}

// The people on a canvas of another size (points may end up outside it;
// make the skeleton smaller to bring them back).
function resizeCanvas(people, oldW, oldH, newW, newH) {
  return people.map((person) => person.map(
    (p) => p && movePoint(p, oldW, oldH, newW, newH)));
}

// One person scaled by factor around the middle of its points, in
// pixels, so its proportions stay.
function scalePerson(person, w, h, factor) {
  const all = person.filter((p) => p);
  if (!all.length) return person;
  const xs = all.map((p) => p[0] * w);
  const ys = all.map((p) => p[1] * h);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  return person.map((p) => p && [
    (cx + (p[0] * w - cx) * factor) / w,
    (cy + (p[1] * h - cy) * factor) / h,
  ]);
}

function anyOutside(people) {
  return people.flat().some(
    (p) => p && (p[0] < 0 || p[0] > 1 || p[1] < 0 || p[1] > 1));
}
// </pure>

let state = { w: 512, h: 512, photo: '', box: null, people: [] };
let selected = 0;
let drag = null;
let photo = null;            // the loaded Image of state.photo
let photoSource = '';
let showPhoto = true;        // the toggle: only the preview, never saved

const root = element.querySelector('.pose-editor') || element;

function parse() {
  try {
    const value = JSON.parse(props.value);
    if (value && Array.isArray(value.people)) {
      state = {
        w: value.w || 512, h: value.h || 512,
        photo: value.photo || '',
        box: Array.isArray(value.box) && value.box.length === 4
          ? value.box : null,
        people: value.people,
      };
    }
  } catch (e) { /* keep the state */ }
  selected = Math.min(selected, Math.max(0, state.people.length - 1));
}

function commit() {
  props.value = JSON.stringify(state);
}

// ---------------------------------------------------------------- the UI

let canvas = null;
let hint = null;
let warning = null;
let widthBox = null;
let heightBox = null;
let photoBox = null;
let list = null;
let rows = [];               // the list's rows, rebuilt when the count changes
let rowsFor = -1;            // how many people the rows were built for
let scaleStart = null;       // {n, person}: the person when its slider was grabbed
let listening = false;

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(label, action, title) {
  const node = make('button', '', label);
  node.type = 'button';
  if (title) node.title = title;
  node.addEventListener('click', (event) => {
    event.stopPropagation();
    action();
  });
  return node;
}

function numberBox(label) {
  const box = make('input');
  box.type = 'number';
  box.min = MIN_SIDE;
  box.max = MAX_CANVAS;
  box.step = 8;
  box.title = label;
  box.style.width = '6em';
  return box;
}

function build() {
  if (canvas && root.contains(canvas)) return;
  root.innerHTML = '';
  const bar = make('div', 'pose-bar');

  const sizes = make('span');
  widthBox = numberBox('Canvas width');
  heightBox = numberBox('Canvas height');
  sizes.append('Canvas ', widthBox, ' × ', heightBox);
  bar.appendChild(sizes);
  widthBox.addEventListener('input', onSize);
  heightBox.addEventListener('input', onSize);

  const label = make('label');
  photoBox = make('input');
  photoBox.type = 'checkbox';
  photoBox.checked = showPhoto;
  photoBox.addEventListener('change', () => {
    showPhoto = photoBox.checked;
    draw();
  });
  label.append(photoBox, ' Show the photo');
  bar.appendChild(label);

  hint = make('span', 'pose-hint',
    'Drag a point. Shift+drag moves the whole skeleton. '
    + 'Right-click removes a point; click its faint circle to bring it back.');
  bar.appendChild(hint);
  warning = make('div', 'pose-hint');
  canvas = make('canvas');
  list = make('div', 'pose-list');

  const stage = make('div', 'pose-stage');
  stage.appendChild(canvas);
  const wrap = make('div', 'pose-wrap');
  wrap.append(stage, list);
  root.append(bar, warning, wrap);
  rowsFor = -1;

  canvas.addEventListener('mousedown', onDown);
  canvas.addEventListener('contextmenu', onMenu);
  if (!listening) {
    listening = true;
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }
}

// ------------------------------------------------------------ the list

function rebuildList() {
  list.innerHTML = '';
  const head = make('div', 'pose-list-head');
  head.append(make('span', '', 'Skeletons'),
    button('Add', addPerson, 'Add a skeleton'));
  list.appendChild(head);
  if (!state.people.length) {
    list.appendChild(make('div', 'pose-hint',
      'None yet: add one, or add the people of a photo.'));
  }
  rows = state.people.map((person, n) => {
    const row = make('div', 'pose-row');
    const name = make('span');
    const info = make('span', 'pose-hint');
    const size = make('input');
    size.type = 'range';
    size.min = 25;
    size.max = 200;
    size.value = 100;
    size.title = 'Size of this skeleton (goes back to 100% when released)';
    size.addEventListener('input', () => onPersonScale(n, size));
    size.addEventListener('change', () => {
      scaleStart = null;
      size.value = 100;
    });
    size.addEventListener('click', (event) => event.stopPropagation());
    const del = button('✕', () => deletePerson(n), 'Delete this skeleton');
    const label = make('div');
    label.append(name, ' ', info);
    row.append(label, del, size);
    row.addEventListener('click', () => {
      selected = n;
      draw();
    });
    list.appendChild(row);
    return { row, name, info };
  });
  rowsFor = state.people.length;
}

function updateList() {
  if (rowsFor !== state.people.length) rebuildList();
  state.people.forEach((person, n) => {
    const { row, name, info } = rows[n];
    name.textContent = `Skeleton ${n + 1}`;
    info.textContent = `${person.filter((p) => p).length}/${person.length}`
      + ' points';
    row.classList.toggle('selected', n === selected);
  });
}

function onPersonScale(n, slider) {
  // Always from the person as it was when the slider was grabbed, so
  // going back and forth doesn't drift
  if (!scaleStart || scaleStart.n !== n) {
    scaleStart = { n, person: JSON.parse(JSON.stringify(state.people[n])) };
  }
  state.people[n] = scalePerson(
    scaleStart.person, state.w, state.h, Number(slider.value) / 100);
  commit();
  draw();
}

function onSize() {
  const w = canvasSide(widthBox.value);
  const h = canvasSide(heightBox.value);
  if (w === null || h === null || (w === state.w && h === state.h)) return;
  state.people = resizeCanvas(state.people, state.w, state.h, w, h);
  if (state.box) {
    const [x0, y0] = movePoint([state.box[0], state.box[1]],
      state.w, state.h, w, h);
    const [x1, y1] = movePoint([state.box[2], state.box[3]],
      state.w, state.h, w, h);
    state.box = [x0, y0, x1, y1];
  }
  state.w = w;
  state.h = h;
  commit();
  draw();
}

function scale() {
  return Math.min(1, MAX_SIDE / Math.max(state.w, state.h));
}

// -------------------------------------------------------------- geometry

function ghost(person, i) {
  // Where a missing point would be: next to a point it is joined to.
  const sx = BODY_HEIGHT * state.h / state.w;
  for (const [a, b] of LIMBS) {
    const j = a === i ? b : (b === i ? a : -1);
    if (j >= 0 && person[j]) {
      return [
        person[j][0] + (TEMPLATE[i][0] - TEMPLATE[j][0]) * sx,
        person[j][1] + (TEMPLATE[i][1] - TEMPLATE[j][1]) * BODY_HEIGHT,
      ];
    }
  }
  return [0.5 + TEMPLATE[i][0] * sx, 0.1 + TEMPLATE[i][1] * BODY_HEIGHT];
}

const clamp = (v) => Math.min(1, Math.max(0, v));

function where(event) {
  const box = canvas.getBoundingClientRect();
  return [
    clamp((event.clientX - box.left) / box.width),
    clamp((event.clientY - box.top) / box.height),
    box.width,
  ];
}

function find(x, y, pixels) {
  // The closest point to (x, y) within reach: the selected person's
  // missing points count too. pixels is the width of the canvas on screen.
  const aspect = state.h / state.w;
  let best = null;
  let nearest = GRAB;
  state.people.forEach((person, n) => {
    person.forEach((point, i) => {
      const at = point || (n === selected ? ghost(person, i) : null);
      if (!at) return;
      const far = Math.hypot(
        (at[0] - x) * pixels, (at[1] - y) * pixels * aspect);
      // On a tie the selected person wins
      if (far < nearest || (far === nearest && n === selected)) {
        nearest = far;
        best = { n, i, ghost: !point };
      }
    });
  });
  return best;
}

// ----------------------------------------------------------------- input

function onDown(event) {
  if (event.button !== 0) return;
  const [x, y, pixels] = where(event);
  const hit = find(x, y, pixels);
  if (!hit) return;
  selected = hit.n;
  const person = state.people[hit.n];
  if (hit.ghost) person[hit.i] = ghost(person, hit.i);
  drag = {
    n: hit.n, i: hit.i, whole: event.shiftKey,
    last: [x, y],
  };
  event.preventDefault();
  draw();
}

function onMove(event) {
  if (!drag) return;
  const [x, y] = where(event);
  const person = state.people[drag.n];
  if (drag.whole) {
    const dx = x - drag.last[0];
    const dy = y - drag.last[1];
    for (let i = 0; i < person.length; i++) {
      if (person[i]) {
        person[i] = [clamp(person[i][0] + dx), clamp(person[i][1] + dy)];
      }
    }
    drag.last = [x, y];
  } else {
    person[drag.i] = [x, y];
  }
  draw();
}

function onUp() {
  if (!drag) return;
  drag = null;
  commit();
}

function onMenu(event) {
  event.preventDefault();
  const [x, y, pixels] = where(event);
  const hit = find(x, y, pixels);
  if (!hit || hit.ghost) return;
  selected = hit.n;
  state.people[hit.n][hit.i] = null;
  commit();
  draw();
}

function addPerson() {
  const sx = BODY_HEIGHT * state.h / state.w;
  const shift = (state.people.length % 3 - 1) * 0.25;
  state.people.push(TEMPLATE.map((t) => [
    clamp(0.5 + shift + t[0] * sx),
    clamp(0.1 + t[1] * BODY_HEIGHT),
  ]));
  selected = state.people.length - 1;
  commit();
  draw();
}

function deletePerson(n) {
  if (n < 0 || n >= state.people.length) return;
  state.people.splice(n, 1);
  if (n < selected) selected -= 1;
  selected = Math.min(selected, Math.max(0, state.people.length - 1));
  commit();
  draw();
}

// --------------------------------------------------------------- drawing

function loadPhoto(then) {
  if (state.photo === photoSource) return then();
  photoSource = state.photo;
  photo = null;
  if (!state.photo) return then();
  const image = new Image();
  image.onload = () => { photo = image; then(); };
  image.src = state.photo;
  then();
}

function draw() {
  build();
  const s = scale();
  canvas.width = Math.round(state.w * s);
  canvas.height = Math.round(state.h * s);
  canvas.style.width = '100%';
  canvas.style.maxWidth = canvas.width + 'px';
  canvas.style.height = 'auto';
  canvas.style.background = '#000';
  canvas.style.touchAction = 'none';
  const g = canvas.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, canvas.width, canvas.height);
  photoBox.disabled = !state.photo;
  if (photo && showPhoto) {
    const [x0, y0, x1, y1] = state.box || [0, 0, 1, 1];
    g.globalAlpha = 0.35;
    g.drawImage(photo, x0 * canvas.width, y0 * canvas.height,
      (x1 - x0) * canvas.width, (y1 - y0) * canvas.height);
    g.globalAlpha = 1;
  }
  if (Number(widthBox.value) !== state.w) widthBox.value = state.w;
  if (Number(heightBox.value) !== state.h) heightBox.value = state.h;
  updateList();
  warning.textContent = anyOutside(state.people)
    ? 'Some points are outside the canvas: they are moved to its edge when saved.'
    : '';
  const at = (p) => [p[0] * canvas.width, p[1] * canvas.height];
  const radius = Math.max(3, Math.min(canvas.width, canvas.height) / 100);

  state.people.forEach((person, n) => {
    g.globalAlpha = n === selected ? 1 : 0.7;
    LIMBS.forEach(([a, b], k) => {
      if (!person[a] || !person[b]) return;
      const [x1, y1] = at(person[a]);
      const [x2, y2] = at(person[b]);
      g.strokeStyle = COLORS[k];
      g.globalAlpha = (n === selected ? 1 : 0.7) * 0.6;
      g.lineWidth = radius * 2;
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
    });
    g.globalAlpha = n === selected ? 1 : 0.7;
    person.forEach((p, i) => {
      if (p) {
        const [x, y] = at(p);
        g.fillStyle = COLORS[i];
        g.beginPath();
        g.arc(x, y, radius, 0, 2 * Math.PI);
        g.fill();
      } else if (n === selected) {
        const [x, y] = at(ghost(person, i));
        g.globalAlpha = 0.35;
        g.strokeStyle = COLORS[i];
        g.lineWidth = 2;
        g.beginPath();
        g.arc(x, y, radius, 0, 2 * Math.PI);
        g.stroke();
        g.globalAlpha = 1;
      }
    });
    if (n === selected && person.some((p) => p)) {
      const first = person.find((p) => p);
      const [x, y] = at(first);
      g.strokeStyle = '#fff';
      g.lineWidth = 1;
      g.beginPath();
      g.arc(x, y, radius + 4, 0, 2 * Math.PI);
      g.stroke();
    }
  });
  g.globalAlpha = 1;
}

function refresh() {
  parse();
  loadPhoto(draw);
}

watch('value', refresh);
refresh();
