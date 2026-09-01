// Rule-authoring widget: place similarity copies of a parent image.
// A placement is [x, y, scale, angle_deg]: the base box (world-width BASE),
// scaled, rotated clockwise, centered at (x, y) in the unit-square world.
// `parent` is one placement (the base frame); `children` is a list of them.

const BASE = 0.34; // parent base-box world-width (fraction of the stage)

function render({ model, el }) {
  el.classList.add("rimg-root");

  const container = document.createElement("div");
  container.className = "rimg-container";

  const toolbar = document.createElement("div");
  toolbar.className = "rimg-toolbar";
  const addBtn = button("+ child", "add a child copy");
  const removeBtn = button("remove", "remove selected child");
  const clearBtn = button("clear", "remove all children");
  toolbar.append(addBtn, removeBtn, clearBtn);

  const stage = document.createElement("div");
  stage.className = "rimg-stage";
  container.append(toolbar, stage);
  el.append(container);

  function applySize() {
    const s = model.get("size") || 320;
    container.style.width = s + "px";
  }
  applySize();
  model.on("change:size", applySize);

  let parent = readList(model.get("parent"), [0.5, 0.5, 1.0, 0.0]);
  let children = readChildren();
  let sel = { kind: "parent" }; // {kind:'parent'} | {kind:'child', i} | null
  let aspect = 1;
  let active = null; // in-progress gesture

  // --- geometry ---
  function boxSize(scale) {
    return { w: BASE * scale, h: (BASE * scale) / aspect };
  }
  function placementOf(s) {
    return s && s.kind === "parent" ? parent : children[s.i];
  }

  function positionEl(node, p) {
    const [x, y, scale, angle] = p;
    const { w, h } = boxSize(scale);
    node.style.width = w * 100 + "%";
    node.style.height = h * 100 + "%";
    node.style.left = (x - w / 2) * 100 + "%";
    node.style.top = (y - h / 2) * 100 + "%";
    node.style.transform = `rotate(${angle}deg)`;
  }

  function makeEl(kind, i, p) {
    const isSel =
      sel && sel.kind === kind && (kind === "parent" || sel.i === i);
    const node = document.createElement("div");
    node.className =
      "rimg-el rimg-" + kind + (isSel ? " selected" : "");
    node.dataset.kind = kind;
    if (kind === "child") node.dataset.i = i;

    const img = document.createElement("img");
    img.className = "rimg-img";
    img.src = model.get("parent_image") || "";
    img.draggable = false;
    node.appendChild(img);

    if (isSel) {
      const rot = document.createElement("div");
      rot.className = "rimg-handle rimg-rotate";
      rot.dataset.role = "rotate";
      const sc = document.createElement("div");
      sc.className = "rimg-handle rimg-scale";
      sc.dataset.role = "scale";
      node.append(rot, sc);
    }
    positionEl(node, p);
    return node;
  }

  function draw() {
    stage.innerHTML = "";
    stage.appendChild(makeEl("parent", -1, parent));
    children.forEach((c, i) => stage.appendChild(makeEl("child", i, c)));
  }

  function nodeFor(s) {
    if (!s) return null;
    return s.kind === "parent"
      ? stage.querySelector(".rimg-parent")
      : stage.querySelector(`.rimg-child[data-i="${s.i}"]`);
  }

  // --- pointer gestures ---
  function worldOf(ev) {
    const r = stage.getBoundingClientRect();
    return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
  }

  stage.addEventListener("pointerdown", (ev) => {
    const elNode = ev.target.closest(".rimg-el");
    if (!elNode) {
      // click empty space -> deselect
      if (sel) {
        sel = null;
        draw();
      }
      return;
    }
    ev.preventDefault();
    const kind = elNode.dataset.kind;
    const i = kind === "child" ? parseInt(elNode.dataset.i) : -1;
    const role = ev.target.dataset.role || "pan";

    const newSel = kind === "parent" ? { kind } : { kind, i };
    const changedSel =
      !sel || sel.kind !== newSel.kind || sel.i !== newSel.i;
    sel = newSel;
    if (changedSel) draw();

    const p = placementOf(sel);
    const [wx, wy] = worldOf(ev);
    active = { role, sel, startPointer: [wx, wy], startPlacement: p.slice() };
    stage.setPointerCapture(ev.pointerId);
  });

  stage.addEventListener("pointermove", (ev) => {
    if (!active) return;
    const [wx, wy] = worldOf(ev);
    const p = placementOf(active.sel);
    const [sx, sy, sScale, sAngle] = active.startPlacement;

    if (active.role === "pan") {
      p[0] = clamp01(sx + (wx - active.startPointer[0]));
      p[1] = clamp01(sy + (wy - active.startPointer[1]));
    } else if (active.role === "scale") {
      const d0 = dist(active.startPointer, [sx, sy]) || 1e-6;
      const d1 = dist([wx, wy], [sx, sy]);
      p[2] = clampRange(sScale * (d1 / d0), 0.02, 3);
    } else if (active.role === "rotate") {
      const a = Math.atan2(wy - sy, wx - sx) * (180 / Math.PI) + 90;
      p[3] = a;
    }
    positionEl(nodeFor(active.sel), p);
  });

  stage.addEventListener("pointerup", () => {
    if (!active) return;
    active = null;
    commit();
  });

  // --- buttons ---
  addBtn.addEventListener("click", () => {
    const [px, py] = parent;
    children.push([clamp01(px), clamp01(py + 0.28), 0.55, 0]);
    sel = { kind: "child", i: children.length - 1 };
    commit();
    draw();
  });
  removeBtn.addEventListener("click", () => {
    if (!sel || sel.kind !== "child") return;
    children.splice(sel.i, 1);
    sel = children.length ? { kind: "child", i: Math.min(sel.i, children.length - 1) } : null;
    commit();
    draw();
  });
  clearBtn.addEventListener("click", () => {
    children = [];
    sel = { kind: "parent" };
    commit();
    draw();
  });

  // --- model sync ---
  function commit() {
    model.set("parent", parent.slice());
    model.set("children", children.map((c) => c.slice()));
    model.save_changes();
  }
  function readChildren() {
    const c = model.get("children");
    return Array.isArray(c) ? c.map((r) => r.slice()) : [];
  }
  model.on("change:children", () => {
    children = readChildren();
    if (sel && sel.kind === "child" && sel.i >= children.length)
      sel = children.length ? { kind: "child", i: children.length - 1 } : null;
    draw();
  });
  model.on("change:parent", () => {
    parent = readList(model.get("parent"), [0.5, 0.5, 1.0, 0.0]);
    draw();
  });
  model.on("change:parent_image", () => {
    // refresh aspect once the new image reports its natural size
    const probe = new Image();
    probe.onload = () => {
      if (probe.naturalHeight) aspect = probe.naturalWidth / probe.naturalHeight;
      draw();
    };
    probe.src = model.get("parent_image") || "";
    draw();
  });

  const ro = new ResizeObserver(() => draw());
  ro.observe(stage);

  // initial aspect probe
  const first = new Image();
  first.onload = () => {
    if (first.naturalHeight) aspect = first.naturalWidth / first.naturalHeight;
    draw();
  };
  first.src = model.get("parent_image") || "";
  draw();

  return () => ro.disconnect();
}

function button(label, title) {
  const b = document.createElement("button");
  b.className = "rimg-btn";
  b.textContent = label;
  if (title) b.title = title;
  return b;
}
function readList(v, fallback) {
  return Array.isArray(v) && v.length === 4 ? v.slice() : fallback.slice();
}
function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}
function clampRange(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export default { render };
