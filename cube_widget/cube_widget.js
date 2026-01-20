// Simple SVG-based isometric cube widget
// No Three.js, no rotation - just clear clickable axes

// Axis colors - consistent RGB scheme
const AXIS_COLORS = {
  x: "#e74c3c",  // Red
  y: "#27ae60",  // Green
  z: "#3498db",  // Blue
};

const COLORS = {
  wireframe: "#95a5a6",
};

// Isometric projection helpers - more angle for better depth
const ISO_ANGLE = Math.PI / 5; // 36 degrees (was 30) - more depth visible
const SCALE = 110;

function isoProject(x, y, z) {
  // Convert 3D coordinates to 2D isometric projection
  const isoX = (x - z) * Math.cos(ISO_ANGLE);
  const isoY = (x + z) * Math.sin(ISO_ANGLE) - y;
  return { x: isoX * SCALE, y: isoY * SCALE };
}

function render({ model, el }) {
  el.classList.add("cube-widget-root");

  const container = document.createElement("div");
  container.className = "cube-container";
  el.appendChild(container);

  // SVG container
  const svgContainer = document.createElement("div");
  svgContainer.className = "svg-container";
  container.appendChild(svgContainer);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "-200 -180 400 360");
  svg.setAttribute("class", "cube-svg");
  svgContainer.appendChild(svg);

  // Cube vertices (unit cube centered at origin)
  const vertices = {
    v0: { x: -0.5, y: -0.5, z: -0.5 }, // front-bottom-left
    v1: { x: 0.5, y: -0.5, z: -0.5 },  // front-bottom-right
    v2: { x: 0.5, y: 0.5, z: -0.5 },   // front-top-right
    v3: { x: -0.5, y: 0.5, z: -0.5 },  // front-top-left
    v4: { x: -0.5, y: -0.5, z: 0.5 },  // back-bottom-left
    v5: { x: 0.5, y: -0.5, z: 0.5 },   // back-bottom-right
    v6: { x: 0.5, y: 0.5, z: 0.5 },    // back-top-right
    v7: { x: -0.5, y: 0.5, z: 0.5 },   // back-top-left
  };

  // Project all vertices
  function getProjectedVertices() {
    const projected = {};
    for (const [key, v] of Object.entries(vertices)) {
      projected[key] = isoProject(v.x, v.y, v.z);
    }
    return projected;
  }

  // Draw wireframe cube
  function drawWireframe(p) {
    const edges = [
      // Front face
      [p.v0, p.v1], [p.v1, p.v2], [p.v2, p.v3], [p.v3, p.v0],
      // Back face
      [p.v4, p.v5], [p.v5, p.v6], [p.v6, p.v7], [p.v7, p.v4],
      // Connecting edges
      [p.v0, p.v4], [p.v1, p.v5], [p.v2, p.v6], [p.v3, p.v7],
    ];

    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "wireframe");

    edges.forEach(([start, end]) => {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", start.x);
      line.setAttribute("y1", start.y);
      line.setAttribute("x2", end.x);
      line.setAttribute("y2", end.y);
      line.setAttribute("stroke", COLORS.wireframe);
      line.setAttribute("stroke-width", "2");
      group.appendChild(line);
    });

    return group;
  }

  // Draw axis with label
  function drawAxis(axis, p, config, lockIndex) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `axis axis-${axis}`);
    group.style.cursor = "pointer";

    let start, end, labelPos;
    let axisColor = "#e74c3c"; // X default red

    if (axis === "x") {
      start = p.v0;
      end = p.v1;
      const extended = isoProject(0.8, -0.5, -0.5);
      labelPos = extended;
      axisColor = "#e74c3c";
    } else if (axis === "y") {
      start = p.v0;
      end = p.v3;
      const extended = isoProject(-0.5, 0.8, -0.5);
      labelPos = extended;
      axisColor = "#27ae60";
    } else {
      start = p.v0;
      end = p.v4;
      const extended = isoProject(-0.5, -0.5, 0.8);
      labelPos = extended;
      axisColor = "#3498db";
    }

    // Determine badge based on lock order (color stays consistent with axis)
    let displayColor = axisColor;
    let badge = "";
    if (lockIndex === 0) {
      badge = "①";
    } else if (lockIndex === 1) {
      badge = "②";
    } else if (lockIndex === 2) {
      badge = "③";
    }

    // Draw axis line (thicker, more visible)
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", start.x);
    line.setAttribute("y1", start.y);
    line.setAttribute("x2", labelPos.x);
    line.setAttribute("y2", labelPos.y);
    line.setAttribute("stroke", displayColor);
    line.setAttribute("stroke-width", lockIndex >= 0 ? "4" : "3");
    group.appendChild(line);

    // Draw arrow head
    const arrowSize = 8;
    const angle = Math.atan2(labelPos.y - start.y, labelPos.x - start.x);
    const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    const ax = labelPos.x;
    const ay = labelPos.y;
    const points = [
      `${ax},${ay}`,
      `${ax - arrowSize * Math.cos(angle - 0.4)},${ay - arrowSize * Math.sin(angle - 0.4)}`,
      `${ax - arrowSize * Math.cos(angle + 0.4)},${ay - arrowSize * Math.sin(angle + 0.4)}`,
    ].join(" ");
    arrow.setAttribute("points", points);
    arrow.setAttribute("fill", displayColor);
    group.appendChild(arrow);

    // Label text (create first to measure)
    const labelText = badge ? `${badge} ${config.name}` : config.name;
    const textWidth = labelText.length * 8 + 16; // Approximate width
    const bgWidth = Math.max(50, textWidth);
    const bgHeight = 26;

    // Label background
    const labelBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    labelBg.setAttribute("x", labelPos.x - bgWidth / 2 + 12);
    labelBg.setAttribute("y", labelPos.y - bgHeight / 2);
    labelBg.setAttribute("width", bgWidth);
    labelBg.setAttribute("height", bgHeight);
    labelBg.setAttribute("rx", "4");
    labelBg.setAttribute("fill", lockIndex >= 0 ? displayColor : "white");
    labelBg.setAttribute("stroke", displayColor);
    labelBg.setAttribute("stroke-width", lockIndex >= 0 ? "0" : "1.5");
    labelBg.setAttribute("class", "axis-label-bg");
    group.appendChild(labelBg);

    // Label text
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", labelPos.x + 12);
    text.setAttribute("y", labelPos.y + 4);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "12");
    text.setAttribute("font-weight", "600");
    text.setAttribute("fill", lockIndex >= 0 ? "white" : displayColor);
    text.setAttribute("class", "axis-label-text");
    text.textContent = labelText;
    group.appendChild(text);

    // Click handler
    group.addEventListener("click", () => {
      const lockedOrder = model.get("locked_order") || [];
      if (lockedOrder.includes(axis)) {
        // Unlock: remove from locked_order
        model.set("locked_order", lockedOrder.filter((a) => a !== axis));
      } else {
        // Lock: add to locked_order AND set value to middle
        const axisConfig = model.get(`${axis}_axis`) || { values: [0, 0.5, 1] };
        const values = axisConfig.values || [0, 0.5, 1];
        const middleIndex = Math.floor(values.length / 2);
        const middleValue = values[middleIndex];

        const currentAxisValues = model.get("axis_values") || {};
        model.set("axis_values", { ...currentAxisValues, [axis]: middleValue });
        model.set("locked_order", [...lockedOrder, axis]);
      }
      model.save_changes();
    });

    // Hover effects
    group.addEventListener("mouseenter", () => {
      labelBg.setAttribute("transform", "scale(1.1)");
      labelBg.style.transformOrigin = `${labelPos.x + 15}px ${labelPos.y}px`;
    });
    group.addEventListener("mouseleave", () => {
      labelBg.removeAttribute("transform");
    });

    return group;
  }

  // Draw selection plane - uses axis color
  function drawPlane(axis, normalizedValue, p) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "selection-plane");

    const v = normalizedValue; // 0 to 1
    const color = AXIS_COLORS[axis];

    let corners;
    if (axis === "x") {
      const x = -0.5 + v;
      corners = [
        isoProject(x, -0.5, -0.5),
        isoProject(x, 0.5, -0.5),
        isoProject(x, 0.5, 0.5),
        isoProject(x, -0.5, 0.5),
      ];
    } else if (axis === "y") {
      const y = -0.5 + v;
      corners = [
        isoProject(-0.5, y, -0.5),
        isoProject(0.5, y, -0.5),
        isoProject(0.5, y, 0.5),
        isoProject(-0.5, y, 0.5),
      ];
    } else {
      const z = -0.5 + v;
      corners = [
        isoProject(-0.5, -0.5, z),
        isoProject(0.5, -0.5, z),
        isoProject(0.5, 0.5, z),
        isoProject(-0.5, 0.5, z),
      ];
    }

    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", corners.map((c) => `${c.x},${c.y}`).join(" "));
    polygon.setAttribute("fill", color);
    polygon.setAttribute("fill-opacity", "0.4");
    polygon.setAttribute("stroke", color);
    polygon.setAttribute("stroke-width", "2");
    group.appendChild(polygon);

    return group;
  }

  // Draw selection line - uses second locked axis color
  function drawLine(axis1, value1, axis2, value2) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "selection-line");

    const freeAxis = ["x", "y", "z"].find((a) => a !== axis1 && a !== axis2);
    const color = AXIS_COLORS[axis2]; // Line color matches the 2nd locked axis

    // Calculate the line endpoints
    const pos = { x: 0, y: 0, z: 0 };
    pos[axis1] = -0.5 + value1;
    pos[axis2] = -0.5 + value2;

    let start, end;
    if (freeAxis === "x") {
      start = isoProject(-0.5, pos.y, pos.z);
      end = isoProject(0.5, pos.y, pos.z);
    } else if (freeAxis === "y") {
      start = isoProject(pos.x, -0.5, pos.z);
      end = isoProject(pos.x, 0.5, pos.z);
    } else {
      start = isoProject(pos.x, pos.y, -0.5);
      end = isoProject(pos.x, pos.y, 0.5);
    }

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", start.x);
    line.setAttribute("y1", start.y);
    line.setAttribute("x2", end.x);
    line.setAttribute("y2", end.y);
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "4");
    line.setAttribute("stroke-linecap", "round");
    group.appendChild(line);

    return group;
  }

  // Draw selection point - uses third locked axis color
  function drawPoint(x, y, z, thirdAxis) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "selection-point");

    const pos = isoProject(-0.5 + x, -0.5 + y, -0.5 + z);
    const color = AXIS_COLORS[thirdAxis];

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", pos.x);
    circle.setAttribute("cy", pos.y);
    circle.setAttribute("r", "10");
    circle.setAttribute("fill", color);
    circle.setAttribute("stroke", "white");
    circle.setAttribute("stroke-width", "2");
    group.appendChild(circle);

    return group;
  }

  // Control panel
  const controlPanel = document.createElement("div");
  controlPanel.className = "cube-controls";
  container.appendChild(controlPanel);

  const stateDisplay = document.createElement("div");
  stateDisplay.className = "state-display";
  controlPanel.appendChild(stateDisplay);

  const slidersContainer = document.createElement("div");
  slidersContainer.className = "sliders-container";
  controlPanel.appendChild(slidersContainer);

  const resetButton = document.createElement("button");
  resetButton.className = "reset-button";
  resetButton.textContent = "Reset";
  resetButton.addEventListener("click", () => {
    model.set("locked_order", []);
    model.save_changes();
  });
  controlPanel.appendChild(resetButton);

  // Helper functions
  function getAxisConfig(axis) {
    return model.get(`${axis}_axis`) || { name: axis.toUpperCase(), values: [0, 0.5, 1] };
  }

  function normalizeValue(axis, value) {
    const config = getAxisConfig(axis);
    const values = config.values || [0, 1];
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (max === min) return 0.5;
    return (value - min) / (max - min);
  }

  // Update display - minimal
  function updateStateDisplay() {
    const lockedOrder = model.get("locked_order") || [];
    if (lockedOrder.length === 0) {
      stateDisplay.textContent = "Click an axis to begin";
      stateDisplay.style.display = "block";
    } else {
      stateDisplay.style.display = "none";
    }
  }

  // Update sliders - smooth range inputs
  function updateSliders() {
    const lockedOrder = model.get("locked_order") || [];
    const axisValues = model.get("axis_values") || {};

    // Only rebuild if locked axes changed
    const currentAxes = Array.from(slidersContainer.querySelectorAll("[data-axis]")).map(el => el.dataset.axis);
    const needsRebuild = lockedOrder.length !== currentAxes.length ||
                         !lockedOrder.every((a, i) => a === currentAxes[i]);

    if (needsRebuild) {
      slidersContainer.innerHTML = "";

      lockedOrder.forEach((axis, index) => {
        const config = getAxisConfig(axis);
        const values = config.values || [0, 1];
        const min = Math.min(...values);
        const max = Math.max(...values);

        const row = document.createElement("div");
        row.className = "slider-row";
        row.dataset.axis = axis;

        const label = document.createElement("span");
        label.className = "slider-label";
        label.style.color = AXIS_COLORS[axis];
        label.textContent = config.name;
        row.appendChild(label);

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = min;
        slider.max = max;
        slider.step = (max - min) / 100; // Smooth steps
        slider.value = axisValues[axis] ?? (min + max) / 2;
        slider.className = "axis-slider";
        slider.style.accentColor = AXIS_COLORS[axis];
        row.appendChild(slider);

        const valueSpan = document.createElement("span");
        valueSpan.className = "slider-value";
        valueSpan.textContent = (axisValues[axis] ?? (min + max) / 2).toFixed(1);
        row.appendChild(valueSpan);

        slider.addEventListener("input", () => {
          const newValue = parseFloat(slider.value);
          valueSpan.textContent = newValue.toFixed(1);
          const newAxisValues = { ...model.get("axis_values"), [axis]: newValue };
          model.set("axis_values", newAxisValues);
          model.save_changes();
        });

        slidersContainer.appendChild(row);
      });
    } else {
      // Just update values without rebuilding
      lockedOrder.forEach((axis) => {
        const row = slidersContainer.querySelector(`[data-axis="${axis}"]`);
        if (row) {
          const slider = row.querySelector("input");
          const valueSpan = row.querySelector(".slider-value");
          const currentValue = axisValues[axis];
          if (slider && currentValue !== undefined && parseFloat(slider.value) !== currentValue) {
            slider.value = currentValue;
            valueSpan.textContent = currentValue.toFixed(1);
          }
        }
      });
    }
  }

  // Main render function
  function renderSVG() {
    // Clear SVG
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }

    const p = getProjectedVertices();
    const lockedOrder = model.get("locked_order") || [];
    const axisValues = model.get("axis_values") || {};

    // Draw wireframe first (back)
    svg.appendChild(drawWireframe(p));

    // Draw selection geometry
    if (lockedOrder.length >= 1) {
      const axis1 = lockedOrder[0];
      const value1 = normalizeValue(axis1, axisValues[axis1] ?? 0.5);
      svg.appendChild(drawPlane(axis1, value1, p));
    }

    if (lockedOrder.length >= 2) {
      const axis1 = lockedOrder[0];
      const axis2 = lockedOrder[1];
      const value1 = normalizeValue(axis1, axisValues[axis1] ?? 0.5);
      const value2 = normalizeValue(axis2, axisValues[axis2] ?? 0.5);
      svg.appendChild(drawLine(axis1, value1, axis2, value2));
    }

    if (lockedOrder.length >= 3) {
      const x = normalizeValue("x", axisValues.x ?? 0.5);
      const y = normalizeValue("y", axisValues.y ?? 0.5);
      const z = normalizeValue("z", axisValues.z ?? 0.5);
      const thirdAxis = lockedOrder[2];
      svg.appendChild(drawPoint(x, y, z, thirdAxis));
    }

    // Draw axes on top
    ["x", "y", "z"].forEach((axis) => {
      const config = getAxisConfig(axis);
      const lockIndex = lockedOrder.indexOf(axis);
      svg.appendChild(drawAxis(axis, p, config, lockIndex));
    });

    updateStateDisplay();
    updateSliders();
  }

  // Initial render
  renderSVG();

  // Listen for model changes
  model.on("change:locked_order", renderSVG);
  model.on("change:axis_values", renderSVG);
  model.on("change:x_axis", renderSVG);
  model.on("change:y_axis", renderSVG);
  model.on("change:z_axis", renderSVG);

  return () => {
    // Cleanup
  };
}

export default { render };
