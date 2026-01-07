export async function render({ model, el }) {
  const { default: Plotly } = await import("https://esm.sh/plotly.js-dist-min@2.28.0");

  const container = document.createElement("div");
  container.className = "slice-plotter-container";

  // Controls Area
  const controls = document.createElement("div");
  controls.className = "controls";
  
  const axisLabel = document.createElement("label");
  axisLabel.textContent = "Slice Axis: ";
  const axisSelect = document.createElement("select");
  ["x", "y"].forEach(axis => {
      const opt = document.createElement("option");
      opt.value = axis;
      opt.text = axis.toUpperCase();
      axisSelect.appendChild(opt);
  });
  axisSelect.value = model.get("slice_axis");
  
  const sliderLabel = document.createElement("label");
  sliderLabel.textContent = " Index: ";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.style.flex = "1";
  
  const valDisplay = document.createElement("span");
  valDisplay.style.minWidth = "30px";
  valDisplay.style.textAlign = "right";
  
  controls.appendChild(axisLabel);
  controls.appendChild(axisSelect);
  controls.appendChild(sliderLabel);
  controls.appendChild(slider);
  controls.appendChild(valDisplay);

  // Plots Area
  const plotsContainer = document.createElement("div");
  plotsContainer.className = "plots-container";

  const div3d = document.createElement("div");
  div3d.className = "plot-3d";
  const div2d = document.createElement("div");
  div2d.className = "plot-2d";

  plotsContainer.appendChild(div3d);
  plotsContainer.appendChild(div2d);

  container.appendChild(controls);
  container.appendChild(plotsContainer);
  el.appendChild(container);

  function getData() {
    return {
      x: model.get("x"),
      y: model.get("y"),
      z: model.get("z"),
      axis: model.get("slice_axis"),
      idx: model.get("slice_index")
    };
  }

  function draw() {
    const { x, y, z, axis, idx } = getData();
    if (!z || !Array.isArray(z) || z.length === 0) {
        div3d.innerHTML = "No data provided";
        return;
    }

    // Dimensions
    // z[row][col] -> y[row], x[col]
    const numRows = z.length;     // Y size
    const numCols = z[0].length;  // X size

    // Determine max index for current axis
    const maxIdx = (axis === "y") ? numRows - 1 : numCols - 1;
    slider.max = String(maxIdx);
    
    // Validate current index
    let currentIdx = idx;
    if (currentIdx > maxIdx) currentIdx = maxIdx;
    if (currentIdx < 0) currentIdx = 0;
    
    // Update UI if model was out of sync (don't save here to avoid loop, just display)
    slider.value = String(currentIdx);
    valDisplay.textContent = String(currentIdx);

    // --- 3D Plot ---
    // Create the surface
    const surfaceTrace = {
        type: 'surface',
        z: z,
        x: x && x.length ? x : undefined,
        y: y && y.length ? y : undefined,
        colorscale: 'Viridis',
        showscale: false,
        opacity: 0.8
    };

    // Create a visual indicator of the slice plane/line on the 3D plot
    // This is a bit tricky to do perfectly as a plane without clutter, 
    // but we can draw a 3D line on the surface where the slice is.
    let sliceLineX, sliceLineY, sliceLineZ;
    
    if (axis === "y") {
        // Fixed Y row (index currentIdx)
        // Y coords are constant for this line
        const yVal = (y && y.length) ? y[currentIdx] : currentIdx;
        sliceLineY = Array(numCols).fill(yVal);
        sliceLineX = (x && x.length) ? x : Array.from({length: numCols}, (_, i) => i);
        sliceLineZ = z[currentIdx];
    } else {
        // Fixed X col (index currentIdx)
        // X coords are constant
        const xVal = (x && x.length) ? x[currentIdx] : currentIdx;
        sliceLineX = Array(numRows).fill(xVal);
        sliceLineY = (y && y.length) ? y : Array.from({length: numRows}, (_, i) => i);
        sliceLineZ = z.map(row => row[currentIdx]);
    }

    const sliceTrace3d = {
        type: 'scatter3d',
        mode: 'lines',
        x: sliceLineX,
        y: sliceLineY,
        z: sliceLineZ,
        line: {
            color: 'red',
            width: 5
        },
        name: 'Slice'
    };

    const layout3d = {
        title: '3D Surface',
        margin: { t: 30, b: 0, l: 0, r: 0 },
        autosize: true,
        scene: {
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' },
            zaxis: { title: 'Z' }
        }
    };

    Plotly.react(div3d, [surfaceTrace, sliceTrace3d], layout3d);


    // --- 2D Plot ---
    const trace2d = {
        type: 'scatter',
        mode: 'lines',
        x: (axis === 'y') ? sliceLineX : sliceLineY,
        y: sliceLineZ,
        line: { color: 'red' }
    };

    const layout2d = {
        margin: { t: 30, b: 40, l: 50, r: 20 },
        title: `2D Slice at ${axis.toUpperCase()} index ${currentIdx}`,
        xaxis: { title: axis === 'y' ? 'X' : 'Y' },
        yaxis: { title: 'Z' },
        autosize: true
    };

    Plotly.react(div2d, [trace2d], layout2d);
  }

  // Initial draw
  draw();

  // Model Listeners
  model.on("change:x", draw);
  model.on("change:y", draw);
  model.on("change:z", draw);
  model.on("change:slice_axis", draw);
  model.on("change:slice_index", draw);

  // UI Listeners
  slider.addEventListener("input", () => {
      const val = parseInt(slider.value);
      valDisplay.textContent = val;
      // We can optimistically draw here for performance, then save
      // But we need to update the model so the backend knows (if needed)
      // For pure client-side viz, we can just trigger draw logic, but 
      // let's stick to model binding.
      model.set("slice_index", val);
      model.save_changes();
  });

  axisSelect.addEventListener("change", () => {
      model.set("slice_axis", axisSelect.value);
      model.set("slice_index", 0); 
      model.save_changes();
  });
  
  // Handle resize
  const resizeObserver = new ResizeObserver(() => {
      Plotly.Plots.resize(div3d);
      Plotly.Plots.resize(div2d);
  });
  resizeObserver.observe(container);
}
