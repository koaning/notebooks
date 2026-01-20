function render({ model, el }) {
  const container = document.createElement("div");
  container.className = "circular-road-root";
  el.appendChild(container);

  const canvas = document.createElement("canvas");
  canvas.className = "circular-road-canvas";
  canvas.width = 500;
  canvas.height = 500;
  container.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  const roadRadius = 120;
  const stepSize = 1; // Move forward 1 pixel per step (Bret Victor's model)

  let path = [];
  let isDragging = false;

  // Load Bret Victor's car image (32x16 pixels)
  const carImage = new Image();
  carImage.src = "data:image/png;base64,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";
  let carImageLoaded = false;
  carImage.onload = () => {
    carImageLoaded = true;
    draw();
  };

  // Normalize angle difference to [-π, π]
  function angleDiff(a, b) {
    let diff = a - b;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    return diff;
  }

  // Compute the car's path using Bret Victor's model:
  // At each step:
  //   1. Move forward 1 pixel
  //   2. If left of road (inside circle), turn right by angle
  //   3. If right of road (outside circle), turn left by angle
  function computePath(steeringAngleDeg, roadWidth) {
    const steeringAngle = (steeringAngleDeg * Math.PI) / 180;
    const result = [];

    let carX = roadRadius;
    let carY = 0;
    let carAngle = Math.PI / 2; // Facing up (tangent to circle)
    let totalAngleTraveled = 0;

    // Continue until car completes 1 full lap
    const maxSteps = 10000; // Safety limit
    let steps = 0;

    while (totalAngleTraveled < 2 * Math.PI && steps < maxSteps) {
      result.push([carX, carY, carAngle]);
      steps++;

      const prevRoadAngle = Math.atan2(carY, carX);

      // Move car forward 1 pixel in current direction
      carX += Math.cos(carAngle) * stepSize;
      carY += Math.sin(carAngle) * stepSize;

      // Check if car is left or right of road
      const distFromCenter = Math.sqrt(carX * carX + carY * carY);
      const innerEdge = roadRadius - roadWidth / 2;
      const outerEdge = roadRadius + roadWidth / 2;

      if (distFromCenter < innerEdge) {
        // Left of road (inside circle) - turn right
        carAngle -= steeringAngle;
      } else if (distFromCenter > outerEdge) {
        // Right of road (outside circle) - turn left
        carAngle += steeringAngle;
      }

      // Track progress around the circle
      const newRoadAngle = Math.atan2(carY, carX);
      const angleDelta = angleDiff(newRoadAngle, prevRoadAngle);
      if (angleDelta > 0) {
        totalAngleTraveled += angleDelta;
      }
    }

    return result;
  }

  // Find closest point on path to given coordinates
  function findClosestPointIndex(x, y) {
    let minDist = Infinity;
    let minIndex = 0;

    for (let i = 0; i < path.length; i++) {
      const dx = path[i][0] - x;
      const dy = path[i][1] - y;
      const dist = dx * dx + dy * dy;
      if (dist < minDist) {
        minDist = dist;
        minIndex = i;
      }
    }

    return minIndex;
  }

  // Compute total path length
  function computePathLength(pathData) {
    let total = 0;
    for (let i = 1; i < pathData.length; i++) {
      const dx = pathData[i][0] - pathData[i - 1][0];
      const dy = pathData[i][1] - pathData[i - 1][1];
      total += Math.sqrt(dx * dx + dy * dy);
    }
    return total;
  }

  function draw() {
    const angle = model.get("angle");
    const roadWidth = model.get("road_width");
    const position = model.get("position");

    // Recompute path when angle or road_width changes
    path = computePath(angle, roadWidth);

    // Compute and sync total length
    const totalLength = computePathLength(path);
    if (model.get("total_length") !== totalLength) {
      model.set("total_length", totalLength);
      model.save_changes();
    }

    // Find path bounds for auto-scaling
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const [px, py] of path) {
      minX = Math.min(minX, px);
      maxX = Math.max(maxX, px);
      minY = Math.min(minY, py);
      maxY = Math.max(maxY, py);
    }
    // Include road bounds
    minX = Math.min(minX, -roadRadius - roadWidth);
    maxX = Math.max(maxX, roadRadius + roadWidth);
    minY = Math.min(minY, -roadRadius - roadWidth);
    maxY = Math.max(maxY, roadRadius + roadWidth);

    const pathWidth = maxX - minX;
    const pathHeight = maxY - minY;
    const padding = 30;
    const scale = Math.min(
      (canvas.width - 2 * padding) / pathWidth,
      (canvas.height - 2 * padding) / pathHeight
    );

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const offsetX = -(minX + maxX) / 2 * scale;
    const offsetY = (minY + maxY) / 2 * scale;

    // Convert path coordinates to canvas coordinates
    function pathToCanvas(pathX, pathY) {
      return [
        centerX + pathX * scale + offsetX,
        centerY - pathY * scale + offsetY
      ];
    }

    function canvasToPath(canvasX, canvasY) {
      return [
        (canvasX - centerX - offsetX) / scale,
        -(canvasY - centerY - offsetY) / scale
      ];
    }

    // Store for mouse events
    canvas._pathToCanvas = pathToCanvas;
    canvas._canvasToPath = canvasToPath;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw road (gray ring)
    ctx.beginPath();
    const [cx, cy] = pathToCanvas(0, 0);
    ctx.arc(cx, cy, roadRadius * scale, 0, 2 * Math.PI);
    ctx.strokeStyle = "#e5e5e5";
    ctx.lineWidth = roadWidth * scale;
    ctx.stroke();

    // Draw road edge lines
    ctx.beginPath();
    ctx.arc(cx, cy, (roadRadius - roadWidth / 2) * scale, 0, 2 * Math.PI);
    ctx.strokeStyle = "#999";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, (roadRadius + roadWidth / 2) * scale, 0, 2 * Math.PI);
    ctx.strokeStyle = "#999";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw path (the car's trajectory)
    if (path.length > 1) {
      ctx.beginPath();
      const [startX, startY] = pathToCanvas(path[0][0], path[0][1]);
      ctx.moveTo(startX, startY);

      for (let i = 1; i < path.length; i++) {
        const [px, py] = pathToCanvas(path[i][0], path[i][1]);
        ctx.lineTo(px, py);
      }

      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw car at current position
    if (path.length > 0) {
      const pathIndex = Math.floor(position * (path.length - 1));
      const clampedIndex = Math.max(0, Math.min(path.length - 1, pathIndex));
      const [carPathX, carPathY, carAngle] = path[clampedIndex];
      const [carCanvasX, carCanvasY] = pathToCanvas(carPathX, carPathY);

      // Draw car using Bret Victor's actual car.png image
      ctx.save();
      ctx.translate(carCanvasX, carCanvasY);
      ctx.rotate(-carAngle); // Car points in direction of travel

      // Car image is 32x16 pixels, scale it appropriately
      const carDrawWidth = 32 * scale * 0.8;
      const carDrawHeight = 16 * scale * 0.8;

      if (carImageLoaded) {
        ctx.drawImage(
          carImage,
          -carDrawWidth / 2,
          -carDrawHeight / 2,
          carDrawWidth,
          carDrawHeight
        );
      } else {
        // Fallback: draw a simple red rectangle while image loads
        ctx.fillStyle = "#e53935";
        ctx.fillRect(-carDrawWidth / 2, -carDrawHeight / 2, carDrawWidth, carDrawHeight);
      }

      ctx.restore();
    }
  }

  // Mouse event handlers for dragging
  canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    updatePositionFromMouse(e);
  });

  canvas.addEventListener("mousemove", (e) => {
    if (isDragging) {
      updatePositionFromMouse(e);
    }
  });

  document.addEventListener("mouseup", () => {
    isDragging = false;
  });

  function updatePositionFromMouse(e) {
    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    if (canvas._canvasToPath) {
      const [pathX, pathY] = canvas._canvasToPath(canvasX, canvasY);
      const closestIndex = findClosestPointIndex(pathX, pathY);
      const newPosition = path.length > 1 ? closestIndex / (path.length - 1) : 0;

      model.set("position", newPosition);
      model.save_changes();
    }
  }

  // Listen for model changes
  model.on("change:angle", draw);
  model.on("change:road_width", draw);
  model.on("change:position", draw);

  // Initial draw
  draw();

  return () => {
    document.removeEventListener("mouseup", () => {
      isDragging = false;
    });
  };
}

export default { render };
