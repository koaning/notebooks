import * as d3 from "https://esm.sh/d3@7";

const NAMES = {
  3: ["Rock", "Paper", "Scissors"],
  5: ["Rock", "Paper", "Scissors", "Lizard", "Spock"],
};

function getNames(n) {
  if (NAMES[n]) return NAMES[n];
  return Array.from({ length: n }, (_, i) => String.fromCharCode(65 + i));
}

function buildTournament(n) {
  const k = Math.floor((n - 1) / 2);
  const isBalanced = n % 2 === 1;
  const edges = [];

  for (let i = 0; i < n; i++) {
    for (let j = 1; j <= k; j++) {
      edges.push({ source: i, target: (i + j) % n });
    }
  }

  // For even n, add one extra edge per pair of diametrically opposite nodes
  if (!isBalanced) {
    for (let i = 0; i < n / 2; i++) {
      const opposite = i + n / 2;
      // Alternate direction to spread imbalance
      if (i % 2 === 0) {
        edges.push({ source: i, target: opposite });
      } else {
        edges.push({ source: opposite, target: i });
      }
    }
  }

  // Compute out-degrees
  const outDegree = new Array(n).fill(0);
  for (const e of edges) {
    outDegree[e.source]++;
  }

  return { edges, outDegree, isBalanced, k };
}

function render({ model, el }) {
  const container = document.createElement("div");
  container.className = "rpsls-container";
  el.appendChild(container);

  const width = 500;
  const height = 500;
  const centerX = width / 2;
  const centerY = height / 2 - 10;
  const radius = 180;

  const svg = d3
    .select(container)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`);

  const defs = svg.append("defs");

  const edgeGroup = svg.append("g");
  const nodeGroup = svg.append("g");
  const labelGroup = svg.append("g");
  const balanceText = svg
    .append("text")
    .attr("class", "balance-label")
    .attr("x", centerX)
    .attr("y", height - 15);

  function updateMarkers(nodeRadius, markerSize) {
    defs.selectAll("marker").remove();

    const refX = nodeRadius / 1.4 + 10;

    const markerConfigs = [
      { id: "arrowhead", fill: "#666" },
      { id: "arrowhead-red", fill: "#e74c3c" },
      { id: "arrowhead-beats", fill: "#27ae60" },
      { id: "arrowhead-loses", fill: "#e67e22" },
    ];

    for (const { id, fill } of markerConfigs) {
      defs
        .append("marker")
        .attr("id", id)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", refX)
        .attr("refY", 0)
        .attr("markerWidth", markerSize)
        .attr("markerHeight", markerSize)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-4L10,0L0,4")
        .attr("fill", fill);
    }
  }

  function update() {
    const n = model.get("n");
    const names = getNames(n);
    const { edges, outDegree, isBalanced, k } = buildTournament(n);

    // Scale sizes down as n grows
    const nodeRadius = d3.scaleLinear().domain([3, 9]).range([22, 12]).clamp(true)(n);
    const markerSize = d3.scaleLinear().domain([3, 9]).range([8, 5]).clamp(true)(n);
    const strokeWidth = d3.scaleLinear().domain([3, 9]).range([2, 1]).clamp(true)(n);
    const labelOffset = nodeRadius + 12;
    const fontSize = d3.scaleLinear().domain([3, 9]).range([13, 10]).clamp(true)(n);

    updateMarkers(nodeRadius, markerSize);

    // Node positions on a circle (start from top)
    const nodes = names.map((name, i) => {
      const angle = (2 * Math.PI * i) / n - Math.PI / 2;
      return {
        name,
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle),
        outDegree: outDegree[i],
      };
    });

    // Color scale based on balance
    const balancedColor = "#3498db";
    const colorScale = d3
      .scaleSequential(d3.interpolateRdYlGn)
      .domain([Math.min(...outDegree) - 0.5, Math.max(...outDegree) + 0.5]);

    // Edges
    const balancedEdgeCount = n * k;
    const edgeDataMarked = edges.map((e, i) => ({
      source: e.source,
      target: e.target,
      x1: nodes[e.source].x,
      y1: nodes[e.source].y,
      x2: nodes[e.target].x,
      y2: nodes[e.target].y,
      isExtra: i >= balancedEdgeCount,
    }));

    const lines = edgeGroup.selectAll("line").data(edgeDataMarked);
    lines.exit().remove();
    const linesEnter = lines.enter().append("line");
    const allLines = linesEnter
      .merge(lines)
      .attr("class", "edge")
      .attr("x1", (d) => d.x1)
      .attr("y1", (d) => d.y1)
      .attr("x2", (d) => d.x2)
      .attr("y2", (d) => d.y2)
      .attr("stroke", (d) => (d.isExtra ? "#e74c3c" : "#888"))
      .attr("stroke-width", strokeWidth)
      .attr("stroke-dasharray", (d) => (d.isExtra ? "5,4" : "none"))
      .attr("marker-end", (d) =>
        d.isExtra ? "url(#arrowhead-red)" : "url(#arrowhead)"
      );

    // Hover helpers
    function highlightNode(idx) {
      // Dim all edges, then color outgoing green and incoming orange
      allLines
        .attr("stroke", (d) => {
          if (d.source === idx) return "#27ae60";
          if (d.target === idx) return "#e67e22";
          return "#e0e0e0";
        })
        .attr("stroke-width", (d) =>
          d.source === idx || d.target === idx ? strokeWidth * 2 : strokeWidth * 0.5
        )
        .attr("stroke-dasharray", (d) =>
          d.source === idx || d.target === idx ? "none" : (d.isExtra ? "5,4" : "none")
        )
        .attr("marker-end", (d) => {
          if (d.source === idx) return "url(#arrowhead-beats)";
          if (d.target === idx) return "url(#arrowhead-loses)";
          return "url(#arrowhead)";
        });

      // Raise connected edges above dimmed ones in the DOM
      allLines.filter((d) => d.source === idx || d.target === idx).raise();

      // Dim unrelated nodes
      allCircles.attr("opacity", (d, i) => {
        const connected = edgeDataMarked.some(
          (e) => (e.source === idx && e.target === i) || (e.target === idx && e.source === i)
        );
        return i === idx || connected ? 1 : 0.3;
      });
      allLabels.attr("opacity", (d, i) => {
        const connected = edgeDataMarked.some(
          (e) => (e.source === idx && e.target === i) || (e.target === idx && e.source === i)
        );
        return i === idx || connected ? 1 : 0.3;
      });
    }

    function resetHighlight() {
      allLines
        .attr("stroke", (d) => (d.isExtra ? "#e74c3c" : "#888"))
        .attr("stroke-width", strokeWidth)
        .attr("stroke-dasharray", (d) => (d.isExtra ? "5,4" : "none"))
        .attr("marker-end", (d) =>
          d.isExtra ? "url(#arrowhead-red)" : "url(#arrowhead)"
        );
      allCircles.attr("opacity", 1);
      allLabels.attr("opacity", 1);
    }

    // Nodes
    const circles = nodeGroup.selectAll("circle").data(nodes);
    circles.exit().remove();
    const circlesEnter = circles.enter().append("circle");
    const allCircles = circlesEnter
      .merge(circles)
      .attr("class", "node-circle")
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y)
      .attr("r", nodeRadius)
      .attr("fill", (d) =>
        isBalanced ? balancedColor : colorScale(d.outDegree)
      )
      .style("cursor", "pointer")
      .on("mouseenter", (event, d) => {
        const idx = nodes.indexOf(d);
        highlightNode(idx);
      })
      .on("mouseleave", () => resetHighlight());

    // Labels
    const labels = labelGroup.selectAll("text").data(nodes);
    labels.exit().remove();
    const labelsEnter = labels.enter().append("text");
    const allLabels = labelsEnter
      .merge(labels)
      .attr("class", "node-label")
      .attr("x", (d) => {
        const dx = d.x - centerX;
        return d.x + (dx / radius) * labelOffset;
      })
      .attr("y", (d) => {
        const dy = d.y - centerY;
        return d.y + (dy / radius) * labelOffset + 5;
      })
      .attr("text-anchor", "middle")
      .attr("font-size", fontSize)
      .text((d) => d.name);

    // Balance text
    if (isBalanced) {
      balanceText
        .text(`Balanced: each element beats exactly ${k} others`)
        .attr("fill", "#27ae60");
    } else {
      const degrees = [...new Set(outDegree)].sort().join(" or ");
      balanceText
        .text(`Not balanced: elements beat ${degrees} others (dashed = extra edges)`)
        .attr("fill", "#e74c3c");
    }
  }

  update();
  model.on("change:n", update);
}

export default { render };
