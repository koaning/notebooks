# Manim Video Guidelines

## Setup

### LaTeX Path
On macOS with MacTeX, add LaTeX to PATH before importing manim:
```python
import os
os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
```

### Dependencies
```toml
requires-python = ">=3.12,<3.14"
dependencies = [
    "manim>=0.18.0",
    "av>=14.0.0",  # For ffmpeg 8+ compatibility
]
```

## Common Pitfalls

### Text Overlap with Axes
- **Problem**: Annotations (like "R = 49") can overlap with axis labels (like "R (red balls)")
- **Solution**: Place annotations with explicit positioning that accounts for axis labels
  ```python
  # Bad: Places label directly below, may overlap with axis label
  label.next_to(axes.c2p(x, 0), DOWN, buff=0.3)

  # Better: Place to the side or use larger buff
  label.next_to(axes.c2p(x, 0), DOWN + LEFT, buff=0.5)
  # Or place above the point instead
  label.next_to(axes.c2p(x, y), UP, buff=0.2)
  ```

### Chart Positioning
- **Problem**: Charts with `to_edge(DOWN)` can push axis labels off screen
- **Solution**: Center the chart and shift slightly
  ```python
  # Bad: May push labels off bottom
  axes.to_edge(DOWN, buff=0.5)

  # Better: Center and shift
  axes.move_to(ORIGIN).shift(DOWN * 0.5)
  ```

### Axis Labels
- Use `buff` parameter to control spacing from axis
- For y-axis labels with fractions, may need extra space: `buff=0.4` or more

## Typography

### Use LaTeX for Math
- Use `MathTex` for equations: `MathTex(r"\frac{R}{R+1}")`
- Use `Tex` for text with math: `Tex("$R$ (red balls)")`
- Escape percent signs: `Tex("98\\%")`

### Aligned Equations
Use `&=` for aligned equals signs:
```python
MathTex(
    r"\frac{R}{R + 1} &= 0.98 \\",
    r"R &= 0.98 \cdot (R + 1) \\",
    r"R &= 49"
)
```

### Font Sizes
Consistent sizing hierarchy:
- Title: 48-56
- Main formula: 72
- Labels/annotations: 28-36
- Axis labels: 24-32

## Animation Style

### Minimal Text
- Video should support voiceover, not replace it
- Show formulas and charts, not explanatory sentences
- Use color to highlight, not text

### Pacing
- `self.wait(1)` for pauses between ideas
- `run_time=2` for important animations
- `run_time=0.3-0.5` for quick transitions

## Colors
```python
# Standard manim colors
RED, BLUE, YELLOW, GREEN, WHITE, GREY

# Use YELLOW for highlights/annotations
# Use RED/BLUE for data (consistent with the content meaning)
```

## Resolution

Use low resolution (720p) during development for fast iteration, then switch to high resolution (1080p or 4K) for final render.

```python
# Low-res for development (fast)
config.pixel_height = 720
config.pixel_width = 1280
config.frame_rate = 30

# High-res for final render
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_rate = 60

# 4K for maximum quality
config.pixel_height = 2160
config.pixel_width = 3840
config.frame_rate = 60
```

In marimo notebooks, use a dropdown/toggle to switch between presets.
