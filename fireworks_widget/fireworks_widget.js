function createFirework(canvas, x, y) {
  const colors = [
    '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff',
    '#ff8800', '#88ff00', '#0088ff', '#ff0088', '#8800ff', '#00ff88'
  ];
  const particles = [];
  const particleCount = 50 + Math.random() * 30;
  const color = colors[Math.floor(Math.random() * colors.length)];
  
  for (let i = 0; i < particleCount; i++) {
    const angle = (Math.PI * 2 * i) / particleCount + Math.random() * 0.5;
    const speed = 2 + Math.random() * 4;
    particles.push({
      x: x,
      y: y,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      color: color,
      life: 1.0,
      decay: 0.015 + Math.random() * 0.01,
      size: 2 + Math.random() * 3
    });
  }
  
  return particles;
}

function render({ model, el }) {
  el.classList.add('fireworks-root');
  
  const canvas = document.createElement('canvas');
  canvas.className = 'fireworks-canvas';
  el.appendChild(canvas);
  
  const button = document.createElement('button');
  button.className = 'fireworks-button';
  button.textContent = '🎆 Launch Fireworks!';
  el.appendChild(button);
  
  const ctx = canvas.getContext('2d');
  let animationId = null;
  let particles = [];
  
  function resizeCanvas() {
    const rect = el.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
  }
  
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Update and draw particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += 0.15; // gravity
      p.vx *= 0.98; // air resistance
      p.life -= p.decay;
      
      if (p.life > 0 && p.x >= 0 && p.x <= canvas.width && p.y >= 0 && p.y <= canvas.height) {
        ctx.globalAlpha = p.life;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
        particles[i] = p;
      } else {
        particles.splice(i, 1);
      }
    }
    
    if (particles.length > 0) {
      animationId = requestAnimationFrame(animate);
    } else {
      animationId = null;
    }
  }
  
  function launchFireworks() {
    // Create multiple fireworks at random positions
    const fireworkCount = 3 + Math.floor(Math.random() * 3);
    for (let i = 0; i < fireworkCount; i++) {
      const x = canvas.width * 0.2 + Math.random() * canvas.width * 0.6;
      const y = canvas.height * 0.2 + Math.random() * canvas.height * 0.3;
      const newParticles = createFirework(canvas, x, y);
      particles = particles.concat(newParticles);
    }
    
    if (!animationId) {
      animate();
    }
  }
  
  button.addEventListener('click', launchFireworks);
  
  const resizeObserver = new ResizeObserver(() => {
    resizeCanvas();
  });
  resizeObserver.observe(el);
  
  resizeCanvas();
  
  // Listen for trigger from Python
  model.on('change:trigger', () => {
    launchFireworks();
  });
  
  return () => {
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
    resizeObserver.disconnect();
  };
}

export default { render };
