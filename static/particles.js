const canvas = document.getElementById('bg-canvas');
const ctx = canvas.getContext('2d');

let width, height;
function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
}
resize();
window.addEventListener('resize', resize);

const NUM_PARTICLES = 110;
const MAX_DISTANCE = 150;
const particles = [];

const mouse = { x: -9999, y: -9999 };
window.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
});
window.addEventListener('mouseleave', () => {
    mouse.x = -9999;
    mouse.y = -9999;
});

for (let i = 0; i < NUM_PARTICLES; i++) {
    particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 1.8 + 1,
        pulseOffset: Math.random() * Math.PI * 2,
        pulseSpeed: 0.015 + Math.random() * 0.02
    });
}

let scanY = 0;
const SCAN_SPEED = 0.6;

function draw(time) {
    ctx.clearRect(0, 0, width, height);

    for (const p of particles) {
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const distToMouse = Math.sqrt(dx * dx + dy * dy);

        if (distToMouse < 160) {
            const push = (160 - distToMouse) / 160;
            p.x -= (dx / distToMouse) * push * 1.2;
            p.y -= (dy / distToMouse) * push * 1.2;
        }

        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > width)  p.vx *= -1;
        if (p.y < 0 || p.y > height) p.vy *= -1;

        p.x = Math.max(0, Math.min(width, p.x));
        p.y = Math.max(0, Math.min(height, p.y));
    }

    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const a = particles[i];
            const b = particles[j];
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < MAX_DISTANCE) {
                const opacity = (1 - dist / MAX_DISTANCE) * 0.22;
                ctx.strokeStyle = `rgba(34, 211, 238, ${opacity})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                ctx.stroke();
            }
        }
    }

    for (const p of particles) {
        const pulse = (Math.sin(time * p.pulseSpeed + p.pulseOffset) + 1) / 2;
        const glowRadius = p.radius + pulse * 1.5;
        const alpha = 0.35 + pulse * 0.45;

        ctx.fillStyle = `rgba(34, 211, 238, ${alpha * 0.25})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, glowRadius * 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(34, 211, 238, ${alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, glowRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    scanY += SCAN_SPEED;
    if (scanY > height + 100) scanY = -100;

    const scanGradient = ctx.createLinearGradient(0, scanY - 60, 0, scanY + 60);
    scanGradient.addColorStop(0, 'rgba(34, 211, 238, 0)');
    scanGradient.addColorStop(0.5, 'rgba(34, 211, 238, 0.04)');
    scanGradient.addColorStop(1, 'rgba(34, 211, 238, 0)');
    ctx.fillStyle = scanGradient;
    ctx.fillRect(0, scanY - 60, width, 120);

    requestAnimationFrame(draw);
}

requestAnimationFrame(draw);