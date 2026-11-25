/**
 * High-end Canvas Visualizer for the Astro-Dendritic Spiking Neuron
 * Renders:
 * 1. Somatic membrane potential transitions and flash rings on action potentials.
 * 2. Dendritic branch plateau states with electric spark travel.
 * 3. Astrocytic calcium cloud glowing and emitting chemical particles.
 * 4. Active synapses and travelling axonal spike pulses.
 * 5. Dynamic oscilloscope graphing for telemetry signals.
 */

class NeuronVisualizer {
    constructor(canvasId, neuron) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.neuron = neuron;
        
        // Particle arrays
        this.spikeParticles = [];
        this.astroParticles = [];
        
        // Animation states
        this.somaFlash = 0.0; // Soma spike flash intensity
        this.dendriteFlash = [0, 0]; // Branch plateau timers
        this.astroPulse = 0.0;
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        // Adjust resolution for Retina screens
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.width = rect.width;
        this.height = rect.height;
    }

    /**
     * Add pre-synaptic spike animation particle traveling along channel line
     */
    addSpikePulse(channelId) {
        this.spikeParticles.push({
            channelId: channelId,
            progress: 0.0,
            speed: 0.04 + Math.random() * 0.02
        });
    }

    /**
     * Trigger glial particle emission from astrocyte towards synapses
     */
    emitAstroParticles(amount) {
        const x = this.width * 0.35;
        const y = this.height * 0.5;
        
        for (let i = 0; i < amount; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = 0.5 + Math.random() * 1.5;
            this.astroParticles.push({
                x: x + (Math.random() - 0.5) * 60,
                y: y + (Math.random() - 0.5) * 120,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                alpha: 0.8,
                life: 1.0,
                decay: 0.01 + Math.random() * 0.015
            });
        }
    }

    /**
     * Render the neuron network morphology
     */
    draw(activeSpikes, somaSpiked, activeTask) {
        // Clear with slight alpha to create motion trails
        this.ctx.fillStyle = 'rgba(7, 7, 10, 0.25)';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Grid Background overlay
        this.drawGridBackground();

        // Draw Pong game if active
        if (activeTask && activeTask.name === "Neuro-Pong Game") {
            this.drawPongGame(activeTask);
        }

        // Node Positions (Responsive layout mapping)
        const somaX = this.width * 0.75;
        const somaY = this.height * 0.5;
        const somaRadius = 45;

        const branchForkX = this.width * 0.5;
        const branchForkY = this.height * 0.5;
        
        // Dendrite endpoints (A and B)
        const dendrites = [
            { x: this.width * 0.35, y: this.height * 0.3 }, // Branch A (top)
            { x: this.width * 0.35, y: this.height * 0.7 }  // Branch B (bottom)
        ];

        // Input Channel Source Positions
        const inputY = [
            this.height * 0.15,
            this.height * 0.35,
            this.height * 0.65,
            this.height * 0.85
        ];
        const inputX = this.width * 0.1;

        // 1. Draw Astrocytic Ca²⁺ Regulatory Cloud (tripartite synapse boundary)
        this.drawAstrocyteCloud(this.width * 0.32, this.height * 0.5);

        // 2. Draw Input Channel Wires
        this.ctx.lineWidth = 2;
        for (let i = 0; i < 4; i++) {
            const destDend = this.neuron.synapses.find(s => s.sourceChannel === i);
            if (!destDend) continue;
            const targetY = destDend.targetBranchId === 0 ? dendrites[0].y : dendrites[1].y;
            const targetX = destDend.targetBranchId === 0 ? dendrites[0].x : dendrites[1].x;

            // Wire path (bezier curve)
            this.ctx.beginPath();
            this.ctx.moveTo(inputX, inputY[i]);
            this.ctx.bezierCurveTo(
                (inputX + targetX) / 2, inputY[i],
                (inputX + targetX) / 2, targetY,
                targetX, targetY
            );
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            this.ctx.stroke();

            // Source Input Node
            this.ctx.beginPath();
            this.ctx.arc(inputX, inputY[i], 6, 0, Math.PI * 2);
            this.ctx.fillStyle = activeSpikes[i] ? 'var(--accent-voltage)' : 'var(--text-muted)';
            if (activeSpikes[i]) {
                this.ctx.shadowBlur = 15;
                this.ctx.shadowColor = 'var(--accent-voltage)';
                this.addSpikePulse(i);
            } else {
                this.ctx.shadowBlur = 0;
            }
            this.ctx.fill();
            this.ctx.shadowBlur = 0; // reset
        }

        // 3. Draw Dendritic Branches
        this.neuron.branches.forEach((b, idx) => {
            const dest = dendrites[idx];
            
            // Draw branch segment
            this.ctx.beginPath();
            this.ctx.moveTo(somaX, somaY);
            this.ctx.bezierCurveTo(
                branchForkX, branchForkY,
                branchForkX, dest.y,
                dest.x, dest.y
            );
            
            // Compute line weight and glow based on branch potential
            const potentialNorm = Math.min(1.0, b.potential / b.threshold);
            this.ctx.lineWidth = 4 + potentialNorm * 4;
            
            if (b.isPlateau) {
                this.ctx.strokeStyle = `rgba(168, 85, 247, 0.85)`;
                this.ctx.shadowBlur = 25;
                this.ctx.shadowColor = 'var(--accent-dendrite)';
            } else {
                this.ctx.strokeStyle = `rgba(168, 85, 247, ${0.15 + potentialNorm * 0.4})`;
                this.ctx.shadowBlur = potentialNorm * 10;
                this.ctx.shadowColor = 'var(--accent-dendrite)';
            }
            this.ctx.stroke();
            this.ctx.shadowBlur = 0; // reset
            
            // Dendritic active endpoint
            this.ctx.beginPath();
            this.ctx.arc(dest.x, dest.y, 8, 0, Math.PI * 2);
            this.ctx.fillStyle = b.isPlateau ? 'var(--accent-dendrite)' : `rgba(168, 85, 247, ${0.3 + potentialNorm * 0.7})`;
            if (b.isPlateau) {
                this.ctx.shadowBlur = 20;
                this.ctx.shadowColor = 'var(--accent-dendrite)';
            }
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        });

        // 4. Draw Synapses
        this.neuron.synapses.forEach(syn => {
            const destDend = dendrites[syn.targetBranchId];
            // Compute offset positioning along the dendritic arbor endpoint
            // (Distribute synapses around target branch coordinates)
            const offset = (syn.id - 1.5) * 16;
            const synX = destDend.x + offset;
            const synY = destDend.y + Math.sin(offset) * 10;

            // Render synapse node
            this.ctx.beginPath();
            this.ctx.arc(synX, synY, 5 + syn.effectiveWeight * 1.5, 0, Math.PI * 2);
            
            // Scale color brightness based on weight
            const alpha = 0.4 + (syn.effectiveWeight / 5.0) * 0.6;
            this.ctx.fillStyle = `rgba(16, 185, 129, ${alpha})`;
            
            if (syn.preTrace > 0.05) {
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.lineWidth = 1.5;
                this.ctx.stroke();
                this.ctx.shadowBlur = 10;
                this.ctx.shadowColor = 'var(--accent-synapse)';
            }
            
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        });

        // 5. Draw Traveling Spikes (axonal particles)
        this.spikeParticles.forEach((p, idx) => {
            p.progress += p.speed;
            if (p.progress >= 1.0) {
                this.spikeParticles.splice(idx, 1);
                return;
            }

            // Interpolate position along bezier curve
            const startY = inputY[p.channelId];
            const destDendIdx = this.neuron.synapses.find(s => s.sourceChannel === p.channelId).targetBranchId;
            const targetY = dendrites[destDendIdx].y;
            const targetX = dendrites[destDendIdx].x;

            const t = p.progress;
            const cp1x = (inputX + targetX) / 2;
            const cp1y = startY;
            const cp2x = (inputX + targetX) / 2;
            const cp2y = targetY;

            // Cubic Bezier formula
            const px = Math.pow(1-t, 3)*inputX + 3*Math.pow(1-t, 2)*t*cp1x + 3*(1-t)*t*t*cp2x + Math.pow(t, 3)*targetX;
            const py = Math.pow(1-t, 3)*startY + 3*Math.pow(1-t, 2)*t*cp1y + 3*(1-t)*t*t*cp2y + Math.pow(t, 3)*targetY;

            this.ctx.beginPath();
            this.ctx.arc(px, py, 4, 0, Math.PI * 2);
            this.ctx.fillStyle = '#ffffff';
            this.ctx.shadowBlur = 12;
            this.ctx.shadowColor = 'var(--accent-voltage)';
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        });

        // 6. Draw Astrocyte Chemical Diffusion Particles
        if (this.neuron.astrocyte.calcium > 0.3 && Math.random() < 0.15) {
            this.emitAstroParticles(2);
        }
        this.astroParticles.forEach((p, idx) => {
            p.x += p.vx;
            p.y += p.vy;
            p.life -= p.decay;
            
            if (p.life <= 0) {
                this.astroParticles.splice(idx, 1);
                return;
            }

            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 3 * p.life, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(236, 72, 153, ${p.life * 0.4})`;
            this.ctx.shadowBlur = 4;
            this.ctx.shadowColor = 'var(--accent-astro)';
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        });

        // 7. Draw Soma
        if (somaSpiked) {
            this.somaFlash = 1.0;
        } else {
            this.somaFlash -= 0.06;
            if (this.somaFlash < 0) this.somaFlash = 0;
        }

        // Soma Outer Glow Ring
        this.ctx.beginPath();
        this.ctx.arc(somaX, somaY, somaRadius + this.somaFlash * 25, 0, Math.PI * 2);
        this.ctx.strokeStyle = `rgba(99, 102, 241, ${0.1 + this.somaFlash * 0.6})`;
        this.ctx.lineWidth = 2 + this.somaFlash * 5;
        this.ctx.stroke();

        // Soma Body
        this.ctx.beginPath();
        this.ctx.arc(somaX, somaY, somaRadius, 0, Math.PI * 2);
        
        // Voltage status mapping
        const voltNorm = Math.min(1.0, this.neuron.soma.v / this.neuron.soma.threshold);
        const grad = this.ctx.createRadialGradient(somaX, somaY, 0, somaX, somaY, somaRadius);
        
        grad.addColorStop(0, somaSpiked ? '#ffffff' : `rgba(99, 102, 241, ${0.4 + voltNorm * 0.5})`);
        grad.addColorStop(1, `rgba(15, 15, 23, 0.9)`);
        
        this.ctx.fillStyle = grad;
        this.ctx.strokeStyle = somaSpiked ? '#ffffff' : 'rgba(99, 102, 241, 0.4)';
        this.ctx.lineWidth = 2;
        
        if (voltNorm > 0.1 || somaSpiked) {
            this.ctx.shadowBlur = 10 + voltNorm * 20 + this.somaFlash * 30;
            this.ctx.shadowColor = 'var(--accent-soma)';
        }
        this.ctx.fill();
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;

        // Draw LMU Delay Vector Telemetry Ring (visual representation of memory)
        if (this.neuron.features.lmu) {
            this.drawLMURing(somaX, somaY, somaRadius + 10);
        }
    }

    drawPongGame(task) {
        const w = this.width;
        const h = this.height;

        // Center line
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([8, 8]);
        this.ctx.beginPath();
        this.ctx.moveTo(w / 2, 0);
        this.ctx.lineTo(w / 2, h);
        this.ctx.stroke();
        this.ctx.setLineDash([]); // reset

        // Draw Dopamine Gated Border Glow
        if (Math.abs(task.dopamine) > 0.05) {
            const alpha = Math.min(0.3, Math.abs(task.dopamine) * 0.15);
            this.ctx.strokeStyle = task.dopamine > 0 ? `rgba(16, 185, 129, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
            this.ctx.lineWidth = 6;
            this.ctx.strokeRect(0, 0, w, h);
        }

        // Draw Left Paddle (Opponent AI)
        const oppY = (task.opponentY / 100) * h;
        const oppH = (task.opponentHeight / 100) * h;
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.roundRect(15, oppY - oppH / 2, 8, oppH, 4);
        this.ctx.fill();
        this.ctx.stroke();

        // Draw Right Paddle (Neuron Controlled)
        const plyY = (task.paddleY / 100) * h;
        const plyH = (task.paddleHeight / 100) * h;
        this.ctx.fillStyle = 'rgba(99, 102, 241, 0.2)';
        this.ctx.strokeStyle = 'var(--accent-soma)';
        this.ctx.lineWidth = 1.5;
        
        this.ctx.save();
        this.ctx.shadowBlur = 12;
        this.ctx.shadowColor = 'var(--accent-soma)';
        this.ctx.beginPath();
        this.ctx.roundRect(w - 23, plyY - plyH / 2, 8, plyH, 4);
        this.ctx.fill();
        this.ctx.stroke();
        this.ctx.restore();

        // Draw Ball
        const ballX = (task.ballX / 100) * w;
        const ballY = (task.ballY / 100) * h;
        this.ctx.beginPath();
        this.ctx.arc(ballX, ballY, 6, 0, Math.PI * 2);
        this.ctx.fillStyle = '#ffffff';
        this.ctx.save();
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = '#ffffff';
        this.ctx.fill();
        this.ctx.restore();
    }

    drawGridBackground() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.015)';
        this.ctx.lineWidth = 1;
        const gridSpacing = 40;
        
        for (let x = 0; x < this.width; x += gridSpacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        for (let y = 0; y < this.height; y += gridSpacing) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
    }

    drawAstrocyteCloud(x, y) {
        const ca = this.neuron.astrocyte.calcium;
        const limitCa = Math.min(3.0, ca) / 3.0; // normalized max
        
        this.ctx.save();
        this.ctx.beginPath();
        
        // Semi-transparent astrocytic envelope shape (blob)
        this.ctx.arc(x, y - 40, 50 + limitCa * 20, 0, Math.PI * 2);
        this.ctx.arc(x + 40, y + 20, 60 + limitCa * 15, 0, Math.PI * 2);
        this.ctx.arc(x - 30, y + 30, 55 + limitCa * 25, 0, Math.PI * 2);
        
        this.ctx.closePath();
        
        const grad = this.ctx.createRadialGradient(x, y, 20, x, y, 110 + limitCa * 30);
        grad.addColorStop(0, `rgba(236, 72, 153, ${0.03 + limitCa * 0.15})`);
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        
        this.ctx.fillStyle = grad;
        this.ctx.fill();
        
        // Soft glowing border
        if (limitCa > 0.05) {
            this.ctx.strokeStyle = `rgba(236, 72, 153, ${limitCa * 0.15})`;
            this.ctx.lineWidth = 1.5;
            this.ctx.stroke();
        }
        this.ctx.restore();
    }

    drawLMURing(somaX, somaY, radius) {
        // Render 4 LMU state coefficients around the soma
        const lmu = this.neuron.branches[0].lmuState; // use Branch A's delay states
        const numStates = lmu.length;
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.arc(somaX, somaY, radius, 0, Math.PI * 2);
        this.ctx.stroke();

        for (let i = 0; i < numStates; i++) {
            const angle = (i / numStates) * Math.PI * 2 - Math.PI / 2;
            const val = lmu[i];
            
            // Offset point based on coefficient magnitude
            const ptX = somaX + Math.cos(angle) * (radius + val * 12);
            const ptY = somaY + Math.sin(angle) * (radius + val * 12);

            // Connect lines
            this.ctx.beginPath();
            this.ctx.moveTo(somaX + Math.cos(angle) * radius, somaY + Math.sin(angle) * radius);
            this.ctx.lineTo(ptX, ptY);
            this.ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();

            // Draw data bead
            this.ctx.beginPath();
            this.ctx.arc(ptX, ptY, 3, 0, Math.PI * 2);
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fill();
        }
    }
}

class TelemetryOscilloscope {
    constructor(canvasId, bufferSize = 150) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.bufferSize = bufferSize;
        this.buffers = {};
        
        this.resize();
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.width = rect.width;
        this.height = rect.height;
    }

    pushValue(lineId, value) {
        if (!this.buffers[lineId]) {
            this.buffers[lineId] = [];
        }
        this.buffers[lineId].push(value);
        if (this.buffers[lineId].length > this.bufferSize) {
            this.buffers[lineId].shift();
        }
    }

    draw(lineConfigs) {
        this.ctx.fillStyle = '#060609';
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Draw zero grid line
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.height * 0.8);
        this.ctx.lineTo(this.width, this.height * 0.8);
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();

        lineConfigs.forEach(config => {
            const buffer = this.buffers[config.id];
            if (!buffer || buffer.length < 2) return;

            this.ctx.beginPath();
            const step = this.width / this.bufferSize;
            
            for (let i = 0; i < buffer.length; i++) {
                const x = i * step;
                // Scale value to fit y boundaries
                const normalizedVal = (buffer[i] - config.min) / (config.max - config.min);
                const y = this.height - normalizedVal * this.height;
                
                if (i === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }

            this.ctx.strokeStyle = config.color;
            this.ctx.lineWidth = config.lineWidth || 2;
            
            if (config.glow) {
                this.ctx.shadowBlur = 8;
                this.ctx.shadowColor = config.color;
            }
            this.ctx.stroke();
            this.ctx.shadowBlur = 0; // reset
        });
    }
}
