/**
 * UI Controller and Simulation Coordinator
 * Binds DOM elements, manages parameter changes, handles execution loops, 
 * and feeds telemetry data into real-time oscilloscopes.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Core Engine & Tasks
    const neuron = new AstroDendriticNeuron();
    
    const tasks = {
        xor: new XORTask(),
        sequence: new SequenceTask(),
        noise: new NoiseTask(),
        pong: new PongTask(),
        conditioning: new ConditioningTask(),
        frequency: new FrequencyTask(),
        navigation: new NavigationTask()
    };
    
    let activeTask = tasks.xor;
    activeTask.setupNeuron(neuron);

    // 2. Initialize Visualizers & Oscilloscopes
    const mainVisualizer = new NeuronVisualizer('neuron-canvas', neuron);
    
    const oscSoma = new TelemetryOscilloscope('soma-telemetry', 150);
    const oscDendrite = new TelemetryOscilloscope('dendrite-telemetry', 150);
    const oscAstro = new TelemetryOscilloscope('astro-telemetry', 150);
    const oscProgress = new TelemetryOscilloscope('progress-telemetry', 150);

    // 3. UI State variables
    let isPlaying = true;
    let tickIndex = 0;
    let totalSpikes = 0;

    // DOM Controls
    const btnPlayPause = document.getElementById('btn-play-pause');
    const btnStep = document.getElementById('btn-step');
    const btnReset = document.getElementById('btn-reset');
    
    const toggleDendrites = document.getElementById('toggle-dendrites');
    const toggleAstrocyte = document.getElementById('toggle-astrocyte');
    const toggleLMU = document.getElementById('toggle-lmu');

    // Parameters
    const paramVth = document.getElementById('param-vth');
    const paramDth = document.getElementById('param-dth');
    const paramTplateau = document.getElementById('param-tplateau');
    const paramAstroGain = document.getElementById('param-astrogain');
    const paramStdpLr = document.getElementById('param-stdplr');
    const paramSpeed = document.getElementById('param-speed');

    // Value Indicators
    const valVth = document.getElementById('val-vth');
    const valDth = document.getElementById('val-dth');
    const valTplateau = document.getElementById('val-tplateau');
    const valAstroGain = document.getElementById('val-astrogain');
    const valStdpLr = document.getElementById('val-stdplr');
    const valSpeed = document.getElementById('val-speed');

    let simSpeed = 60; // Default max speed (Hz)
    let lastStepTime = 0;

    // Stats
    const statSpikes = document.getElementById('stat-spikes');
    const statEpoch = document.getElementById('stat-epoch');
    const statAccuracy = document.getElementById('stat-accuracy');
    const statCalcium = document.getElementById('stat-calcium');
    
    // Floating Pills
    const metricVoltage = document.getElementById('metric-voltage');
    const metricGain = document.getElementById('metric-gain');
    const statusText = document.getElementById('simulation-status');

    // 4. Parameter Binding Listeners
    function updateParameters() {
        neuron.soma.threshold = parseFloat(paramVth.value);
        valVth.textContent = neuron.soma.threshold.toFixed(1);

        neuron.branches.forEach(b => {
            b.threshold = parseFloat(paramDth.value);
            b.plateauDuration = parseInt(paramTplateau.value);
        });
        valDth.textContent = parseFloat(paramDth.value).toFixed(1);
        valTplateau.textContent = paramTplateau.value + 'ms';

        neuron.astrocyte.gainStrength = parseFloat(paramAstroGain.value);
        valAstroGain.textContent = neuron.astrocyte.gainStrength.toFixed(2);

        neuron.learningRate = parseFloat(paramStdpLr.value);
        valStdpLr.textContent = neuron.learningRate.toFixed(3);
    }

    [paramVth, paramDth, paramTplateau, paramAstroGain, paramStdpLr].forEach(el => {
        el.addEventListener('input', updateParameters);
    });

    paramSpeed.addEventListener('input', () => {
        simSpeed = parseInt(paramSpeed.value);
        valSpeed.textContent = simSpeed === 60 ? 'Max (60Hz)' : simSpeed + ' Hz';
    });

    // Synchronize UI values with neuron's default state
    function syncSlidersToNeuron() {
        paramVth.value = neuron.soma.threshold;
        paramDth.value = neuron.branches[0].threshold;
        paramTplateau.value = neuron.branches[0].plateauDuration;
        paramAstroGain.value = neuron.astrocyte.gainStrength;
        paramStdpLr.value = neuron.learningRate;
        updateParameters();
    }
    syncSlidersToNeuron();

    // 5. Toggles Binding
    function updateToggles() {
        neuron.features.dendrites = toggleDendrites.checked;
        neuron.features.astrocyte = toggleAstrocyte.checked;
        neuron.features.lmu = toggleLMU.checked;
    }
    [toggleDendrites, toggleAstrocyte, toggleLMU].forEach(el => {
        el.addEventListener('change', updateToggles);
    });

    // 6. Playback Control Listeners
    btnPlayPause.addEventListener('click', () => {
        isPlaying = !isPlaying;
        if (isPlaying) {
            btnPlayPause.textContent = "Pause Simulation";
            btnPlayPause.classList.add('play-btn-glow');
            statusText.textContent = "Simulation Active";
            animationLoop();
        } else {
            btnPlayPause.textContent = "Resume Simulation";
            btnPlayPause.classList.remove('play-btn-glow');
            statusText.textContent = "Simulation Paused";
        }
    });

    btnStep.addEventListener('click', () => {
        if (!isPlaying) {
            simulationStep();
        }
    });

    btnReset.addEventListener('click', () => {
        neuron.reset();
        activeTask.reset();
        tickIndex = 0;
        totalSpikes = 0;
        lastStepTime = 0;
        
        // Reset speed
        paramSpeed.value = 60;
        simSpeed = 60;
        valSpeed.textContent = 'Max (60Hz)';
        
        // Clear telemetry lines
        oscSoma.buffers = {};
        oscDendrite.buffers = {};
        oscAstro.buffers = {};
        oscProgress.buffers = {};
        
        statSpikes.textContent = '0';
        statEpoch.textContent = '0';
        statAccuracy.textContent = '0.0%';
        statCalcium.textContent = '0.00';
        metricVoltage.textContent = '0.00 mV';
        metricGain.textContent = '1.00x';
        
        mainVisualizer.spikeParticles = [];
        mainVisualizer.astroParticles = [];
        
        mainVisualizer.draw([false, false, false, false], false);
    });

    // 7. Task Selection
    const taskCards = document.querySelectorAll('.task-card');
    taskCards.forEach(card => {
        card.addEventListener('click', () => {
            taskCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            
            const taskKey = card.getAttribute('data-task');
            activeTask = tasks[taskKey];
            
            // Reconfigure neuron structure/weights to match task specs
            neuron.reset();
            activeTask.setupNeuron(neuron);
            
            // Synchronize toggles based on task defaults
            toggleDendrites.checked = neuron.features.dendrites;
            toggleAstrocyte.checked = neuron.features.astrocyte;
            toggleLMU.checked = neuron.features.lmu;
            
            syncSlidersToNeuron();
            
            // Trigger a reset
            btnReset.click();
        });
    });

    // 8. Core Simulation Step Frame
    function simulationStep() {
        // A. Generate task input spikes
        const inputSpikes = activeTask.generateInputs(tickIndex);
        
        // B. Run single tick update on neuron
        const spiked = neuron.tick(inputSpikes, 1.0);
        
        // C. Feed spiked result back to task evaluator
        activeTask.evaluate(neuron, tickIndex, spiked);
        
        if (spiked) {
            totalSpikes++;
        }

        // D. Push values to Telemetry Oscilloscopes
        // Soma potential (draw vertical line height on spike event)
        const somaPotentialPlot = spiked ? neuron.soma.threshold * 1.6 : neuron.soma.v;
        oscSoma.pushValue('soma_v', somaPotentialPlot);
        
        // Dendritic potentials
        oscDendrite.pushValue('dend_a', neuron.branches[0].potential);
        oscDendrite.pushValue('dend_b', neuron.branches[1].potential);
        
        // Astrocytic Ca2+ & average weight (or dopamine in Pong/Navigation)
        const isDopamineTask = (activeTask.name === "Neuro-Pong Game" || activeTask.name === "1D Grid Navigation");
        if (isDopamineTask) {
            oscAstro.pushValue('dopamine', activeTask.dopamine);
        } else {
            oscAstro.pushValue('calcium', neuron.astrocyte.calcium);
        }
        const avgWeight = neuron.synapses.reduce((acc, s) => acc + s.weight, 0) / neuron.synapses.length;
        oscAstro.pushValue('weight', avgWeight);

        // Push running accuracy value to progress history every 30 ticks (slower timescale)
        if (tickIndex % 30 === 0) {
            oscProgress.pushValue('accuracy', activeTask.accuracy);
        }

        // E. Render Visualizers
        mainVisualizer.draw(inputSpikes, spiked, activeTask);
        
        oscSoma.draw([
            { id: 'soma_v', color: 'var(--accent-voltage)', min: 0, max: 25, glow: true }
        ]);
        oscDendrite.draw([
            { id: 'dend_a', color: 'var(--accent-dendrite)', min: 0, max: 12 },
            { id: 'dend_b', color: 'var(--accent-voltage)', min: 0, max: 12 }
        ]);
        
        if (isDopamineTask) {
            oscAstro.draw([
                { id: 'dopamine', color: 'var(--accent-astro)', min: -2.2, max: 3.2, lineWidth: 2, glow: true },
                { id: 'weight', color: 'var(--accent-synapse)', min: 0, max: 5.0, lineWidth: 1.5 }
            ]);
        } else {
            oscAstro.draw([
                { id: 'calcium', color: 'var(--accent-astro)', min: 0, max: 2.5, lineWidth: 2, glow: true },
                { id: 'weight', color: 'var(--accent-synapse)', min: 0, max: 5.0, lineWidth: 1.5 }
            ]);
        }

        oscProgress.draw([
            { id: 'accuracy', color: 'var(--accent-synapse)', min: 0, max: 100, lineWidth: 2.5, glow: true }
        ]);

        // F. Update UI Stats Panels
        statSpikes.textContent = totalSpikes;
        statEpoch.textContent = activeTask.epoch;
        statAccuracy.textContent = activeTask.accuracy.toFixed(1) + '%';
        
        if (isDopamineTask) {
            statCalcium.previousElementSibling.textContent = "Dopamine Level:";
            statCalcium.textContent = activeTask.dopamine.toFixed(2);
        } else {
            statCalcium.previousElementSibling.textContent = "Astrocytic Ca²⁺ level:";
            statCalcium.textContent = neuron.astrocyte.calcium.toFixed(2);
        }
        
        metricVoltage.textContent = neuron.soma.v.toFixed(2) + ' mV';
        metricGain.textContent = neuron.astrocyte.gainFactor.toFixed(2) + 'x';
        
        tickIndex++;
    }

    // 9. Main Animation Loop
    function animationLoop(timestamp) {
        if (!isPlaying) return;
        
        if (simSpeed === 60) {
            simulationStep();
        } else {
            const interval = 1000 / simSpeed;
            if (!lastStepTime) lastStepTime = timestamp;
            const elapsed = timestamp - lastStepTime;
            
            if (elapsed >= interval) {
                simulationStep();
                // Adjust lastStepTime keeping any fractional elapsed time
                lastStepTime = timestamp - (elapsed % interval);
            }
        }
        
        requestAnimationFrame(animationLoop);
    }

    // Start on load
    requestAnimationFrame(animationLoop);
});
