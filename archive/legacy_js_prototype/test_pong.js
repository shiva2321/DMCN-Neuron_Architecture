const fs = require('fs');
const path = require('path');

// Mock browser globals
global.document = {
    addEventListener: () => {},
    getElementById: () => ({ getContext: () => ({ scale: () => {} }) }),
    querySelectorAll: () => []
};
global.window = {
    addEventListener: () => {},
    devicePixelRatio: 1
};

// Load neuron.js and tasks.js
const neuronCode = fs.readFileSync(path.join(__dirname, 'experiment', 'neuron.js'), 'utf8');
const tasksCode = fs.readFileSync(path.join(__dirname, 'experiment', 'tasks.js'), 'utf8');

// Evaluate them in global scope
eval(neuronCode + "; global.AstroDendriticNeuron = AstroDendriticNeuron; global.Synapse = Synapse; global.DendriticBranch = DendriticBranch; global.Astrocyte = Astrocyte; global.Soma = Soma;");
eval(tasksCode + "; global.XORTask = XORTask; global.SequenceTask = SequenceTask; global.NoiseTask = NoiseTask; global.PongTask = PongTask;");

console.log("Constructing Neuron and PongTask...");
const neuron = new AstroDendriticNeuron();
const task = new PongTask();
task.setupNeuron(neuron);

// Apply balanced forces test
task.evaluate = function(neuron, tickIndex, spiked) {
    if (spiked) {
        this.paddleY = Math.max(this.paddleHeight / 2, this.paddleY - 22.0); // Balanced spike lift
    } else {
        this.paddleY = Math.min(100 - this.paddleHeight / 2, this.paddleY + 0.8); // Gentle drift
    }
    
    this.opponentY += (this.ballY - this.opponentY) * 0.12;
    this.opponentY = Math.max(this.opponentHeight / 2, Math.min(100 - this.opponentHeight / 2, this.opponentY));

    this.ballX += this.ballVX;
    this.ballY += this.ballVY;
    if (this.ballY <= 1) { this.ballY = 1; this.ballVY = -this.ballVY; }
    if (this.ballY >= 99) { this.ballY = 99; this.ballVY = -this.ballVY; }
    
    // Opponent collision
    if (this.ballX <= 4 && this.ballX >= 2) {
        const halfHeight = this.opponentHeight / 2;
        if (this.ballY >= this.opponentY - halfHeight && this.ballY <= this.opponentY + halfHeight) {
            this.ballX = 4;
            this.ballVX = -this.ballVX * 1.05;
        }
    }
    if (this.ballX < 0) {
        this.hits++;
        this.ballReset();
    }

    // Player collision
    let isHit = false;
    let isMiss = false;
    if (this.ballX >= 96 && this.ballX <= 98) {
        const halfHeight = this.paddleHeight / 2;
        if (this.ballY >= this.paddleY - halfHeight && this.ballY <= this.paddleY + halfHeight) {
            this.ballX = 96;
            this.ballVX = -this.ballVX * 1.05;
            isHit = true;
        }
    }
    if (this.ballX > 100) {
        isMiss = true;
    }

    this.dopamine *= 0.92;

    if (isHit) {
        this.hits++;
        this.dopamine = 2.0;
        if (this.ballY < this.paddleY) {
            neuron.synapses.filter(s => s.sourceChannel === 0).forEach(s => s.weight = Math.min(5.0, s.weight + 0.4));
        } else {
            neuron.synapses.filter(s => s.sourceChannel === 1).forEach(s => s.weight = Math.max(0.1, s.weight - 0.4));
        }
    }

    if (isMiss) {
        this.misses++;
        this.dopamine = -2.0;
        this.ballReset();
        if (this.ballY < this.paddleY) {
            neuron.synapses.filter(s => s.sourceChannel === 0).forEach(s => s.weight = Math.min(5.0, s.weight + 0.6));
        } else {
            neuron.synapses.filter(s => s.sourceChannel === 1).forEach(s => s.weight = Math.max(0.1, s.weight - 0.6));
        }
    }

    const totalAttempts = this.hits + this.misses;
    if (totalAttempts > 0) {
        this.accuracy = (this.hits / totalAttempts) * 100;
    }
};

console.log("Running simulation for 3000 ticks...");
for (let tick = 0; tick < 3000; tick++) {
    const inputs = task.generateInputs(tick);
    const spiked = neuron.tick(inputs, 1.0);
    task.evaluate(neuron, tick, spiked);
    
    if (tick % 500 === 0) {
        console.log(`[Tick ${tick}] Hits: ${task.hits}, Misses: ${task.misses}, Accuracy: ${task.accuracy.toFixed(1)}%, Weights: Above=${neuron.synapses[0].weight.toFixed(2)}, Below=${neuron.synapses[1].weight.toFixed(2)}`);
    }
}

console.log(`Final stats - Hits: ${task.hits}, Misses: ${task.misses}, Accuracy: ${task.accuracy.toFixed(1)}%`);
console.log("Final weights:", neuron.synapses.map(s => `${s.sourceChannel}->${s.targetBranchId}: w=${s.weight.toFixed(2)}`));
