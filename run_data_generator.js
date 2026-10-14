/**
 * Data Generator for publication-ready plots
 * Runs the 11 tasks and generates experiment_results.json
 * in the exact format expected by plot_results.py
 */

const fs   = require('fs');
const path = require('path');

global.document = { addEventListener:()=>{}, getElementById:()=>({getContext:()=>({scale:()=>{}})}), querySelectorAll:()=>[] };
global.window   = { addEventListener:()=>{}, devicePixelRatio:1 };

const neuronCode = fs.readFileSync(path.join(__dirname,'experiment','neuron.js'),'utf8');
const tasksCode  = fs.readFileSync(path.join(__dirname,'experiment','tasks.js'),'utf8');

eval(neuronCode + `; global.AstroDendriticNeuron=AstroDendriticNeuron; global.Synapse=Synapse; global.DendriticBranch=DendriticBranch; global.Astrocyte=Astrocyte; global.Soma=Soma;`);
eval(tasksCode  + `; global.XORTask=XORTask; global.SequenceTask=SequenceTask; global.NoiseTask=NoiseTask; global.PongTask=PongTask; global.ConditioningTask=ConditioningTask; global.FrequencyTask=FrequencyTask; global.NavigationTask=NavigationTask; global.ReconstructionTask=ReconstructionTask; global.TMazeTask=TMazeTask; global.Navigation2DTask=Navigation2DTask; global.IntervalTimingTask=IntervalTimingTask;`);

const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
const std  = a => { if(a.length<=1) return 0; const m=mean(a); return Math.sqrt(a.reduce((s,v)=>s+Math.pow(v-m,2),0)/(a.length-1)); };

function applyMask(neuron, mask) {
    if (mask === 'full') return;
    if (mask === 'dend') {
        neuron.features.astrocyte = false;
        neuron.features.lmu       = false;
        neuron.allowsAstrocyte    = false;
        return;
    }
    if (mask === 'point') {
        neuron.features.dendrites = false;
        neuron.features.astrocyte = false;
        neuron.features.lmu       = false;
        neuron.allowsAstrocyte    = false;
        return;
    }
}

const data = {};

// 1. XOR
console.log("Running XOR data generation...");
const xorTask = new XORTask();
const xorNeuron = new AstroDendriticNeuron();
xorTask.setupNeuron(xorNeuron);

const xorHistory = [];
// Run a single representative run to collect training history
for (let epoch = 1; epoch <= 100; epoch++) {
    const epochLen = xorTask.epochDuration;
    for (let t = 0; t < epochLen; t++) {
        const globalTick = t + (epoch-1)*epochLen;
        const inp = xorTask.generateInputs(globalTick);
        const spiked = xorNeuron.tick(inp, 1.0);
        xorTask.evaluate(xorNeuron, globalTick, spiked);
    }
    xorHistory.push({ epoch: epoch, accuracy: xorTask.accuracy });
}

// Final weights
const xorWeights = xorNeuron.synapses.map(s => ({
    channel: s.sourceChannel,
    branch: s.targetBranchId,
    w: s.weight
}));

data.xor = {
    history: xorHistory,
    finalWeights: xorWeights
};

// 2. Sequence Detection
console.log("Running Sequence data generation...");
const seqTask = new SequenceTask();
const seqNeuron = new AstroDendriticNeuron();
seqTask.setupNeuron(seqNeuron);
seqTask.jitter = 0;

const seqTimeline = [];
for (let t = 0; t < 200; t++) {
    const inp = seqTask.generateInputs(t);
    const spiked = seqNeuron.tick(inp, 1.0);
    seqTask.evaluate(seqNeuron, t, spiked);
    seqTimeline.push({
        tick: t,
        soma_v: seqNeuron.soma.v,
        b0_pot: seqNeuron.branches[0].potential,
        b1_pot: seqNeuron.branches[1].potential,
        b0_isPl: seqNeuron.branches[0].isPlateau,
        b1_isPl: seqNeuron.branches[1].isPlateau,
        spiked: spiked
    });
}

// Jitter Sweep
const jitters = [0, 5, 10, 15, 20, 25];
const nadAccs = [];
const nadStds = [];
const pointAccs = [];
const pointStds = [];

for (const j of jitters) {
    const runJitter = (mask) => {
        const accs = [];
        for (let run = 0; run < 30; run++) {
            const neuron = new AstroDendriticNeuron();
            const task = new SequenceTask();
            task.setupNeuron(neuron);
            applyMask(neuron, mask);
            task.jitter = j;
            for (let t = 0; t < 2000; t++) {
                const inp = task.generateInputs(t);
                const spiked = neuron.tick(inp, 1.0);
                task.evaluate(neuron, t, spiked);
            }
            accs.push(task.accuracy);
        }
        return accs;
    };
    
    const nadRaw = runJitter('full');
    const pointRaw = runJitter('point');
    
    nadAccs.push(mean(nadRaw));
    nadStds.push(std(nadRaw));
    pointAccs.push(mean(pointRaw));
    pointStds.push(std(pointRaw));
}

data.sequence = {
    timeline: seqTimeline,
    jitter_sweep: {
        jitters: jitters,
        nad_acc: nadAccs,
        nad_std: nadStds,
        point_acc: pointAccs,
        point_std: pointStds
    }
};

// 3. Homeostatic Noise
console.log("Running Noise task data generation...");
const noiseTask = new NoiseTask();
const noiseNeuron = new AstroDendriticNeuron();
noiseTask.setupNeuron(noiseNeuron);

const noiseTimeline = [];
for (let t = 0; t < 400; t++) {
    const inp = noiseTask.generateInputs(t);
    const spiked = noiseNeuron.tick(inp, 1.0);
    noiseTask.evaluate(noiseNeuron, t, spiked);
    noiseTimeline.push({
        tick: t,
        calcium: noiseNeuron.astrocyte.calcium,
        gain: noiseNeuron.astrocyte.gainFactor
    });
}

// SNR Sweep
const noiseRates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
const astroOnSpikes = [];
const astroOffSpikes = [];
const pointSpikes = [];
const astroOnFid = [];
const astroOffFid = [];
const pointFid = [];

for (const nr of noiseRates) {
    const runNoise = (mask) => {
        const rawSpikes = [];
        const rawFid = [];
        for (let run = 0; run < 30; run++) {
            const neuron = new AstroDendriticNeuron();
            const task = new NoiseTask();
            task.setupNeuron(neuron);
            applyMask(neuron, mask);
            task.noiseRate = nr;
            
            let spk = 0;
            for (let t = 0; t < 800; t++) {
                const inp = task.generateInputs(t);
                // Force astrocyte ON or OFF specifically in the two phases
                const phaseIndex = Math.floor((t % task.epochDuration) / task.patternDuration);
                if (mask === 'full') {
                    neuron.features.astrocyte = (phaseIndex === 1);
                } else {
                    neuron.features.astrocyte = false;
                }
                const spiked = neuron.tick(inp, 1.0);
                task.evaluate(neuron, t, spiked);
                if (spiked && phaseIndex === 1) spk++;
            }
            rawSpikes.push(spk);
            rawFid.push(task.phase1Fidelity);
        }
        return { spikes: mean(rawSpikes), fidelity: mean(rawFid) };
    };
    
    const onRes = runNoise('full');
    // For Astro OFF, we run with 'full' mask but force features.astrocyte = false during both phases
    const offRes = runNoise('dend');
    const ptRes = runNoise('point');
    
    astroOnSpikes.push(onRes.spikes);
    astroOffSpikes.push(offRes.spikes);
    pointSpikes.push(ptRes.spikes);
    
    astroOnFid.push(onRes.fidelity);
    astroOffFid.push(offRes.fidelity);
    pointFid.push(ptRes.fidelity);
}

data.noise = {
    timeline: noiseTimeline,
    noise_sweep: {
        noise_rates: noiseRates,
        astro_on_spikes: astroOnSpikes,
        astro_off_spikes: astroOffSpikes,
        point_spikes: pointSpikes,
        astro_on_fidelity: astroOnFid,
        astro_off_fidelity: astroOffFid,
        point_fidelity: pointFid
    }
};

// 4. Pong
console.log("Running Pong data generation...");
const pongTask = new PongTask();
const pongNeuron = new AstroDendriticNeuron();
pongTask.setupNeuron(pongNeuron);

const pongHistory = [];
for (let t = 0; t < 8000; t++) {
    const inp = pongTask.generateInputs(t);
    const spiked = pongNeuron.tick(inp, 1.0);
    pongTask.evaluate(pongNeuron, t, spiked);
    if (t % 200 === 0) {
        pongHistory.push({ tick: t, accuracy: pongTask.accuracy });
    }
}
data.pong = {
    history: pongHistory
};

// 5. Conditioning
console.log("Running Conditioning data generation...");
const condTask = new ConditioningTask();
const condNeuron = new AstroDendriticNeuron();
condTask.setupNeuron(condNeuron);

const condHistory = [];
for (let t = 0; t < 2400; t++) {
    const inp = condTask.generateInputs(t);
    const spiked = condNeuron.tick(inp, 1.0);
    condTask.evaluate(condNeuron, t, spiked);
    
    const epochLen = condTask.epochDuration;
    if (t === epochLen - 1) {
        condHistory.push({ epoch: 1, weights: [condNeuron.synapses[0].weight, condNeuron.synapses[1].weight, condNeuron.synapses[2].weight] });
    } else if (t === 2 * epochLen - 1) {
        condHistory.push({ epoch: 2, weights: [condNeuron.synapses[0].weight, condNeuron.synapses[1].weight, condNeuron.synapses[2].weight] });
    } else if (t === 3 * epochLen - 1) {
        condHistory.push({ epoch: 3, weights: [condNeuron.synapses[0].weight, condNeuron.synapses[1].weight, condNeuron.synapses[2].weight] });
    }
}
data.conditioning = {
    history: condHistory
};

// 6. Frequency
console.log("Running Frequency data generation...");
const freqFrequencies = [5, 10, 20, 30, 40, 50, 60, 80, 100];
const nadRates = [];
const nadRatesStd = [];
const pointRates = [];
const pointRatesStd = [];

for (const f of freqFrequencies) {
    const runFreq = (mask) => {
        const rates = [];
        const interval = Math.round(1000/f);
        for (let run = 0; run < 30; run++) {
            const neuron = new AstroDendriticNeuron();
            const task = new FrequencyTask();
            task.setupNeuron(neuron);
            applyMask(neuron, mask);
            task.customInterval = interval;
            
            let spk = 0;
            for (let t = 0; t < 200; t++) {
                const inp = task.generateInputs(t);
                if (neuron.tick(inp, 1.0)) spk++;
            }
            rates.push(spk * 5.0); // 200 ticks = 200ms -> rate in Hz
        }
        return rates;
    };
    
    const nadRaw = runFreq('full');
    const pointRaw = runFreq('point');
    
    nadRates.push(mean(nadRaw));
    nadRatesStd.push(std(nadRaw));
    pointRates.push(mean(pointRaw));
    pointRatesStd.push(std(pointRaw));
}

// Frequency Timeline at 50Hz (interval = 20)
const freqTimelineTask = new FrequencyTask();
const freqTimelineNeuron = new AstroDendriticNeuron();
freqTimelineTask.setupNeuron(freqTimelineNeuron);
freqTimelineTask.customInterval = 20;

const freqTimeline = [];
for (let t = 0; t < 200; t++) {
    const inp = freqTimelineTask.generateInputs(t);
    const spiked = freqTimelineNeuron.tick(inp, 1.0);
    freqTimelineTask.evaluate(freqTimelineNeuron, t, spiked);
    freqTimeline.push({
        tick: t,
        soma_v: freqTimelineNeuron.soma.v,
        b0_pot: freqTimelineNeuron.branches[0].potential,
        b0_isPl: freqTimelineNeuron.branches[0].isPlateau,
        spiked: spiked,
        input_spike: inp[0]
    });
}

data.frequency = {
    freq_sweep: {
        frequencies: freqFrequencies,
        nad_rate: nadRates,
        nad_rate_std: nadRatesStd,
        point_rate: pointRates,
        point_rate_std: pointRatesStd
    },
    timeline: freqTimeline
};

// 7. Navigation
console.log("Running Navigation data generation...");
const navTask = new NavigationTask();
const navNeuron = new AstroDendriticNeuron();
navTask.setupNeuron(navNeuron);

const navHistory = [];
for (let t = 0; t < 5000; t++) {
    const inp = navTask.generateInputs(t);
    const spiked = navNeuron.tick(inp, 1.0);
    navTask.evaluate(navNeuron, t, spiked);
    if (t > 0 && t % 1000 === 0) {
        navHistory.push({ epoch: t / 1000, accuracy: navTask.accuracy });
    }
}
data.navigation = {
    history: navHistory
};

// 8. LMU Reconstruction
console.log("Running LMU Reconstruction data generation...");
const reconTask = new ReconstructionTask();
const reconNeuron = new AstroDendriticNeuron();
reconTask.lmuOrder = 4;
reconTask.setupNeuron(reconNeuron);

const lmuTimelineTicks = [];
const trueDelayed = [];
const reconstructed = [];

for (let t = 0; t < 300; t++) {
    const inp = reconTask.generateInputs(t);
    reconNeuron.tick(inp, 1.0);
    reconTask.evaluate(reconNeuron, t, false);
    
    if (t >= reconTask.delay) {
        lmuTimelineTicks.push(t);
        trueDelayed.push(reconTask.trueDelayedSignals[reconTask.trueDelayedSignals.length - 1]);
        reconstructed.push(reconTask.reconstructedSignals[reconTask.reconstructedSignals.length - 1]);
    }
}

// LMU Order Sweep
const lmuOrders = [2, 3, 4, 5, 6];
const lmuMse = [];
const lmuMseStd = [];

for (const ord of lmuOrders) {
    const mses = [];
    for (let run = 0; run < 30; run++) {
        const neuron = new AstroDendriticNeuron();
        const task = new ReconstructionTask();
        task.lmuOrder = ord;
        task.setupNeuron(neuron);
        
        for (let t = 0; t < 300; t++) {
            const inp = task.generateInputs(t);
            neuron.tick(inp, 1.0);
            task.evaluate(neuron, t, false);
        }
        mses.push(task.mse);
    }
    lmuMse.push(mean(mses));
    lmuMseStd.push(std(mses));
}

data.lmu_reconstruction = {
    timeline: {
        ticks: lmuTimelineTicks,
        true_delayed: trueDelayed,
        reconstructed: reconstructed
    },
    order_sweep: {
        orders: lmuOrders,
        mse: lmuMse,
        mse_std: lmuMseStd
    }
};

// Save outputs
fs.writeFileSync('e:/Neuron_v2/experiment_results.json', JSON.stringify(data, null, 2));
console.log("Successfully generated all timeline and sweep data in e:/Neuron_v2/experiment_results.json!");
