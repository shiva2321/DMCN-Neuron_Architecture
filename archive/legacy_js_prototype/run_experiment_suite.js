/**
 * Comprehensive NAD-Neuron Benchmark Suite
 * Compares THREE neuron configurations across 11 tasks:
 *   - FULL NAD  : dendrites + astrocyte + LMU  (all features)
 *   - DEND ONLY : dendrites + no astrocyte + no LMU
 *   - POINT-LIF : no dendrites, no astrocyte, no LMU (standard baseline)
 *
 * This ablation design answers: "What does each component actually contribute?"
 *
 * N = 50 trials per task-condition (fast run; increase to 100 for paper)
 */

const fs   = require('fs');
const path = require('path');

global.document = { addEventListener:()=>{}, getElementById:()=>({getContext:()=>({scale:()=>{}})}), querySelectorAll:()=>[] };
global.window   = { addEventListener:()=>{}, devicePixelRatio:1 };

const neuronCode = fs.readFileSync(path.join(__dirname,'experiment','neuron.js'),'utf8');
const tasksCode  = fs.readFileSync(path.join(__dirname,'experiment','tasks.js'),'utf8');

eval(neuronCode + `; global.AstroDendriticNeuron=AstroDendriticNeuron; global.Synapse=Synapse; global.DendriticBranch=DendriticBranch; global.Astrocyte=Astrocyte; global.Soma=Soma;`);
eval(tasksCode  + `; global.XORTask=XORTask; global.SequenceTask=SequenceTask; global.NoiseTask=NoiseTask; global.PongTask=PongTask; global.ConditioningTask=ConditioningTask; global.FrequencyTask=FrequencyTask; global.NavigationTask=NavigationTask; global.ReconstructionTask=ReconstructionTask; global.TMazeTask=TMazeTask; global.Navigation2DTask=Navigation2DTask; global.IntervalTimingTask=IntervalTimingTask;`);

// ── helpers ────────────────────────────────────────────────────────────────
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
const std  = a => { if(a.length<=1) return 0; const m=mean(a); return Math.sqrt(a.reduce((s,v)=>s+Math.pow(v-m,2),0)/(a.length-1)); };
const ci95 = a => 1.96 * std(a) / Math.sqrt(a.length);

/** Apply feature mask to neuron AFTER setupNeuron() has been called */
function applyMask(neuron, mask) {
    // mask: 'full' | 'dend' | 'point'
    if (mask === 'full') {
        // respect whatever setupNeuron set
        return;
    }
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

const MASKS  = ['full','dend','point'];
const LABELS = { full:'NAD-Full', dend:'Dend-Only', point:'Point-LIF' };
const N = 100; // trials per condition

const results = {};
const t0 = Date.now();

// ══════════════════════════════════════════════════════════════════════════
// HELPER: run a generic task for `ticks` ticks, return final accuracy
// ══════════════════════════════════════════════════════════════════════════
function runTask(TaskClass, mask, ticks, extraSetup) {
    const accs = [];
    for (let t=0; t<N; t++) {
        const neuron = new AstroDendriticNeuron();
        const task   = new TaskClass();
        task.setupNeuron(neuron);
        applyMask(neuron, mask);
        if (extraSetup) extraSetup(task, neuron, t);
        for (let tick=0; tick<ticks; tick++) {
            const inp    = task.generateInputs(tick);
            const spiked = neuron.tick(inp, 1.0);
            task.evaluate(neuron, tick, spiked);
        }
        accs.push(task.accuracy);
    }
    return accs;
}

// ══════════════════════════════════════════════════════════════════════════
// 1. XOR – nonlinear computation (dendrites essential)
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 1. XOR (nonlinear gate) ━━━');
results.xor = {};
for (const mask of MASKS) {
    const accs = runTask(XORTask, mask, 8000);
    const m = mean(accs);
    results.xor[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%`);
}

// ══════════════════════════════════════════════════════════════════════════
// 2. SEQUENCE DETECTION – temporal order (dendrites + timing)
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 2. Sequence Detection (jitter=0) ━━━');
results.sequence = {};
for (const mask of MASKS) {
    const accs = runTask(SequenceTask, mask, 2000, (task)=>{ task.jitter=0; });
    const m = mean(accs);
    results.sequence[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%`);
}

// Jitter sweep for NAD vs Point-LIF
const jitters = [0,5,10,15,20,25];
results.sequence.jitter_sweep = { jitters, nad:[], point:[] };
for (const j of jitters) {
    const an = mean(runTask(SequenceTask,'full',2000,(task)=>{ task.jitter=j; }));
    const ap = mean(runTask(SequenceTask,'point',2000,(task)=>{ task.jitter=j; }));
    results.sequence.jitter_sweep.nad.push(an);
    results.sequence.jitter_sweep.point.push(ap);
}

// ══════════════════════════════════════════════════════════════════════════
// 3. NOISE ROBUSTNESS – astrocyte homeostasis
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 3. Noise Robustness (SNR sweep) ━━━');
const noiseRates = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
results.noise = { noise_rates: noiseRates, full:[], dend:[], point:[] };
for (const nr of noiseRates) {
    for (const mask of MASKS) {
        const accs = runTask(NoiseTask, mask, 800, (task)=>{ task.noiseRate=nr; });
        results.noise[mask].push({ mean: mean(accs), std: std(accs) });
    }
    console.log(`  noise=${(nr*100).toFixed(0)}%  NAD=${results.noise.full[results.noise.full.length-1].mean.toFixed(1)}%  Point=${results.noise.point[results.noise.point.length-1].mean.toFixed(1)}%`);
}

// ══════════════════════════════════════════════════════════════════════════
// 4. PAVLOVIAN CONDITIONING – astrocyte modulated plasticity
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 4. Pavlovian Conditioning ━━━');
results.conditioning = {};
for (const mask of MASKS) {
    const accs = runTask(ConditioningTask, mask, 2400);
    const m = mean(accs);
    results.conditioning[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%`);
}

// ══════════════════════════════════════════════════════════════════════════
// 5. FREQUENCY RESPONSE – plateau frequency tuning
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 5. Frequency Response Transfer Function ━━━');
const frequencies = [5,10,20,30,40,50,60,80,100];
results.frequency = { frequencies, full:[], point:[] };
for (const f of frequencies) {
    const interval = Math.round(1000/f);
    const rn = []; const rp = [];
    for (let t=0; t<N; t++) {
        for (const [mask, arr] of [['full',rn],['point',rp]]) {
            const neuron = new AstroDendriticNeuron();
            const task   = new FrequencyTask();
            task.setupNeuron(neuron);
            applyMask(neuron, mask);
            task.customInterval = interval;
            let spk=0;
            for (let tick=0; tick<200; tick++) { if(neuron.tick(task.generateInputs(tick),1.0)) spk++; }
            arr.push(spk*5.0);
        }
    }
    results.frequency.full.push({mean:mean(rn),std:std(rn)});
    results.frequency.point.push({mean:mean(rp),std:std(rp)});
    console.log(`  f=${f}Hz  NAD=${mean(rn).toFixed(1)}Hz  Point=${mean(rp).toFixed(1)}Hz`);
}

// ══════════════════════════════════════════════════════════════════════════
// 6. NEURO-PONG – RL control benchmark
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 6. Neuro-Pong RL ━━━');
results.pong = {};
for (const mask of MASKS) {
    const accs = runTask(PongTask, mask, 8000);
    const m = mean(accs);
    results.pong[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%`);
}

// ══════════════════════════════════════════════════════════════════════════
// 7. 1D GRID NAVIGATION – RL control
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 7. 1D Grid Navigation ━━━');
results.navigation_1d = {};
for (const mask of MASKS) {
    const accs = runTask(NavigationTask, mask, 5000);
    const m = mean(accs);
    results.navigation_1d[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%`);
}

// ══════════════════════════════════════════════════════════════════════════
// 8. 2D GRID NAVIGATION – multi-channel spatial integration
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 8. 2D Grid Navigation ━━━');
results.navigation_2d = {};
for (const mask of MASKS) {
    const accs = runTask(Navigation2DTask, mask, 8000);
    const m = mean(accs);
    results.navigation_2d[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%`);
}

// ══════════════════════════════════════════════════════════════════════════
// 9. T-MAZE WORKING MEMORY – LMU temporal memory (key ablation!)
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 9. T-Maze Working Memory (LMU ablation) ━━━');
results.tmaze = {};
for (const mask of MASKS) {
    const accs = runTask(TMazeTask, mask, 3600); // 30 trials × 120 ticks
    const m = mean(accs);
    results.tmaze[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%  (chance=50%)`);
}

// ══════════════════════════════════════════════════════════════════════════
// 10. INTERVAL TIMING – temporal delay reproduction (LMU ablation)
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 10. Interval Timing (delay=50 ticks, tol=±10) ━━━');
results.interval_timing = {};
for (const mask of MASKS) {
    const accs = runTask(IntervalTimingTask, mask, 3600);
    const m = mean(accs);
    results.interval_timing[mask] = { mean: m, std: std(accs), ci: ci95(accs), raw: accs };
    console.log(`  ${LABELS[mask].padEnd(12)}: ${m.toFixed(1)}% ± ${std(accs).toFixed(1)}%  (chance~8%)`);
}

// ══════════════════════════════════════════════════════════════════════════
// 11. LMU RECONSTRUCTION – Legendre delay memory (order sweep)
// ══════════════════════════════════════════════════════════════════════════
console.log('\n━━━ 11. LMU Delay Reconstruction ━━━');
const lmuOrders = [2,3,4,5,6];
results.lmu = { orders: lmuOrders, full:[], point:[] };
for (const ord of lmuOrders) {
    const mn = [], mp = [];
    for (let t=0; t<N; t++) {
        for (const [mask,arr] of [['full',mn],['point',mp]]) {
            const neuron = new AstroDendriticNeuron();
            const task   = new ReconstructionTask();
            task.lmuOrder = ord;
            task.setupNeuron(neuron);
            applyMask(neuron, mask);
            for (let tick=0; tick<task.patternDuration; tick++) {
                neuron.tick(task.generateInputs(tick),1.0);
                task.evaluate(neuron, tick, false);
            }
            arr.push(task.mse);
        }
    }
    results.lmu.full.push({mean:mean(mn),std:std(mn)});
    results.lmu.point.push({mean:mean(mp),std:std(mp)});
    console.log(`  d=${ord}  NAD MSE=${mean(mn).toFixed(5)}  Point MSE=${mean(mp).toFixed(5)}`);
}

// ══════════════════════════════════════════════════════════════════════════
// SUMMARY TABLE
// ══════════════════════════════════════════════════════════════════════════
console.log('\n\n══════════════════════════════════════════════════════════════════════');
console.log('  ABLATION SUMMARY  (NAD-Full vs Dend-Only vs Point-LIF)');
console.log('══════════════════════════════════════════════════════════════════════');
console.log('  Task                    NAD-Full    Dend-Only   Point-LIF   Key Feature');
console.log('  ──────────────────────────────────────────────────────────────────────');

function row(label, key, feat) {
    const r = results[key];
    const fn = (v,s) => `${v.toFixed(1)}±${s.toFixed(1)}`.padEnd(12);
    console.log(`  ${label.padEnd(24)}${fn(r.full.mean,r.full.std)}${fn(r.dend.mean,r.dend.std)}${fn(r.point.mean,r.point.std)}${feat}`);
}

row('XOR gate',             'xor',             'Dendrites');
row('Sequence detection',   'sequence',        'Dendrites');
row('Conditioning',         'conditioning',    'Astrocyte');
row('Pong RL',              'pong',            'All');
row('1D Navigation',        'navigation_1d',   'Dendrites (plateau)');
row('2D Navigation',        'navigation_2d',   'Dendrites (multi-ch)');
row('T-Maze memory',        'tmaze',           'LMU ← KEY');
row('Interval timing',      'interval_timing', 'LMU ← KEY');

// Noise row (special structure)
const noi_rates = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
const ni = 3; // index for 30% noise
const nf2 = results.noise.full[ni], nd2 = results.noise.dend[ni], np2 = results.noise.point[ni];
const fn2 = (o) => `${o.mean.toFixed(1)}±${o.std.toFixed(1)}`.padEnd(12);
console.log(`  ${'Noise @ 30% rate'.padEnd(24)}${fn2(nf2)}${fn2(nd2)}${fn2(np2)}Astrocyte`);

console.log('══════════════════════════════════════════════════════════════════════');
console.log(`\nTotal run time: ${((Date.now()-t0)/1000).toFixed(0)}s`);

// Write full results JSON for plotting
const out = path.join(__dirname,'experiment_results.json');
fs.writeFileSync(out, JSON.stringify(results, null, 2));
console.log(`Results → ${out}`);
