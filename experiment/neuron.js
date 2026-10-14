/**
 * Next-Generation Astro-Dendritic Spiking Neuron Engine
 * Implements:
 * 1. Soma (Leaky Integrate-and-Fire Dynamics)
 * 2. Active Dendrites (Non-linear branch summation + plateau potentials)
 * 3. Astrocytic Modulation (Calcium dynamics + gliotransmission gain control)
 * 4. Legendre Memory Unit (LMU) state-space delay integration
 * 5. Synaptic Plasticity (Spike-Timing-Dependent Plasticity - STDP)
 */

class Synapse {
    constructor(id, sourceChannel, targetBranchId, initialWeight = 1.0) {
        this.id = id;
        this.sourceChannel = sourceChannel;
        this.targetBranchId = targetBranchId;
        
        // Weight parameters
        this.weight = initialWeight;
        this.effectiveWeight = initialWeight;
        this.baseWeight = initialWeight; // Original weight before glial scaling
        
        // STDP Traces
        this.preTrace = 0.0;
        this.tauPre = 20.0; // Decay constant in ticks (ms)
        this.eligibility = 0.0;
        this.tauEligibility = 40.0;
        
        // LMU readout modulation
        this.readoutDelay = null; // optional delay in ticks (ms)
    }

    /**
     * Update STDP traces and eligibility trace
     */
    update(preSpikeOccurred, timeStep) {
        // Pre-synaptic trace decay
        this.preTrace -= (this.preTrace / this.preTraceDecay(timeStep));
        if (this.preTrace < 0.01) this.preTrace = 0;

        // Eligibility trace decay
        this.eligibility -= (this.eligibility / this.eligibilityDecay(timeStep));
        if (this.eligibility < 0.01) this.eligibility = 0;

        if (preSpikeOccurred) {
            this.preTrace = 1.0;
        }
    }

    preTraceDecay(dt) { return this.tauPre / dt; }
    eligibilityDecay(dt) { return this.tauEligibility / dt; }
}

class DendriticBranch {
    constructor(id, threshold = 8.0, plateauDuration = 80) {
        this.id = id;
        this.threshold = threshold;
        this.plateauDuration = plateauDuration; // in steps/ticks
        
        this.potential = 0.0;
        this.tauDend = 15.0; // Dendritic integration time constant
        this.isPlateau = false;
        this.plateauTimer = 0;
        
        // LMU (Legendre Memory Unit) configuration
        this.lmuOrder = 4; // Number of Legendre coefficients
        this.lmuTheta = 100.0; // History delay window size
        this.lmuState = new Float32Array(this.lmuOrder); // State vector h(t)
        this.A = null; // State transition matrix
        this.B = null; // Input mapping vector
        this.initializeLMUMatrices();
    }

    /**
     * Compute A and B matrices for Legendre Memory Unit delay line representation
     */
    initializeLMUMatrices() {
        const d = this.lmuOrder;
        const theta = this.lmuTheta;
        this.A = Array.from({ length: d }, () => new Float32Array(d));
        this.B = new Float32Array(d);
        this.lmuState = new Float32Array(d);

        for (let i = 0; i < d; i++) {
            // Fill B vector: B_i = (2i + 1)/theta * (-1)^i
            this.B[i] = ((2 * i + 1) / theta) * Math.pow(-1, i);
            
            for (let j = 0; j < d; j++) {
                // Fill A matrix:
                // A_ij = (2i + 1)/theta * (-1 if i < j else (-1)^(i-j+1))
                if (i < j) {
                    this.A[i][j] = -((2 * i + 1) / theta);
                } else {
                    this.A[i][j] = ((2 * i + 1) / theta) * Math.pow(-1, i - j + 1);
                }
            }
        }
    }

    /**
     * Update Dendritic integration dynamics
     * @returns {number} Somatic current injection from this branch
     */
    update(synapses, inputSpikes, timeStep, useDendrites, useLMU) {
        // Find synapses targeting this branch
        const branchSynapses = synapses.filter(s => s.targetBranchId === this.id);
        
        // Compute input current
        let synapticCurrent = 0;
        let lmuInputCurrent = 0;
        branchSynapses.forEach(syn => {
            const val = inputSpikes[syn.sourceChannel];
            const hasSpike = typeof val === 'number' ? val : (val ? 1.0 : 0.0);
            
            let inputContribution = syn.effectiveWeight * hasSpike;
            if (syn.readoutDelay !== null && syn.readoutDelay !== undefined) {
                // Fair ablation: if useLMU is false, do not modulate by LMU (so gate acts as normal synapse)
                if (useLMU) {
                    const reconVal = this.reconstructDelay(syn.readoutDelay);
                    const modFactor = Math.max(0.0, Math.min(3.0, reconVal));
                    inputContribution *= modFactor;
                }
            } else {
                // Only non-readout synapses are recorded in LMU memory
                lmuInputCurrent += syn.effectiveWeight * hasSpike;
            }
            synapticCurrent += inputContribution;
        });

        // 1. LMU State Space Continuous Delay integration
        if (useLMU) {
            // x(t) is the average non-readout synaptic input
            const inputVal = lmuInputCurrent;
            const subSteps = 10;
            const dt = timeStep / subSteps;
            for (let step = 0; step < subSteps; step++) {
                const nextState = new Float32Array(this.lmuOrder);
                for (let i = 0; i < this.lmuOrder; i++) {
                    let axSum = 0;
                    for (let j = 0; j < this.lmuOrder; j++) {
                        axSum += this.A[i][j] * this.lmuState[j];
                    }
                    nextState[i] = this.lmuState[i] + dt * (axSum + this.B[i] * inputVal);
                }
                this.lmuState = nextState;
            }
        } else {
            this.lmuState.fill(0);
        }

        // 2. Active Dendrite Plateau Potential integration
        if (useDendrites) {
            if (this.isPlateau) {
                this.plateauTimer -= timeStep;
                if (this.plateauTimer <= 0) {
                    this.isPlateau = false;
                    this.potential = 0.0; // Reset potential after plateau ends
                }
                return 3.0; // Injection current to soma during active plateau
            } else {
                // Leaky integration of dendritic membrane potential
                this.potential += timeStep * (-this.potential / this.tauDend + synapticCurrent);
                
                // Active threshold check: triggers plateau potential
                if (this.potential >= this.threshold) {
                    this.isPlateau = true;
                    this.plateauTimer = this.plateauDuration;
                    return 3.0;
                }
                // Passive dendritic current flow to soma
                return this.potential * 0.15; 
            }
        } else {
            // Fallback to simple passive linear summation bypass
            this.isPlateau = false;
            return synapticCurrent;
        }
    }

    reset() {
        this.potential = 0.0;
        this.isPlateau = false;
        this.plateauTimer = 0;
        this.lmuState.fill(0);
    }

    /**
     * Reconstruct the input value at a past delay tau using Legendre polynomials
     * @param {number} tau past delay time in ticks (0 <= tau <= lmuTheta)
     * @returns {number} reconstructed input value
     */
    reconstructDelay(tau) {
        const theta = this.lmuTheta;
        const x = 1.0 - (2.0 * tau) / theta; // Normalized time in [-1, 1]
        
        // Evaluate Legendre polynomials up to lmuOrder at x using recurrence relation
        const P = new Float32Array(this.lmuOrder);
        P[0] = 1.0;
        if (this.lmuOrder > 1) {
            P[1] = x;
            for (let n = 1; n < this.lmuOrder - 1; n++) {
                P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1);
            }
        }
        
        let sum = 0.0;
        for (let i = 0; i < this.lmuOrder; i++) {
            sum += Math.pow(-1, i) * P[i] * this.lmuState[i];
        }
        return sum;
    }
}

class Astrocyte {
    constructor(alpha = 0.08, decayRate = 0.005, gainStrength = 0.40) {
        this.alpha = alpha; // Calcium accumulation rate
        this.decayRate = decayRate; // Calcium clearance rate
        this.gainStrength = gainStrength; // Feedback regulation factor
        
        this.calcium = 0.0;
        this.gliotransmitter = 0.0;
        this.gainFactor = 1.0;
        
        // Target calcium level for homoeostasis
        this.calciumTarget = 0.5;
    }

    /**
     * Update astrocyte calcium levels and synaptic gain control
     */
    update(totalSynapticInput, somaSpiked, timeStep, useAstrocyte) {
        if (!useAstrocyte) {
            this.calcium = 0;
            this.gliotransmitter = 0;
            this.gainFactor = 1.0;
            return;
        }

        // Calcium accumulation driven by presynaptic activity and backpropagating somatic spikes
        const somaticFeedback = somaSpiked ? 0.3 : 0.0;
        const synapticDrive = totalSynapticInput * 0.02;

        this.calcium += timeStep * (
            this.alpha * (synapticDrive + somaticFeedback) - this.decayRate * this.calcium
        );
        this.calcium = Math.max(0, this.calcium);

        // Gliotransmission logic: releases ATP/Adenosine to dynamically scale synapses
        // Homeostatic response: if calcium exceeds the target, damp down synaptic gain
        const error = this.calcium - this.calciumTarget;
        
        // Non-linear gain modification
        const targetGain = 1.0 - this.gainStrength * Math.tanh(error * 1.5);
        
        // Low-pass filter the gain factor to model slow chemical diffusion
        this.gainFactor += 0.05 * (targetGain - this.gainFactor);
        this.gainFactor = Math.max(0.1, Math.min(2.0, this.gainFactor)); // Clamp bounds
    }

    reset() {
        this.calcium = 0.0;
        this.gliotransmitter = 0.0;
        this.gainFactor = 1.0;
    }
}

class Soma {
    constructor(threshold = 15.0, timeConstant = 10.0, resetPotential = 0.0) {
        this.threshold = threshold;
        this.tauM = timeConstant; // membrane time constant
        this.resetPotential = resetPotential;
        
        this.v = 0.0;
        this.refractoryTimer = 0;
        this.postTrace = 0.0; // Post-synaptic STDP trace
        this.tauPost = 20.0;
    }

    /**
     * Update somatic voltage integration
     * @returns {boolean} Whether the soma fired a spike (Action Potential)
     */
    update(currentInput, timeStep) {
        // Post-synaptic trace decay
        this.postTrace -= (this.postTrace / (this.tauPost / timeStep));
        if (this.postTrace < 0.01) this.postTrace = 0;

        if (this.refractoryTimer > 0) {
            this.refractoryTimer -= timeStep;
            this.v = this.resetPotential;
            return false;
        }

        // Leaky integration of somatic membrane potential
        // dv/dt = (-v + I) / tauM
        this.v += timeStep * ((-this.v + currentInput * 5.0) / this.tauM);
        this.v = Math.max(0, this.v); // Clamp hyperpolarization

        if (this.v >= this.threshold) {
            this.v = this.resetPotential;
            this.refractoryTimer = 4; // Refractory period in steps (ms)
            this.postTrace = 1.0; // Spike event boosts post trace
            return true;
        }

        return false;
    }

    reset() {
        this.v = 0.0;
        this.refractoryTimer = 0;
        this.postTrace = 0.0;
    }
}

class AstroDendriticNeuron {
    constructor(numChannels = 4) {
        this.numChannels = numChannels;
        
        // Create 2 Dendritic Branches
        this.branches = [
            new DendriticBranch(0, 8.0, 80),  // Branch A
            new DendriticBranch(1, 8.0, 80)   // Branch B
        ];

        // Create Soma
        this.soma = new Soma(15.0, 10.0, 0.0);

        // Create Astrocyte
        this.astrocyte = new Astrocyte(0.08, 0.005, 0.40);

        // Create 4 Synapses connecting input channels to specific dendritic branches
        // Channels 0 & 1 target Branch A. Channels 2 & 3 target Branch B.
        this.synapses = [
            new Synapse(0, 0, 0, 1.5),
            new Synapse(1, 1, 0, 1.5),
            new Synapse(2, 2, 1, 1.5),
            new Synapse(3, 3, 1, 1.5)
        ];

        // Global learning settings
        this.learningRate = 0.05;

        // Model feature toggles
        this.features = {
            dendrites: true,
            astrocyte: true,
            lmu: true
        };
    }

    /**
     * Processes one time-step of simulation
     * @param {Array<boolean>} inputSpikes Array of presynaptic spike flags for each input channel
     * @returns {boolean} Whether the neuron generated a somatic output spike
     */
    tick(inputSpikes, timeStep = 1.0) {
        // 1. Update Synaptic STDP Traces (presynaptic side)
        this.synapses.forEach(syn => {
            const preSpike = inputSpikes[syn.sourceChannel];
            syn.update(preSpike, timeStep);
            
            // Apply glial scaling to get effective weight
            syn.effectiveWeight = syn.weight * this.astrocyte.gainFactor;
        });

        // 2. Update Dendritic Branches and compute total somatic input current
        let totalSomaticCurrent = 0;
        let totalSynapticDrive = 0;

        this.branches.forEach(branch => {
            const branchCurrent = branch.update(
                this.synapses, 
                inputSpikes, 
                timeStep, 
                this.features.dendrites,
                this.features.lmu
            );
            
            totalSomaticCurrent += branchCurrent;
            
            // Accumulate active input count for the astrocyte
            this.synapses.filter(s => s.targetBranchId === branch.id).forEach(s => {
                if (inputSpikes[s.sourceChannel]) totalSynapticDrive += 1.0;
            });
        });

        // 3. Update Soma (integrate currents and check for output spike)
        const somaSpiked = this.soma.update(totalSomaticCurrent, timeStep);

        // 4. Update Astrocyte (calcium levels and gain regulation)
        this.astrocyte.update(totalSynapticDrive, somaSpiked, timeStep, this.features.astrocyte);

        // 5. Update Hebbian Eligibility Traces (depolarization correlation)
        this.synapses.forEach(syn => {
            const branch = this.branches[syn.targetBranchId];
            // If Point-LIF, use soma threshold/potential, otherwise dendritic potential
            const branchPlateau = this.features.dendrites ? (branch.isPlateau ? 8.0 : 0.0) : 0.0;
            const localDepol = (this.features.dendrites ? branch.potential : this.soma.v) + branchPlateau + (somaSpiked ? 3.0 : 0.0);
            
            if (localDepol > 0.1) {
                syn.eligibility += 0.05 * syn.preTrace * localDepol;
            }
        });

        // 6. Apply Learning Rule (STDP with Astrocytic Gating)
        if (somaSpiked && this.learningRate > 0) {
            // Post-synaptic spike occurred: strengthen synapses active recently (LTP)
            this.synapses.forEach(syn => {
                const plasticityGate = Math.max(0.1, 1.0 - this.astrocyte.calcium * 0.5);
                const deltaW = this.learningRate * syn.preTrace * plasticityGate;
                syn.weight = Math.min(5.0, syn.weight + deltaW);
            });
        }

        // Pre-synaptic spike occurred: if soma did not fire, weaken (LTD)
        if (this.learningRate > 0) {
            this.synapses.forEach(syn => {
                if (inputSpikes[syn.sourceChannel] && !somaSpiked) {
                    const postTrace = this.soma.postTrace;
                    if (postTrace > 0.05) {
                        const plasticityGate = Math.max(0.1, 1.0 - this.astrocyte.calcium * 0.5);
                        const deltaW = -this.learningRate * postTrace * 0.6 * plasticityGate;
                        syn.weight = Math.max(0.1, syn.weight + deltaW);
                    }
                }
            });
        }

        return somaSpiked;
    }

    /**
     * Apply dopamine reward feedback to synapses using eligibility traces (R-STDP three-factor rule)
     * With branch-specific competitive learning to break routing symmetry.
     * @param {number} reward dopamine reward (positive) or punishment (negative)
     * @param {number} learningRate specific learning rate for reinforcement
     */
    applyDopamineReward(reward, learningRate = 0.05) {
        // Find which branch was active (dominant potential or plateaued)
        let dominantBranchId = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < this.branches.length; i++) {
            const val = this.branches[i].potential + (this.branches[i].isPlateau ? 8.0 : 0.0);
            if (val > maxVal) {
                maxVal = val;
                dominantBranchId = i;
            }
        }

        this.synapses.forEach(syn => {
            const plasticityGate = Math.max(0.1, 1.0 - this.astrocyte.calcium * 0.5);
            let effectiveReward = reward;
            // Competitive learning: if synapse is not on dominant branch, depress it (or invert reward)
            if (syn.targetBranchId !== dominantBranchId) {
                effectiveReward = -reward * 0.5;
            }
            const deltaW = learningRate * syn.eligibility * effectiveReward * plasticityGate;
            syn.weight = Math.max(0.05, Math.min(5.0, syn.weight + deltaW));
        });
    }

    reset() {
        this.soma.reset();
        this.astrocyte.reset();
        this.branches.forEach(b => b.reset());
        this.synapses.forEach(syn => {
            syn.weight = syn.baseWeight;
            syn.effectiveWeight = syn.baseWeight;
            syn.preTrace = 0;
            syn.eligibility = 0;
        });
    }

    resetStates() {
        this.soma.reset();
        this.astrocyte.reset();
        this.branches.forEach(b => b.reset());
        this.synapses.forEach(syn => {
            syn.preTrace = 0;
            syn.eligibility = 0;
        });
    }
}
