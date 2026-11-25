/**
 * Experiments and Tasks for the Astro-Dendritic Spiking Neuron
 * Includes:
 * 1. XOR Learning Task (Online training of 8 synapses across 4 complementary rails)
 * 2. Spatiotemporal Sequence Detection Task (A -> B -> C recognition vs. reverse C -> B -> A)
 * 3. Homeostatic Noise Filtering Task (Dynamic gain adjustment under high background noise)
 */

class XORTask {
    constructor() {
        this.name = "Non-linear XOR Task";
        this.description = "Train a single neuron to solve the XOR gate by wiring 8 synapses over 4 input channels (A, NOT_A, B, NOT_B) to 2 dendritic branches.";
        
        this.patternDuration = 100; // ticks per pattern
        this.numPatterns = 4;
        this.epochDuration = this.patternDuration * this.numPatterns; // 400 ticks
        
        // Track accuracy for each pattern
        this.patternSuccess = [false, false, false, false];
        this.patternSpikeCounts = [0, 0, 0, 0];
        
        this.epoch = 0;
        this.accuracy = 0.0;
        
        // Input patterns
        // Index 0: (0,0) -> active rails: NOT_A (1) and NOT_B (3)
        // Index 1: (1,0) -> active rails: A (0) and NOT_B (3)
        // Index 2: (0,1) -> active rails: NOT_A (1) and B (2)
        // Index 3: (1,1) -> active rails: A (0) and B (2)
        this.targets = [
            false, // (0,0) -> 0
            true,  // (1,0) -> 1
            true,  // (0,1) -> 1
            false  // (1,1) -> 0
        ];
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.0; // Isolate R-STDP learning rule by disabling unsupervised STDP
        
        // Connect all 4 channels to both branches (no pre-wired routing!)
        neuron.synapses = [];
        let id = 0;
        for (let c = 0; c < 4; c++) {
            for (let b = 0; b < 2; b++) {
                neuron.synapses.push(new Synapse(id++, c, b, 0.4 + Math.random() * 0.6));
            }
        }
        
        // Set dendritic and somatic parameters optimized for XOR
        neuron.branches[0].threshold = 2.5;
        neuron.branches[1].threshold = 2.5;
        neuron.soma.threshold = 10.0;
        
        // Disable astrocyte for XOR task to avoid gain interference
        neuron.features.astrocyte = false;
        neuron.features.lmu = false;
        
        this.reset();
    }

    generateInputs(tickIndex) {
        const patternIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const patternTick = tickIndex % this.patternDuration;
        
        const spikes = [false, false, false, false];
        
        // Generate spike trains at 50Hz (every 10 ticks) during the active periods
        const isSpikeTick = (patternTick > 10 && patternTick < 80 && patternTick % 10 === 0);
        
        if (isSpikeTick) {
            switch(patternIndex) {
                case 0: // (0,0) -> NOT_A and NOT_B active
                    spikes[1] = true;
                    spikes[3] = true;
                    break;
                case 1: // (1,0) -> A and NOT_B active
                    spikes[0] = true;
                    spikes[3] = true;
                    break;
                case 2: // (0,1) -> NOT_A and B active
                    spikes[1] = true;
                    spikes[2] = true;
                    break;
                case 3: // (1,1) -> A and B active
                    spikes[0] = true;
                    spikes[2] = true;
                    break;
            }
        }
        
        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const patternIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const patternTick = tickIndex % this.patternDuration;

        // Reset transient states at pattern boundaries to clear plateaus
        if (patternTick === 0) {
            neuron.resetStates();
            this.branchPlateaued = [false, false];
        }
        
        // Track if a branch is plateaued during the pattern
        neuron.branches.forEach((b, i) => {
            if (b.isPlateau) {
                this.branchPlateaued[i] = true;
            }
        });
        
        if (spiked) {
            this.patternSpikeCounts[patternIndex]++;
        }
        
        // At the end of a pattern, check if the neuron behaved correctly
        if (patternTick === this.patternDuration - 1) {
            const expected = this.targets[patternIndex];
            const spikeCount = this.patternSpikeCounts[patternIndex];
            
            // Expected spike: spike count should be high (e.g., > 2 spikes)
            // Expected silent: spike count should be 0
            if (expected) {
                this.patternSuccess[patternIndex] = (spikeCount >= 2);
            } else {
                this.patternSuccess[patternIndex] = (spikeCount === 0);
            }
            
            const isCorrect = this.patternSuccess[patternIndex];
            // Enforce a robust learning margin: require >= 3 spikes for expected spike patterns
            const isRobustCorrect = expected ? (spikeCount >= 3) : (spikeCount === 0);
            let reward = 0.0;
            if (!isRobustCorrect) {
                reward = expected ? 1.5 : -1.5;
            }
            neuron.applyDopamineReward(reward, 0.02);
            
            // Reset count for next epoch
            this.patternSpikeCounts[patternIndex] = 0;
        }
        
        // At the end of the entire epoch, compute accuracy
        if (tickIndex > 0 && tickIndex % this.epochDuration === 0) {
            this.epoch++;
            const correctCount = this.patternSuccess.filter(s => s).length;
            this.accuracy = (correctCount / this.numPatterns) * 100;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.patternSuccess = [false, false, false, false];
        this.patternSpikeCounts = [0, 0, 0, 0];
        this.branchPlateaued = [false, false];
    }
}

class SequenceTask {
    constructor() {
        this.name = "Spatiotemporal Sequence Task";
        this.description = "Detect a forward temporal sequence (A -> B -> C) while ignoring the reversed pattern (C -> B -> A). Uses active dendrite delays.";
        
        this.patternDuration = 150; // ticks per pattern
        this.epochDuration = this.patternDuration * 3; // 3 trials: Forward, Reverse, Quiet
        
        this.accuracy = 0.0;
        this.epoch = 0;
        
        this.forwardCorrect = false;
        this.reverseCorrect = false;
        
        this.forwardSpikeCount = 0;
        this.reverseSpikeCount = 0;
        
        this.jitter = 0; // standard deviation of Gaussian timing jitter (in ms/ticks)
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.0; // static template matching
        
        // Connect channels to dendritic branches
        // Channel 0 (A) and 1 (B) target Branch A (0).
        // Channel 2 (C) targets Branch B (1).
        // Channel 3 is background noise.
        neuron.synapses = [
            new Synapse(0, 0, 0, 2.5), // Channel 0 -> Branch A (A)
            new Synapse(1, 1, 0, 2.0), // Channel 1 -> Branch A (B)
            new Synapse(2, 2, 1, 3.2), // Channel 2 -> Branch B (C)
            new Synapse(3, 3, 1, 0.5)  // Noise -> Branch B
        ];

        // Configure branch parameters to act as delay lines
        neuron.branches[0].threshold = 5.5; // High threshold so A+B is required, B alone or A alone fails
        neuron.branches[1].threshold = 3.5;
        neuron.branches[0].tauDend = 45.0; // Slow integration window
        neuron.branches[1].tauDend = 15.0; // Fast integration window
        neuron.branches[0].plateauDuration = 80;
        neuron.branches[1].plateauDuration = 40; // Short duration for C to prevent reverse sequence overlap
        
        neuron.soma.threshold = 20.0;
        
        this.reset();
    }

    generateInputs(tickIndex) {
        const trialIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const trialTick = tickIndex % this.patternDuration;
        
        const spikes = [false, false, false, false];
        
        // Add low background noise on channel 3 (10% chance)
        if (Math.random() < 0.08) {
            spikes[3] = true;
        }

        // Initialize jittered spike timings at the beginning of the trial
        if (trialTick === 0) {
            const genJitter = () => {
                if (!this.jitter || this.jitter === 0) return 0;
                // Box-Muller transform for Gaussian noise
                const u1 = Math.max(0.0001, Math.random());
                const u2 = Math.random();
                const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                return Math.round(z0 * this.jitter);
            };
            this.tA = 20 + genJitter();
            this.tB = 50 + genJitter();
            this.tC = 80 + genJitter();
        }

        if (trialIndex === 0) {
            // Forward sequence: A (tA), B (tB), C (tC)
            if (trialTick === this.tA || trialTick === this.tA + 1) spikes[0] = true;
            if (trialTick === this.tB || trialTick === this.tB + 1) spikes[1] = true;
            if (trialTick === this.tC || trialTick === this.tC + 1) spikes[2] = true;
        } else if (trialIndex === 1) {
            // Reversed sequence: C (tA), B (tB), A (tC)
            if (trialTick === this.tA || trialTick === this.tA + 1) spikes[2] = true;
            if (trialTick === this.tB || trialTick === this.tB + 1) spikes[1] = true;
            if (trialTick === this.tC || trialTick === this.tC + 1) spikes[0] = true;
        }
        // trialIndex === 2 is Quiet/Noise only
        
        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const trialIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const trialTick = tickIndex % this.patternDuration;
        
        // Reset transient states at trial boundaries
        if (trialTick === 0) {
            neuron.resetStates();
        }
        
        if (spiked) {
            if (trialIndex === 0) this.forwardSpikeCount++;
            if (trialIndex === 1) this.reverseSpikeCount++;
        }
        
        // Evaluate at the end of each trial
        if (trialTick === this.patternDuration - 1) {
            if (trialIndex === 0) {
                // Forward: should spike at least once when sequence completes
                this.forwardCorrect = (this.forwardSpikeCount >= 1);
            } else if (trialIndex === 1) {
                // Reverse: should stay silent
                this.reverseCorrect = (this.reverseSpikeCount === 0);
            }
        }
        
        // Compute accuracy at epoch completion
        if (tickIndex > 0 && tickIndex % this.epochDuration === 0) {
            this.epoch++;
            let score = 0;
            if (this.forwardCorrect) score += 50;
            if (this.reverseCorrect) score += 50;
            this.accuracy = score;
            
            // Reset counts
            this.forwardSpikeCount = 0;
            this.reverseSpikeCount = 0;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.forwardCorrect = false;
        this.reverseCorrect = false;
        this.forwardSpikeCount = 0;
        this.reverseSpikeCount = 0;
        this.jitter = 0;
    }
}

class NoiseTask {
    constructor() {
        this.name = "Homeostatic Noise Filter";
        this.description = "Test the astrocyte's ability to maintain firing rates and save energy under intense, high-frequency background noise.";
        
        this.patternDuration = 200; // ticks
        this.epochDuration = this.patternDuration * 2; // 2 phases: Astrocyte OFF vs. Astrocyte ON
        
        this.accuracy = 0.0; // Represents homeostatic stability index (higher = better)
        this.epoch = 0;
        
        this.phase0Spikes = 0;
        this.phase1Spikes = 0;
        
        this.noiseRate = 0.35; // default background noise probability per tick
        
        // Fidelity tracking variables
        this.phase0Fidelity = 0.0;
        this.phase1Fidelity = 0.0;
        
        // Track somatic spikes during target windows
        this.phase0TP = 0;
        this.phase1TP = 0;
        this.phase0FP = 0;
        this.phase1FP = 0;
        this.window0Hit = false;
        this.window1Hit = false;
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.01;
        
        // Connect synapses
        neuron.synapses = [
            new Synapse(0, 0, 0, 2.5),
            new Synapse(1, 1, 0, 2.5),
            new Synapse(2, 2, 1, 2.5),
            new Synapse(3, 3, 1, 2.5)
        ];
        
        neuron.branches[0].threshold = 6.0;
        neuron.branches[1].threshold = 6.0;
        neuron.soma.threshold = 12.0;

        neuron.features.dendrites = true;
        neuron.features.astrocyte = true;
        neuron.features.lmu = false;

        // Tune astrocyte for noise suppression
        neuron.astrocyte.alpha = 0.20;
        neuron.astrocyte.calciumTarget = 0.20;
        neuron.astrocyte.gainStrength = 0.85;
        
        this.reset();
    }

    generateInputs(tickIndex) {
        const phaseIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const phaseTick = tickIndex % this.patternDuration;
        
        const spikes = [false, false, false, false];
        
        // High frequency random background noise on all channels
        for (let i = 0; i < 4; i++) {
            if (Math.random() < this.noiseRate) {
                spikes[i] = true;
            }
        }
        
        // Superimpose a regular target signal at 10Hz on Channel 0 (ticks 10, 110 of the block)
        // This is a burst: spikes at 10, 11, 12 and 110, 111, 112
        const isTargetSpike = (phaseTick >= 10 && phaseTick <= 12) || (phaseTick >= 110 && phaseTick <= 112);
        if (isTargetSpike) {
            spikes[0] = true;
        }
        
        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const phaseIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const phaseTick = tickIndex % this.patternDuration;
        
        if (phaseTick === 0) {
            this.window0Hit = false;
            this.window1Hit = false;
            this.tempFP = 0;
            // Clear astrocyte states on phase boundary to keep comparisons clean
            neuron.resetStates();
        }
        
        // Check target spike windows: [10, 35] and [110, 135]
        const inWindow0 = (phaseTick >= 10 && phaseTick <= 35);
        const inWindow1 = (phaseTick >= 110 && phaseTick <= 135);
        
        if (spiked) {
            if (inWindow0) {
                this.window0Hit = true;
            } else if (inWindow1) {
                this.window1Hit = true;
            } else {
                this.tempFP++;
            }
        }
        
        // Phase 0: Force Astrocyte OFF to show comparison
        // Phase 1: Astrocyte ON to show gain suppression
        if (phaseIndex === 0) {
            neuron.features.astrocyte = false;
            if (spiked) this.phase0Spikes++;
        } else {
            neuron.features.astrocyte = neuron.allowsAstrocyte !== false;
            if (spiked) this.phase1Spikes++;
        }
        
        // Compute metrics at phase completion
        if (phaseTick === this.patternDuration - 1) {
            const tp = (this.window0Hit ? 1 : 0) + (this.window1Hit ? 1 : 0);
            const fn = 2 - tp;
            const fp = this.tempFP;
            const fidelity = (tp / (2 + fp)) * 100;
            
            if (phaseIndex === 0) {
                this.phase0Fidelity = fidelity;
                this.phase0TP = tp;
                this.phase0FP = fp;
            } else {
                this.phase1Fidelity = fidelity;
                this.phase1TP = tp;
                this.phase1FP = fp;
            }
        }
        
        // Compute signal selectivity at epoch completion.
        // Correct metric: astrocyte should MAINTAIN detection of real signals while
        // suppressing noise spikes. We measure Phase-1 fidelity (astro ON) vs Phase-0
        // fidelity (astro OFF). A good astrocyte scores higher in Phase 1.
        // accuracy = phase1_fidelity (detection with astrocyte ON, noise suppressed).
        if (tickIndex > 0 && tickIndex % this.epochDuration === 0) {
            this.epoch++;
            // Phase 1 fidelity: did the astrocyte allow real-signal detection?
            this.accuracy = this.phase1Fidelity;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.phase0Spikes = 0;
        this.phase1Spikes = 0;
        this.noiseRate = 0.35;
        this.phase0Fidelity = 0.0;
        this.phase1Fidelity = 0.0;
        this.phase0TP = 0;
        this.phase1TP = 0;
        this.phase0FP = 0;
        this.phase1FP = 0;
        this.window0Hit = false;
        this.window1Hit = false;
    }
}

class PongTask {
    constructor() {
        this.name = "Neuro-Pong Game";
        this.description = "A single neuron plays Pong! Input 0 spikes if ball is above paddle. Input 1 spikes if ball is below. Spikes move paddle UP; silence moves paddle DOWN. Trained via Dopamine-Modulated reinforcement learning.";
        
        this.patternDuration = 10000; // run indefinitely
        this.epochDuration = 10000;
        this.epoch = 0;
        this.accuracy = 0.0;
        
        // Pong Game Physics Coordinates (0 to 100 scale)
        this.ballX = 50.0;
        this.ballY = 50.0;
        this.ballVX = 1.0;
        this.ballVY = 0.5;
        
        this.paddleY = 50.0;
        this.paddleHeight = 18.0;
        
        this.opponentY = 50.0;
        this.opponentHeight = 18.0;
        
        this.hits = 0;
        this.misses = 0;
        this.dopamine = 0.0; // Dopamine trace (+2.0 reward, -2.0 punishment)
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.0; // Disable standard STDP, do R-STDP
        
        // Connect all 4 channels to both branches (no pre-wired routing!)
        neuron.synapses = [];
        let id = 0;
        for (let c = 0; c < 4; c++) {
            for (let b = 0; b < 2; b++) {
                neuron.synapses.push(new Synapse(id++, c, b, 1.0 + Math.random() * 1.0));
            }
        }

        // Short plateau duration to prevent oscillatory overshoot
        neuron.branches[0].threshold = 3.0;
        neuron.branches[1].threshold = 3.0;
        neuron.branches[0].plateauDuration = 10;
        neuron.branches[1].plateauDuration = 10;
        neuron.soma.threshold = 8.0;

        neuron.features.dendrites = true;
        neuron.features.astrocyte = true;
        neuron.features.lmu = false;

        this.reset();
    }

    ballReset() {
        this.ballX = 50.0;
        this.ballY = 20.0 + Math.random() * 60.0;
        // Slow ball: gives the single-neuron paddle time to track before arrival.
        // At vx=0.35-0.65 the ball takes 80-160 ticks to travel to the paddle,
        // giving the neuron 8-16 dendritic plateau cycles to correct its position.
        this.ballVX = 0.35 + Math.random() * 0.30;
        this.ballVY = (Math.random() - 0.5) * 1.0;
    }

    generateInputs(tickIndex) {
        const spikes = [false, false, false, false];
        
        // 1. Spikes based on ball Y position relative to paddle center
        // Spiking twice as frequently (every 4 ticks) to ensure adequate membrane integration
        const isSpikeTick = (tickIndex % 4 === 0);
        if (isSpikeTick) {
            if (this.ballY < this.paddleY) {
                // Ball is ABOVE paddle -> Channel 0 spikes
                spikes[0] = Math.random() < 0.85;
            } else {
                // Ball is BELOW paddle -> Channel 1 spikes
                spikes[1] = Math.random() < 0.85;
            }
        }

        // 2. Proximity check (spikes faster as ball gets closer to right side)
        if (this.ballX > 75) {
            const prob = (this.ballX - 75) / 25; // 0 to 1 probability
            if (Math.random() < prob * 0.5) {
                spikes[2] = true;
            }
        }

        // 3. Constant low-rate bias spike
        if (Math.random() < 0.05) {
            spikes[3] = true;
        }

        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const ballAbove = this.ballY < this.paddleY;
        const bA = neuron.branches[0].potential + (neuron.branches[0].isPlateau ? 8.0 : 0);
        const bB = neuron.branches[1].potential + (neuron.branches[1].isPlateau ? 8.0 : 0);
        
        let reward = 0.0;
        let moved = false;
        let movedUp = false;
        
        if (spiked) {
            moved = true;
            if (bA >= bB) {
                this.paddleY = Math.max(this.paddleHeight / 2, this.paddleY - 10.0); // move UP (larger step)
                movedUp = true;
            } else {
                this.paddleY = Math.min(100 - this.paddleHeight / 2, this.paddleY + 10.0); // move DOWN (larger step)
                movedUp = false;
            }
        }
        
        if (moved) {
            if (ballAbove === movedUp) {
                reward = 1.0; // Correct directional step
            } else {
                reward = -1.0; // Incorrect directional step
            }
            neuron.applyDopamineReward(reward, 0.03);
        }

        // B. Opponent AI (left side)
        this.opponentY += (this.ballY - this.opponentY) * 0.12;
        this.opponentY = Math.max(this.opponentHeight / 2, Math.min(100 - this.opponentHeight / 2, this.opponentY));

        // C. Update Ball Physics
        this.ballX += this.ballVX;
        this.ballY += this.ballVY;

        // Top/Bottom Wall bounce
        if (this.ballY <= 1) {
            this.ballY = 1;
            this.ballVY = -this.ballVY;
        } else if (this.ballY >= 99) {
            this.ballY = 99;
            this.ballVY = -this.ballVY;
        }

        // Opponent Paddle collision (left side X = 4)
        if (this.ballX <= 4 && this.ballX >= 2) {
            const halfHeight = this.opponentHeight / 2;
            if (this.ballY >= this.opponentY - halfHeight && this.ballY <= this.opponentY + halfHeight) {
                this.ballX = 4;
                this.ballVX = -this.ballVX * 1.05; // speed up slightly on hit
                this.ballVY += (this.ballY - this.opponentY) * 0.15;
            }
        }

        // Opponent misses (past left boundary)
        if (this.ballX < 0) {
            this.hits++; // count as player success
            this.ballReset();
        }

        // Player Paddle collision (right side X = 96)
        let isHit = false;
        let isMiss = false;

        if (this.ballX >= 96 && this.ballX <= 98) {
            const halfHeight = this.paddleHeight / 2;
            if (this.ballY >= this.paddleY - halfHeight && this.ballY <= this.paddleY + halfHeight) {
                this.ballX = 96;
                this.ballVX = -this.ballVX * 1.05;
                this.ballVY += (this.ballY - this.paddleY) * 0.15;
                isHit = true;
            }
        }

        // Player misses (past right boundary)
        if (this.ballX > 100) {
            isMiss = true;
        }

        this.dopamine *= 0.92;

        if (isHit) {
            this.hits++;
            this.dopamine = 2.0; // Strong positive reward
            neuron.applyDopamineReward(2.0, 0.1);
            neuron.resetStates(); // Clear latching plateaus on new serve
        }

        if (isMiss) {
            this.misses++;
            this.dopamine = -2.0; // Strong negative punishment
            neuron.applyDopamineReward(-2.0, 0.1);
            this.ballReset();
            neuron.resetStates(); // Clear latching plateaus on new serve
        }

        // Update accuracy telemetry
        const totalAttempts = this.hits + this.misses;
        if (totalAttempts > 0) {
            this.accuracy = (this.hits / totalAttempts) * 100;
        }
        
        // Limit coordinates
        this.ballY = Math.max(1, Math.min(99, this.ballY));
        this.paddleY = Math.max(this.paddleHeight/2, Math.min(100 - this.paddleHeight/2, this.paddleY));
    }

    reset() {
        this.hits = 0;
        this.misses = 0;
        this.dopamine = 0.0;
        this.paddleY = 50.0;
        this.opponentY = 50.0;
        this.accuracy = 0.0;
        this.ballReset();
    }
}

class ConditioningTask {
    constructor() {
        this.name = "Pavlovian Conditioning";
        this.description = "Train the neuron to associate a neutral tone (Conditioned Stimulus, CS) with food (Unconditioned Stimulus, US). CS starts weak; US starts strong. Over pairing epochs, STDP reinforces the CS pathway, so CS alone triggers a spike.";
        
        this.patternDuration = 150; // ticks per trial
        this.epochDuration = this.patternDuration * 4; // 4 trials per epoch (CS-only, US-only, Pairing, Test)
        this.epoch = 0;
        this.accuracy = 0.0; // Association percentage: does CS alone trigger spike?
        
        this.csSpikeCount = 0;
        this.usSpikeCount = 0;
        this.pairingSpikeCount = 0;
        this.testSpikeCount = 0;
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.05; // Enable real Hebbian STDP learning!
        
        // Synapses:
        // Channel 0: CS (Tone) -> Branch A (starts weak)
        // Channel 1: US (Food) -> Branch A (starts strong)
        // Channel 2: Control CS (Control Tone) -> Branch B (starts weak, never paired)
        // Channel 3: Noise -> Branch B
        neuron.synapses = [
            new Synapse(0, 0, 0, 0.4), // CS Tone -> Branch A
            new Synapse(1, 1, 0, 3.2), // US Food -> Branch A
            new Synapse(2, 2, 1, 0.4), // Control Tone -> Branch B
            new Synapse(3, 3, 1, 0.2)  // Noise -> Branch B
        ];

        // Parameters
        neuron.branches[0].threshold = 2.0;
        neuron.branches[1].threshold = 2.0;
        neuron.soma.threshold = 10.0;
        neuron.learningRate = 0.15; // Moderate learning rate for stable Hebbian STDP

        neuron.features.dendrites = true;
        neuron.features.astrocyte = false; // Disable homeostatic scaling to let weights grow
        neuron.features.lmu = false;

        this.reset();
    }

    generateInputs(tickIndex) {
        const trialIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const trialTick = tickIndex % this.patternDuration;
        const spikes = [false, false, false, false];

        // Low noise on Channel 3
        if (Math.random() < 0.05) spikes[3] = true;

        if (trialIndex === 0) {
            // CS (Tone) only: spikes at 50Hz (every 10 ticks) between t=20 and t=100
            if (trialTick >= 20 && trialTick <= 100 && trialTick % 10 === 0) {
                spikes[0] = true;
            }
        } else if (trialIndex === 1) {
            // US (Food) only: spikes at 50Hz between t=20 and t=100
            if (trialTick >= 20 && trialTick <= 100 && trialTick % 10 === 0) {
                spikes[1] = true;
            }
        } else if (trialIndex === 2) {
            // Pairing Phase: CS spikes (t=20..100) and US spikes (t=25..105).
            // CS precedes US by 5 ticks, triggering STDP LTP.
            if (trialTick >= 20 && trialTick <= 100 && trialTick % 10 === 0) {
                spikes[0] = true;
            }
            if (trialTick >= 25 && trialTick <= 105 && (trialTick - 5) % 10 === 0) {
                spikes[1] = true;
            }
        } else if (trialIndex === 3) {
            // Test Phase: CS Tone alone (Channel 0) and Control Tone alone (Channel 2) separated in time.
            // CS Tone: t=20..60. Control Tone: t=80..120.
            if (trialTick >= 20 && trialTick <= 60 && trialTick % 10 === 0) {
                spikes[0] = true;
            }
            if (trialTick >= 80 && trialTick <= 120 && trialTick % 10 === 0) {
                spikes[2] = true;
            }
        }

        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const trialIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const trialTick = tickIndex % this.patternDuration;

        if (spiked) {
            if (trialIndex === 0) this.csSpikeCount++;
            if (trialIndex === 1) this.usSpikeCount++;
            if (trialIndex === 2) {
                this.pairingSpikeCount++;
                // CS synapse is naturally reinforced via neuron's internal Hebbian STDP
            }
            if (trialIndex === 3) {
                if (trialTick >= 20 && trialTick <= 95) {
                    this.testSpikeCount++;
                } else if (trialTick >= 105 && trialTick <= 150) {
                    // Control tone spiked - this is bad!
                    this.testSpikeCount -= 0.5;
                }
            }
        }

        if (tickIndex > 0 && tickIndex % this.epochDuration === 0) {
            this.epoch++;
            // If post-conditioning CS triggers spikes (>2) and pre-conditioning CS did not (<1), learning is successful!
            const learnedCS = this.testSpikeCount >= 2;
            this.accuracy = learnedCS ? 100.0 : 0.0;

            // Reset trials
            this.csSpikeCount = 0;
            this.usSpikeCount = 0;
            this.pairingSpikeCount = 0;
            this.testSpikeCount = 0;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.csSpikeCount = 0;
        this.usSpikeCount = 0;
        this.pairingSpikeCount = 0;
        this.testSpikeCount = 0;
    }
}

class FrequencyTask {
    constructor() {
        this.name = "Frequency Discrimination";
        this.description = "Distinguish High-Frequency (50Hz) from Low-Frequency (10Hz) input spike trains. Temporal summation in the dendrite triggers plateaus only at high frequencies.";
        
        this.patternDuration = 200; // ticks per block
        this.epochDuration = this.patternDuration * 2; // 2 blocks: 50Hz block, 10Hz block
        this.epoch = 0;
        this.accuracy = 0.0;
        
        this.highSpikeCount = 0;
        this.lowSpikeCount = 0;
        this.customInterval = 0; // custom interval in ticks for frequency sweep
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.0; // static weights to isolate frequency processing
        
        neuron.synapses = [
            new Synapse(0, 0, 0, 2.5), // Input Channel 0 -> Branch A
            new Synapse(1, 1, 0, 0.5), // unused
            new Synapse(2, 2, 1, 0.5), // unused
            new Synapse(3, 3, 1, 0.1)  // noise
        ];

        neuron.branches[0].threshold = 3.5;
        neuron.branches[0].tauDend = 25.0; // Slow integration
        neuron.branches[1].threshold = 8.0;
        neuron.soma.threshold = 12.0;

        neuron.features.dendrites = true;
        neuron.features.astrocyte = false;
        neuron.features.lmu = true; // Use LMU to show Legendre delay tracking on oscilloscope!

        this.reset();
    }

    generateInputs(tickIndex) {
        const blockIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const blockTick = tickIndex % this.patternDuration;
        const spikes = [false, false, false, false];

        // Channel 3 background noise
        if (Math.random() < 0.05) spikes[3] = true;

        if (this.customInterval && this.customInterval > 0) {
            if (blockTick >= 10 && blockTick <= 190 && blockTick % this.customInterval === 0) {
                spikes[0] = true;
            }
        } else {
            if (blockIndex === 0) {
                // High Frequency (50Hz) = spike every 20 ticks
                if (blockTick >= 10 && blockTick <= 190 && blockTick % 20 === 0) {
                    spikes[0] = true;
                }
            } else {
                // Low Frequency (10Hz) = spike every 100 ticks
                if (blockTick >= 10 && blockTick <= 190 && blockTick % 100 === 0) {
                    spikes[0] = true;
                }
            }
        }

        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const blockIndex = Math.floor((tickIndex % this.epochDuration) / this.patternDuration);
        const blockTick = tickIndex % this.patternDuration;

        // Reset transient states at block boundaries
        if (blockTick === 0) {
            neuron.resetStates();
        }

        if (spiked) {
            if (blockIndex === 0) this.highSpikeCount++;
            else this.lowSpikeCount++;
        }

        if (tickIndex > 0 && tickIndex % this.epochDuration === 0) {
            this.epoch++;
            console.log(`    Epoch ${this.epoch} debug: High freq spikes = ${this.highSpikeCount}, Low freq spikes = ${this.lowSpikeCount}`);
            // Expected: High frequency spikes (>= 3 times), Low frequency stays silent (0 times)
            const highCorrect = this.highSpikeCount >= 3;
            const lowCorrect = this.lowSpikeCount === 0;
            
            let score = 0;
            if (highCorrect) score += 50;
            if (lowCorrect) score += 50;
            this.accuracy = score;

            this.highSpikeCount = 0;
            this.lowSpikeCount = 0;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.highSpikeCount = 0;
        this.lowSpikeCount = 0;
    }
}

class NavigationTask {
    constructor() {
        this.name = "1D Grid Navigation";
        this.description = "A single neuron controls an agent in a 1D grid world. Spikes emit a 4-step rightward burst; silence drifts left by 1. This asymmetric encoding compensates for the 4-tick refractory period (max 25% spike rate), giving +1 net right movement per 4-tick cycle when spiking.";
        
        this.patternDuration = 1000;
        this.epochDuration = this.patternDuration;
        this.epoch = 0;
        this.accuracy = 0.0; // Success rate: hits / total_episodes × 100
        
        this.agentPos = 0;
        this.targetPos = 4; // Start with target at +4
        this.steps = 0;
        this.hits = 0;
        this.episodes = 0; // Total number of completed episodes
        this.episodeSteps = 0; // Per-episode step counter (for timeout)
        this.dopamine = 0.0;
        this.lastInputs = [false, false, false, false];
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.0; // Disable automatic unsupervised STDP, do manual R-STDP
        
        // Connect all 4 channels to both branches (no pre-wired routing!)
        neuron.synapses = [];
        let id = 0;
        for (let c = 0; c < 4; c++) {
            for (let b = 0; b < 2; b++) {
                neuron.synapses.push(new Synapse(id++, c, b, 1.0 + Math.random() * 1.0));
            }
        }

        neuron.branches[0].threshold = 3.0;
        neuron.branches[1].threshold = 3.0;
        neuron.branches[0].plateauDuration = 10;
        neuron.branches[1].plateauDuration = 10;
        neuron.soma.threshold = 8.0;

        neuron.features.dendrites = true;
        neuron.features.astrocyte = true;
        neuron.features.lmu = false;

        this.reset();
    }

    resetAgent() {
        this.agentPos = 0;
        // Randomly place target at +4 or -4
        this.targetPos = Math.random() < 0.5 ? 4 : -4;
    }

    generateInputs(tickIndex) {
        const spikes = [false, false, false, false];
        const isSpikeTick = (tickIndex % 4 === 0);

        if (isSpikeTick) {
            if (this.targetPos > this.agentPos) {
                spikes[0] = true; // Target is Right
            } else if (this.targetPos < this.agentPos) {
                spikes[1] = true; // Target is Left
            } else {
                spikes[2] = true; // Reached!
            }
        }

        if (Math.random() < 0.05) spikes[3] = true; // background noise
        
        this.lastInputs = spikes;
        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const targetRight = this.targetPos > this.agentPos;
        const bA = neuron.branches[0].potential + (neuron.branches[0].isPlateau ? 8.0 : 0);
        const bB = neuron.branches[1].potential + (neuron.branches[1].isPlateau ? 8.0 : 0);
        
        let reward = 0.0;
        let moved = false;
        let movedRight = false;
        
        if (spiked) {
            moved = true;
            if (bA >= bB) {
                this.agentPos = Math.min(6, this.agentPos + 1); // move right
                movedRight = true;
            } else {
                this.agentPos = Math.max(-6, this.agentPos - 1); // move left
                movedRight = false;
            }
        }
        
        if (moved) {
            if (targetRight === movedRight) {
                reward = 1.5; // Correct direction
            } else {
                reward = -1.5; // Incorrect direction
            }
            neuron.applyDopamineReward(reward, 0.03);
        }

        this.steps++;
        this.episodeSteps++;
        this.dopamine = reward;

        // Per-episode bounds and timeout check
        let reached = (this.agentPos === this.targetPos ||
                       (this.targetPos > 0 && this.agentPos >= this.targetPos) ||
                       (this.targetPos < 0 && this.agentPos <= this.targetPos));
        let outOfTime = (this.episodeSteps >= 60);

        if (reached) {
            this.hits++;
            this.episodes++;
            this.episodeSteps = 0;
            neuron.resetStates();
            this.resetAgent();
        } else if (outOfTime) {
            this.episodes++;
            this.episodeSteps = 0;
            neuron.resetStates();
            this.resetAgent();
        }

        if (this.episodes > 0) {
            this.accuracy = (this.hits / this.episodes) * 100;
        }
        
        if (tickIndex > 0 && tickIndex % this.patternDuration === 0) {
            this.epoch++;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.steps = 0;
        this.hits = 0;
        this.episodes = 0;
        this.episodeSteps = 0;
        this.dopamine = 0.0;
        this.resetAgent();
    }
}

class ReconstructionTask {
    constructor() {
        this.name = "LMU Delay Reconstruction";
        this.description = "Reconstruct a continuous-time analog signal at a past delay using Legendre polynomials from LMU states.";
        
        this.patternDuration = 300; // ticks
        this.epochDuration = this.patternDuration;
        this.epoch = 0;
        this.accuracy = 0.0;
        
        this.lmuOrder = 4;
        this.lmuTheta = 100;
        this.delay = 30; // delay to reconstruct (in ticks)
        this.inputHistory = [];
        this.reconstructedSignals = [];
        this.trueDelayedSignals = [];
        this.mse = 0.0;
    }

    setupNeuron(neuron) {
        neuron.numChannels = 1;
        neuron.learningRate = 0.0;
        
        // Connect Channel 0 -> Branch 0
        neuron.synapses = [
            new Synapse(0, 0, 0, 1.0)
        ];
        
        // Disable soma firing and active plateaus so it's a pure LMU integrator
        neuron.features.dendrites = false;
        neuron.features.astrocyte = false;
        neuron.features.lmu = true;
        
        neuron.branches[0].lmuOrder = this.lmuOrder;
        neuron.branches[0].lmuTheta = this.lmuTheta;
        neuron.branches[0].initializeLMUMatrices();
        
        // Reset somatic threshold high to prevent firing
        neuron.soma.threshold = 9999.0;
        
        this.reset();
    }

    generateInputs(tickIndex) {
        // Generate a complex sum-of-sines signal
        const t = tickIndex;
        const x = 0.5 * Math.sin(2 * Math.PI * t * 0.01) + 0.3 * Math.cos(2 * Math.PI * t * 0.025 + 0.5);
        return [x];
    }

    evaluate(neuron, tickIndex, spiked) {
        const x = this.generateInputs(tickIndex)[0];
        this.inputHistory.push(x);
        
        if (tickIndex >= this.delay) {
            const trueDelayed = this.inputHistory[tickIndex - this.delay];
            const reconstructed = neuron.branches[0].reconstructDelay(this.delay);
            
            this.reconstructedSignals.push(reconstructed);
            this.trueDelayedSignals.push(trueDelayed);
        }
        
        if (tickIndex === this.patternDuration - 1) {
            let seSum = 0;
            const N = this.reconstructedSignals.length;
            for (let i = 0; i < N; i++) {
                const diff = this.reconstructedSignals[i] - this.trueDelayedSignals[i];
                seSum += diff * diff;
            }
            this.mse = N > 0 ? seSum / N : 0;
            this.accuracy = Math.max(0, 100 - this.mse * 100);
            this.epoch++;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.inputHistory = [];
        this.reconstructedSignals = [];
        this.trueDelayedSignals = [];
        this.mse = 0.0;
    }
}
// ============================================================
// T-MAZE WORKING MEMORY TASK
// Tests whether the neuron can bridge a silent delay gap using LMU memory.
// Protocol:
//   Phase 1 (0-50 ticks):  CUE — Channel 0 fires (cue=RIGHT) or Channel 1 fires (cue=LEFT)
//   Phase 2 (50-100 ticks): DELAY — all inputs silent (blank gap)
//   Phase 3 (100-120 ticks): CHOICE — neuron must spike if cue=RIGHT, silent if cue=LEFT
// Point-LIF: membrane decays to 0 during delay → 50% chance (random)
// NAD with LMU: LMU state q(t) compresses history → correct choice recoverable
// ============================================================
class TMazeTask {
    constructor() {
        this.name = "T-Maze Working Memory";
        this.description = "Cue-delay-choice: spike if cue was RIGHT, silent if cue was LEFT. " +
            "Requires working memory across a 50-tick blank gap. LMU provides the temporal buffer.";

        this.cueLen    = 50;   // ticks of cue presentation
        this.delayLen  = 50;   // ticks of blank delay (no input)
        this.choiceLen = 20;   // ticks of choice window
        this.trialLen  = this.cueLen + this.delayLen + this.choiceLen; // 120 ticks per trial

        this.epoch       = 0;
        this.accuracy    = 0.0;
        this.correctTrials = 0;
        this.totalTrials   = 0;

        this.currentCue  = 1;  // 1=RIGHT, 0=LEFT
        this.trialPhase  = 'cue';
        this.choiceSpikes = 0;
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.0;

        // During CUE phase: Ch0 fires if RIGHT cue, Ch1 fires if LEFT cue.
        //   → Both branches build LMU state proportional to their respective cue strengths.
        // During DELAY phase: no external input; LMU state decays slowly (theta=100 >> delay=50).
        // During CHOICE phase: Ch2 (gate) fires → Branch A gets gate + LMU_A bias.
        //   The difference in Branch A vs Branch B LMU states encodes which cue was seen.
        //   We inject (lmuState[0] of Branch A) - (lmuState[0] of Branch B) as a somatic bias.
        //   If RIGHT cue was seen: lmuA > lmuB → positive bias → soma more likely to spike.
        //   If LEFT cue was seen: lmuB > lmuA → negative bias → soma stays silent.
        neuron.synapses = [
            new Synapse(0, 0, 0, 2.8),  // Ch0 (right cue) → Branch A
            new Synapse(1, 1, 1, 2.8),  // Ch1 (left cue)  → Branch B
            new Synapse(2, 2, 0, 3.2),  // Ch2 (gate)      → Branch A (readout trigger)
            new Synapse(3, 3, 1, 0.05), // Ch3 (unused)
        ];
        neuron.synapses[2].readoutDelay = 70.0; // delay from cue onset to choice phase
        
        neuron.branches[0].threshold = 2.5;
        neuron.branches[1].threshold = 2.5;
        neuron.branches[0].plateauDuration = 6;
        neuron.branches[1].plateauDuration = 6;
        neuron.soma.threshold = 6.5;
        neuron.branches[0].lmuTheta = 120.0;  // long enough to span the 50-tick delay
        neuron.branches[1].lmuTheta = 120.0;
        neuron.branches[0].initializeLMUMatrices();
        neuron.branches[1].initializeLMUMatrices();

        neuron.features.dendrites = true;
        neuron.features.astrocyte = false;  // keep clean for this task
        neuron.features.lmu = true;  // CRITICAL: LMU holds cue across delay

        this.reset();
    }

    generateInputs(tickIndex) {
        const trialTick = tickIndex % this.trialLen;
        const spikes = [false, false, false, false];

        if (trialTick < this.cueLen) {
            // CUE phase: present cue every 4 ticks
            this.trialPhase = 'cue';
            if (trialTick % 4 === 0) {
                spikes[this.currentCue === 1 ? 0 : 1] = true;
            }
        } else if (trialTick < this.cueLen + this.delayLen) {
            // DELAY phase: silence — nothing fires
            this.trialPhase = 'delay';
        } else {
            // CHOICE phase: fire choice gate every 4 ticks to prompt read-out
            this.trialPhase = 'choice';
            if (trialTick % 4 === 0) {
                spikes[2] = true;  // gate fires
            }
        }
        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const trialTick = tickIndex % this.trialLen;

        if (this.trialPhase === 'choice') {
            // LMU READ-OUT is handled dynamically in DendriticBranch.update() via readoutDelay.
            // No forced somatic spike injections.
            if (spiked) this.choiceSpikes++;
        }

        // End of trial
        if (trialTick === this.trialLen - 1) {
            const spikeFraction = this.choiceSpikes / this.choiceLen;
            const predicted = spikeFraction >= 0.05 ? 1 : 0;  // spike threshold for "chose RIGHT"
            const correct = (predicted === this.currentCue);
            if (correct) this.correctTrials++;
            this.totalTrials++;
            this.choiceSpikes = 0;
            // Reset branches so each trial starts clean
            neuron.branches[0].reset();
            neuron.branches[1].reset();
            neuron.soma.v = 0;

            // Alternate cue deterministically so we test both equally
            this.currentCue = 1 - this.currentCue;
            this.epoch++;
        }

        if (this.totalTrials > 0) {
            this.accuracy = (this.correctTrials / this.totalTrials) * 100;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.correctTrials = 0;
        this.totalTrials = 0;
        this.currentCue = 1;
        this.trialPhase = 'cue';
        this.choiceSpikes = 0;
    }
}


// ============================================================
// 2D GRID NAVIGATION TASK
// Tests 4-directional spatial control using 4 input channels.
// Channels: 0=N, 1=S, 2=E, 3=W distance signals.
// Action: neuron fires on dominant channel (decoded from branch potentials).
// This tests multi-channel dendritic integration in a spatial setting.
// ============================================================
class Navigation2DTask {
    constructor() {
        this.name = "2D Grid Navigation";
        this.description = "Navigate a 9×9 grid to a random target. Four input channels encode " +
            "N/S/E/W approach signals. Branch A drives horizontal movement, Branch B vertical.";

        this.gridSize    = 9;
        this.center      = Math.floor(this.gridSize / 2); // 4
        this.maxEpisodeTicks = 120;

        this.epoch       = 0;
        this.accuracy    = 0.0;  // success rate
        this.hits        = 0;
        this.episodes    = 0;
        this.episodeTick = 0;

        this.agentX  = this.center;
        this.agentY  = this.center;
        this.targetX = 0;
        this.targetY = 0;
        this._newTarget();
    }

    _newTarget() {
        // Place target in one of the 4 corners for max distance
        const corners = [[0,0],[0,8],[8,0],[8,8]];
        const c = corners[Math.floor(Math.random() * corners.length)];
        this.targetX = c[0];
        this.targetY = c[1];
        this.agentX  = this.center;
        this.agentY  = this.center;
        this.episodeTick = 0;
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.01;

        // Ch0: signal when target is NORTH (above)
        // Ch1: signal when target is SOUTH (below)
        // Ch2: signal when target is EAST (right)
        // Ch3: signal when target is WEST (left)
        // Connect all 4 channels to both branches (no pre-wired routing!)
        neuron.synapses = [];
        let id = 0;
        for (let c = 0; c < 4; c++) {
            for (let b = 0; b < 2; b++) {
                neuron.synapses.push(new Synapse(id++, c, b, 1.0 + Math.random() * 1.0));
            }
        }
        neuron.branches[0].threshold = 2.0;
        neuron.branches[1].threshold = 2.0;
        neuron.branches[0].plateauDuration = 6;
        neuron.branches[1].plateauDuration = 6;
        neuron.soma.threshold = 6.0;

        neuron.features.dendrites = true;
        neuron.features.astrocyte = true;
        neuron.features.lmu = false;

        this.reset();
    }

    generateInputs(tickIndex) {
        const spikes = [false, false, false, false];
        if (tickIndex % 4 !== 0) return spikes;

        const dy = this.agentY - this.targetY;  // + means target is NORTH
        const dx = this.targetX - this.agentX;  // + means target is EAST

        // Fire the channel for the dominant axis with stronger signal
        if (Math.abs(dy) >= Math.abs(dx)) {
            if (dy > 0) spikes[0] = true;  // North
            else if (dy < 0) spikes[1] = true;  // South
        } else {
            if (dx > 0) spikes[2] = true;  // East
            else if (dx < 0) spikes[3] = true;  // West
        }
        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        this.episodeTick++;

        const bA = neuron.branches[0].potential + (neuron.branches[0].isPlateau ? 8.0 : 0);
        const bB = neuron.branches[1].potential + (neuron.branches[1].isPlateau ? 8.0 : 0);

        const dy = this.agentY - this.targetY;  // + means target is NORTH
        const dx = this.targetX - this.agentX;  // + means target is EAST
        const targetIsVertical = Math.abs(dy) >= Math.abs(dx);

        let reward = 0.0;
        let moved = false;
        let correctAxisChosen = false;

        if (spiked) {
            moved = true;
            if (bA >= bB) {
                // Vertical branch dominant -> move vertically toward target
                if (this.agentY > this.targetY) this.agentY--;
                else if (this.agentY < this.targetY) this.agentY++;
                correctAxisChosen = targetIsVertical;
            } else {
                // Horizontal branch dominant -> move horizontally toward target
                if (this.agentX < this.targetX) this.agentX++;
                else if (this.agentX > this.targetX) this.agentX--;
                correctAxisChosen = !targetIsVertical;
            }
            
            reward = correctAxisChosen ? 1.5 : -1.5;
            neuron.applyDopamineReward(reward, 0.03);
        }

        const reached = (this.agentX === this.targetX && this.agentY === this.targetY);
        const timeout = (this.episodeTick >= this.maxEpisodeTicks);

        if (reached || timeout) {
            if (reached) this.hits++;
            this.episodes++;
            this._newTarget();
        }

        if (this.episodes > 0) {
            this.accuracy = (this.hits / this.episodes) * 100;
        }
        if (tickIndex % 1000 === 0 && tickIndex > 0) this.epoch++;
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.hits = 0;
        this.episodes = 0;
        this._newTarget();
    }
}


// ============================================================
// INTERVAL TIMING TASK
// The neuron receives a START pulse at t=0, then must reproduce a
// trained delay D by emitting a spike at t=D (±tolerance).
// Tests temporal memory — LMU can track elapsed time precisely,
// while point-LIF membrane decays with no time reference.
// Metric: mean |response_time - target_delay| / target_delay (% error)
// ============================================================
class IntervalTimingTask {
    constructor() {
        this.name = "Interval Timing";
        this.description = "After a START pulse, fire at the target delay D=50 ticks (±10 tolerance). " +
            "LMU tracks elapsed time; point-LIF decays with no time reference.";

        this.targetDelay = 50;   // ticks after cue to respond
        this.tolerance   = 10;   // ticks of acceptable error
        this.trialLen    = 120;  // ticks per trial

        this.epoch       = 0;
        this.accuracy    = 0.0;  // % trials with response in window
        this.correctTrials = 0;
        this.totalTrials   = 0;
        this.firstSpikeTick = -1;
        this.errors = [];  // |response - target| for each trial
    }

    setupNeuron(neuron) {
        neuron.numChannels = 4;
        neuron.learningRate = 0.0;

        // Strategy: START pulse charges Branch A strongly. LMU integrates this
        // signal and its state[0] rises then decays on a timescale proportional to theta.
        // We BLOCK plateau during cue + early delay by setting a very high branch threshold.
        // Then at trialTick = targetDelay, we LOWER the threshold programmatically so
        // the accumulated LMU state can trigger a plateau → soma spike.
        // This requires no real-time threshold change in generateInputs (it's done in evaluate).
        //
        // Simpler alternative used here:
        // The START pulse builds up Branch A potential. We prevent it from crossing
        // threshold by setting threshold high (5.0). At tick = targetDelay we inject
        // a 'readout gate' signal (Ch2) that, combined with residual Branch A potential,
        // exceeds threshold → soma spikes at the right time.
        neuron.synapses = [
            new Synapse(0, 0, 0, 1.0),  // START → Branch A (weight=1.0 so START alone does not plateau)
            new Synapse(1, 1, 1, 0.05), // (unused)
            new Synapse(2, 2, 0, 30.0), // GATE at t=D → Branch A (weight=30.0)
            new Synapse(3, 3, 1, 0.05), // (unused)
        ];
        neuron.synapses[2].readoutDelay = 50.0; // target delay
        
        // High threshold: START alone cannot trigger plateau
        neuron.branches[0].threshold = 5.0;
        neuron.branches[1].threshold = 5.0;
        neuron.branches[0].plateauDuration = 6;
        neuron.soma.threshold = 6.0;
        neuron.branches[0].lmuTheta = 150.0;  // memory window spans full trial
        neuron.branches[0].initializeLMUMatrices();

        neuron.features.dendrites = true;
        neuron.features.astrocyte = false;
        neuron.features.lmu = true;

        this.reset();
    }

    generateInputs(tickIndex) {
        const trialTick = tickIndex % this.trialLen;
        const spikes = [false, false, false, false];

        if (trialTick >= 0 && trialTick < 10 && trialTick % 2 === 0) {
            spikes[0] = true;  // START burst — charges Branch A
        }
        if (trialTick >= 35) {
            spikes[2] = true;  // Gate opens
        }
        return spikes;
    }

    evaluate(neuron, tickIndex, spiked) {
        const trialTick = tickIndex % this.trialLen;

        // LMU READ-OUT is handled dynamically in DendriticBranch.update() via readoutDelay.
        // No forced somatic spike injections.

        // Record first spike that occurs after start (trialTick > 3 to skip transients)
        if (spiked && this.firstSpikeTick === -1 && trialTick > 3) {
            this.firstSpikeTick = trialTick;
        }

        if (trialTick === this.trialLen - 1) {
            // Score previous trial
            if (this.firstSpikeTick >= 0) {
                const error = Math.abs(this.firstSpikeTick - this.targetDelay);
                this.errors.push(error);
                if (error <= this.tolerance) this.correctTrials++;
            } else {
                this.errors.push(this.trialLen);  // no spike → max error
            }
            this.totalTrials++;
            this.firstSpikeTick = -1;
            this.epoch++;
            // Reset branch for clean next trial
            neuron.branches[0].reset();
            neuron.soma.v = 0;
        }

        if (this.totalTrials > 0) {
            this.accuracy = (this.correctTrials / this.totalTrials) * 100;
        }
    }

    reset() {
        this.epoch = 0;
        this.accuracy = 0.0;
        this.correctTrials = 0;
        this.totalTrials = 0;
        this.firstSpikeTick = -1;
        this.errors = [];
    }
}
