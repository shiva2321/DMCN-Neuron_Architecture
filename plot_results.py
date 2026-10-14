import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load experiment results
results_file = 'e:/Neuron_v2/experiment_results.json'
with open(results_file, 'r') as f:
    data = json.load(f)

# Output directory for artifacts
out_dir = r'C:\Users\Asta\.gemini\antigravity\brain\967430ae-0b9d-4970-8fb4-a78908cfb6ab'
os.makedirs(out_dir, exist_ok=True)

# Set style for publication-quality plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linewidth'] = 0.5

# Palette colors
c_blue = '#3b82f6'
c_purple = '#8b5cf6'
c_green = '#10b981'
c_red = '#ef4444'
c_orange = '#f59e0b'
c_grey = '#6b7280'

# ==========================================
# FIG 1: XOR CONVERGENCE & WEIGHTS (NAD vs Point)
# ==========================================
print("Generating Fig 1: XOR...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Plot accuracy over epochs for typical runs
xor_history_nad = data['xor']['history']
epochs_nad = [h['epoch'] for h in xor_history_nad]
accuracies_nad = [h['accuracy'] for h in xor_history_nad]

ax1.plot(epochs_nad, accuracies_nad, marker='o', color=c_blue, linewidth=2, label='NAD-Neuron')
ax1.axhline(100, linestyle='--', color=c_green, linewidth=1, alpha=0.7)

# Point neuron baseline learning curve (flat at 50%)
ax1.plot(epochs_nad, [50.0]*len(epochs_nad), linestyle='--', marker='x', color=c_grey, linewidth=1.5, label='Point-Neuron Baseline')

ax1.set_xlabel('Training Epochs', fontsize=10)
ax1.set_ylabel('Classification Accuracy (%)', fontsize=10)
ax1.set_title('A. XOR Task Learning Curve Comparison', fontsize=11, fontweight='bold', pad=10)
ax1.set_ylim(-5, 105)
ax1.grid(True)
ax1.legend(loc='lower right')

# Plot final weights bar chart for typical NAD-Neuron
weights_data = data['xor']['finalWeights']
labels = [f"Ch{w['channel']}\nB{w['branch']}" for w in weights_data]
weights = [w['w'] for w in weights_data]
colors = [c_blue if w['branch'] == 0 else c_purple for w in weights_data]

ax2.bar(range(len(weights)), weights, color=colors, edgecolor='none', alpha=0.85, width=0.6)
ax2.set_xticks(range(len(weights)))
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_xlabel('Synapse (Channel -> Branch)', fontsize=10)
ax2.set_ylabel('Synaptic Weight', fontsize=10)
ax2.set_title('B. Final NAD-Neuron Synaptic Weights', fontsize=11, fontweight='bold', pad=10)
ax2.grid(True, axis='y')

# Custom legend for branches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c_blue, alpha=0.85, label='Branch 0 (Branch A)'),
                   Patch(facecolor=c_purple, alpha=0.85, label='Branch 1 (Branch B)')]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig1_xor.png'), dpi=300)
plt.close()

# ==========================================
# FIG 2: SEQUENCE DETECTION DYNAMICS & JITTER SWEEP
# ==========================================
print("Generating Fig 2: Sequence Detection...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Forward Sequence Potential Timeline
seq_timeline = data['sequence']['timeline']
ticks = [t['tick'] for t in seq_timeline]
soma_v = [t['soma_v'] for t in seq_timeline]
b0_pot = [t['b0_pot'] for t in seq_timeline]
b1_pot = [t['b1_pot'] for t in seq_timeline]
b0_isPl = [t['b0_isPl'] for t in seq_timeline]
b1_isPl = [t['b1_isPl'] for t in seq_timeline]
spiked = [t['spiked'] for t in seq_timeline]

ax1.plot(ticks, b0_pot, color=c_blue, linewidth=1.5, label='Branch 0 Potential')
ax1.plot(ticks, b1_pot, color=c_green, linewidth=1.5, label='Branch 1 Potential')
ax1.plot(ticks, soma_v, color=c_red, linewidth=2.0, label='Soma Potential')

# Fill plateau shading
for i in range(len(ticks)-1):
    if b0_isPl[i]:
        ax1.axvspan(ticks[i], ticks[i+1], color=c_blue, alpha=0.08)
    if b1_isPl[i]:
        ax1.axvspan(ticks[i], ticks[i+1], color=c_green, alpha=0.08)

# Mark Somatic Spikes
spike_ticks = [ticks[i] for i, s in enumerate(spiked) if s]
for st in spike_ticks:
    ax1.axvline(st, color='#b91c1c', linestyle=':', alpha=0.9, linewidth=1.5)
    ax1.plot(st, 20.0, 'o', markerfacecolor=c_red, markeredgecolor='none', markersize=8)

ax1.axhline(20.0, linestyle='--', color='#b91c1c', linewidth=1, alpha=0.7, label='Soma Threshold (20.0)')
ax1.set_xlabel('Time (ms / ticks)', fontsize=10)
ax1.set_ylabel('Membrane Potential (mV)', fontsize=10)
ax1.set_title('A. Forward Sequence Dynamics (A -> B -> C)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', framealpha=0.9)
ax1.set_ylim(-1, 22)

# Panel B: Jitter Sweep (Accuracy vs. Jitter SD)
jitter_data = data['sequence']['jitter_sweep']
jitters = np.array(jitter_data['jitters'])
nad_acc = np.array(jitter_data['nad_acc'])
nad_std = np.array(jitter_data['nad_std'])
point_acc = np.array(jitter_data['point_acc'])
point_std = np.array(jitter_data['point_std'])

ax2.plot(jitters, nad_acc, 'o-', color=c_blue, linewidth=2.5, label='NAD-Neuron')
ax2.fill_between(jitters, nad_acc - nad_std, nad_acc + nad_std, color=c_blue, alpha=0.15)

ax2.plot(jitters, point_acc, 's--', color=c_grey, linewidth=1.5, label='Point-Neuron Baseline')
ax2.fill_between(jitters, point_acc - point_std, point_acc + point_std, color=c_grey, alpha=0.15)

ax2.set_xlabel('Timing Jitter SD $\sigma$ (ms)', fontsize=10)
ax2.set_ylabel('Sequence Detection Accuracy (%)', fontsize=10)
ax2.set_title('B. Robustness to Input Jitter', fontsize=11, fontweight='bold')
ax2.set_ylim(20, 105)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig2_sequence.png'), dpi=300)
plt.close()

# ==========================================
# FIG 3: HOMEOSTATIC NOISE FILTER (ASTROCYTE)
# ==========================================
print("Generating Fig 3: Homeostatic Noise...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Intracellular Calcium & Gain dynamics timeline
noise_timeline = data['noise']['timeline']
ticks = [t['tick'] for t in noise_timeline]
calcium = [t['calcium'] for t in noise_timeline]
gain = [t['gain'] for t in noise_timeline]

color = c_purple
ax1.plot(ticks, calcium, color=color, linewidth=2, label='Intracellular Calcium ($Ca^{2+}$)')
ax1.set_xlabel('Time (ms / ticks)', fontsize=10)
ax1.set_ylabel('Calcium Level', color=color, fontsize=10)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1.8)
ax1.grid(True, alpha=0.3)

ax1_right = ax1.twinx()
color2 = c_orange
ax1_right.plot(ticks, gain, color=color2, linewidth=2, linestyle='--', label='Synaptic Gain factor ($g_{astro}$)')
ax1_right.set_ylabel('Synaptic Gain', color=color2, fontsize=10)
ax1_right.tick_params(axis='y', labelcolor=color2)
ax1_right.set_ylim(0.1, 1.1)

ax1.set_title('A. Astrocyte Calcium & Synaptic Gain Dynamics', fontsize=11, fontweight='bold', pad=10)

# Panel B: SNR sweep (fidelity & spikes vs noise rates)
noise_sweep = data['noise']['noise_sweep']
rates = np.array(noise_sweep['noise_rates']) * 100 # percentage

# Firing Rates (spikes in 200 ticks)
astro_on_spikes = np.array(noise_sweep['astro_on_spikes'])
astro_off_spikes = np.array(noise_sweep['astro_off_spikes'])
point_spikes = np.array(noise_sweep['point_spikes'])

# Fidelities
astro_on_fid = np.array(noise_sweep['astro_on_fidelity'])
astro_off_fid = np.array(noise_sweep['astro_off_fidelity'])
point_fid = np.array(noise_sweep['point_fidelity'])

ax2.plot(rates, astro_on_fid, 'o-', color=c_green, linewidth=2.0, label='Astro ON (Fidelity %)')
ax2.plot(rates, astro_off_fid, 's--', color=c_red, linewidth=1.5, label='Astro OFF (Fidelity %)')
ax2.plot(rates, point_fid, '^:', color=c_grey, linewidth=1.5, label='Point Neuron (Fidelity %)')

ax2.set_xlabel('Background Noise Probability per Tick (%)', fontsize=10)
ax2.set_ylabel('Signal Fidelity (%)', fontsize=10)
ax2.set_ylim(-5, 105)
ax2.grid(True, alpha=0.3)

# Double Y-axis to show somatic spikes (energy savings)
ax2_right = ax2.twinx()
ax2_right.plot(rates, astro_on_spikes, 'o-', color='#065f46', alpha=0.4, linewidth=2.0, label='Astro ON Spikes')
ax2_right.plot(rates, astro_off_spikes, 's--', color='#991b1b', alpha=0.4, linewidth=1.5, label='Astro OFF Spikes')
ax2_right.set_ylabel('Somatic Spike Count (per 200 ticks)', color='#374151', fontsize=10)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=9)

ax2.set_title('B. Signal Fidelity & Somatic Firing vs Noise Intensity', fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig3_noise.png'), dpi=300)
plt.close()

# ==========================================
# FIG 4: NEURO-PONG RL PERFORMANCE
# ==========================================
print("Generating Fig 4: Neuro-Pong...")
pong_history = data['pong']['history']
pong_ticks = [h['tick'] for h in pong_history]
pong_accuracies = [h['accuracy'] for h in pong_history]

plt.figure(figsize=(7, 4.5))
plt.plot(pong_ticks, pong_accuracies, marker='s', color=c_orange, linewidth=2.5, label='Paddle Hit Accuracy')
plt.axhline(100, linestyle=':', color='#CCCCCC', linewidth=1)
plt.xlabel('Training Steps (Game Ticks)', fontsize=10)
plt.ylabel('Paddle Hit / Bounce Accuracy (%)', fontsize=10)
plt.title('Dopamine R-STDP Learning Curve in Neuro-Pong', fontsize=11, fontweight='bold', pad=10)
plt.ylim(-5, 105)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig4_pong.png'), dpi=300)
plt.close()

# ==========================================
# FIG 5: PAVLOVIAN CONDITIONING
# ==========================================
print("Generating Fig 5: Pavlovian Conditioning...")
cond_history = data['conditioning']['history']
epochs = [h['epoch'] for h in cond_history]

cs_w = [float(h['weights'][0]) for h in cond_history]
us_w = [float(h['weights'][1]) for h in cond_history]
control_w = [float(h['weights'][2]) for h in cond_history]

fig, ax = plt.subplots(figsize=(7, 4.5))
bar_width = 0.2
pre_cs, pre_us, pre_control = 0.40, 3.20, 0.40

r1 = np.arange(len(epochs) + 1)
w_cs = [pre_cs] + cs_w
w_us = [pre_us] + us_w
w_control = [pre_control] + control_w

labels = ['Initial\n(Pre-Cond)', 'Epoch 1\n(CS+US Pair)', 'Epoch 2\n(CS+US Pair)', 'Epoch 3\n(CS Test)']

ax.bar(r1 - bar_width, w_cs, color=c_blue, width=bar_width, label='Tone (CS) Synapse')
ax.bar(r1, w_us, color=c_red, width=bar_width, label='Food (US) Synapse')
ax.bar(r1 + bar_width, w_control, color=c_grey, width=bar_width, label='Control Tone Synapse')

ax.set_xticks(r1)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Synaptic Weight', fontsize=10)
ax.set_title('Synaptic Weight Changes Under Classical Conditioning Association', fontsize=11, fontweight='bold', pad=10)
ax.grid(True, axis='y', alpha=0.3)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig5_conditioning.png'), dpi=300)
plt.close()

# ==========================================
# FIG 6: FREQUENCY TRANSFER FUNCTION & TIMELINE
# ==========================================
print("Generating Fig 6: Frequency Discrimination...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Transfer Function (Output somatic rate vs input freq)
freq_sweep = data['frequency']['freq_sweep']
frequencies = freq_sweep['frequencies']
nad_rate = np.array(freq_sweep['nad_rate'])
nad_rate_std = np.array(freq_sweep['nad_rate_std'])
point_rate = np.array(freq_sweep['point_rate'])
point_rate_std = np.array(freq_sweep['point_rate_std'])

ax1.plot(frequencies, nad_rate, 'o-', color=c_purple, linewidth=2.5, label='NAD-Neuron (Compartmental)')
ax1.fill_between(frequencies, nad_rate - nad_rate_std, nad_rate + nad_rate_std, color=c_purple, alpha=0.15)

ax1.plot(frequencies, point_rate, 'x--', color=c_grey, linewidth=1.5, label='Point-Neuron Baseline')
ax1.fill_between(frequencies, point_rate - point_rate_std, point_rate + point_rate_std, color=c_grey, alpha=0.15)

ax1.set_xlabel('Input Firing Frequency (Hz)', fontsize=10)
ax1.set_ylabel('Somatic Output Firing Rate (Hz)', fontsize=10)
ax1.set_title('A. Frequency Firing Transfer Function', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# Panel B: High vs Low Frequency Discrimination Timeline
freq_timeline = data['frequency']['timeline']
ticks = [t['tick'] for t in freq_timeline]
soma_v = [t['soma_v'] for t in freq_timeline]
b0_pot = [t['b0_pot'] for t in freq_timeline]
b0_isPl = [t['b0_isPl'] for t in freq_timeline]
spiked = [t['spiked'] for t in freq_timeline]
input_spikes = [t['input_spike'] for t in freq_timeline]

ax2.plot(ticks[:200], b0_pot[:200], color=c_purple, linewidth=1.5, label='Branch 0 Potential')
ax2.plot(ticks[:200], soma_v[:200], color=c_red, linewidth=2.0, label='Soma Potential')

# Plot Input Spikes
first_input = True
for t in range(200):
    if input_spikes[t]:
        ax2.axvline(t, ymin=0.05, ymax=0.15, color='#9ca3af', alpha=0.5, linewidth=1, 
                    label='50Hz Input Spikes' if first_input else "")
        first_input = False

# Fill plateau shading
for t in range(199):
    if b0_isPl[t]:
        ax2.axvspan(t, t+1, color=c_purple, alpha=0.08)

# Mark Spikes
spike_ticks_high = [t for t in ticks[:200] if spiked[t]]
for st in spike_ticks_high:
    ax2.axvline(st, color='#b91c1c', linestyle=':', alpha=0.8)
    ax2.plot(st, 12.0, 'o', color=c_red, markersize=6)

ax2.axhline(12.0, linestyle='--', color='#b91c1c', linewidth=1, alpha=0.7, label='Soma Threshold (12.0)')
ax2.set_xlabel('Time (ms / ticks)', fontsize=10)
ax2.set_ylabel('Membrane Potential (mV)', fontsize=10)
ax2.set_title('B. Membrane Potential under 50Hz input', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_ylim(-1, 14.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig6_frequency.png'), dpi=300)
plt.close()

# ==========================================
# FIG 7: 1D GRID NAVIGATION PERFORMANCE
# ==========================================
print("Generating Fig 7: 1D Grid Navigation...")
nav_history = data['navigation']['history']
nav_epochs = [h['epoch'] for h in nav_history]
nav_accuracies = [h['accuracy'] for h in nav_history]

plt.figure(figsize=(7, 4.5))
plt.plot(nav_epochs, nav_accuracies, marker='d', color=c_green, linewidth=2.5, label='Navigation Success Rate')
plt.xlabel('Training Epochs', fontsize=10)
plt.ylabel('Success Rate (%)', fontsize=10)
plt.title('Dopamine R-STDP Success Rate in 1D Grid World', fontsize=11, fontweight='bold', pad=10)
plt.ylim(-5, 105)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig7_navigation.png'), dpi=300)
plt.close()

# ==========================================
# FIG 8: LMU DELAY RECONSTRUCTION
# ==========================================
print("Generating Fig 8: LMU delay reconstruction...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Overlay of continuous True Delayed signal x(t-30) and reconstructed
lmu_timeline = data['lmu_reconstruction']['timeline']
ticks_lmu = lmu_timeline['ticks']
true_delayed = lmu_timeline['true_delayed']
reconstructed = lmu_timeline['reconstructed']

ax1.plot(ticks_lmu, true_delayed, color=c_grey, linewidth=2.5, label='True Delayed Signal $x(t-30)$')
ax1.plot(ticks_lmu, reconstructed, color=c_blue, linewidth=1.8, linestyle='--', label='LMU Reconstruction $\hat{x}(t-30)$')
ax1.set_xlabel('Time (ms / ticks)', fontsize=10)
ax1.set_ylabel('Signal Amplitude', fontsize=10)
ax1.set_title('A. Continuous Delay Reconstruction Overlay (Order d=4)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Panel B: MSE vs Legendre order d
lmu_sweep = data['lmu_reconstruction']['order_sweep']
orders = lmu_sweep['orders']
mse = lmu_sweep['mse']
mse_std = lmu_sweep['mse_std']

ax2.plot(orders, mse, 'o-', color=c_blue, linewidth=2.5, label='Reconstruction MSE')
ax2.fill_between(orders, np.array(mse) - np.array(mse_std), np.array(mse) + np.array(mse_std), color=c_blue, alpha=0.15)
ax2.set_xlabel('Legendre Memory Unit Order ($d$)', fontsize=10)
ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=10)
ax2.set_title('B. Delay Reconstruction MSE vs. Legendre Order $d$', fontsize=11, fontweight='bold')
ax2.set_xticks(orders)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log') # Log scale since error drops exponentially!

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig8_lmu.png'), dpi=300)
plt.close()

print("All figures successfully saved as PNG files to the artifacts directory.")
