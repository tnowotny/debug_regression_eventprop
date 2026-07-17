import numpy as np
import matplotlib.pyplot as plt

from ml_genn import Connection, Population, Network
from ml_genn.callbacks import (OptimiserParamSchedule, SpikeRecorder,
                               VarRecorder)
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.synapses import Exponential
from ml_genn.optimisers import Adam

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes
from ml_genn.compilers.event_prop_compiler import default_params

NUM_INPUT = 3
NUM_HIDDEN = 3
NUM_OUTPUT = 3
IN_GROUP_SIZE = 4
IN_ACTIVE_ISI = 10
IN_ACTIVE_INTERVAL = 200
NUM_TRIALS = 300
NUM_EPOCHS = 15
TAU_MEM = 20.0
TAU_SYN = 5.0
LR = 0.001
TRIAL_TIME = 1000
SHOW_TARGET = True
DT = 0.1

# PSP from a single spike if V starts at 0 and I jumps to 1 at t == 0 and dt == 1
def alpha_func(taus, taum, steps, dt):
    x = np.arange(steps)
    y = 1/(taus/taum - 1)*(np.exp(-x*dt/taus)-np.exp(-x*dt/taum))
    return y

# Convert frequencies of each component into row vector
pos = [1000, 2000, 3000]
amp = [ 1.0, -1.0, 2.0 ]

# Calculate Y* target
trial_steps = int(TRIAL_TIME/DT)
y_star = np.zeros((trial_steps, NUM_OUTPUT))
for i in range(NUM_OUTPUT):
    y_star[pos[i]:, i] = amp[i]*alpha_func(TAU_SYN,TAU_MEM,trial_steps-pos[i], DT)

if SHOW_TARGET:
    plt.figure()
    for i in range(NUM_OUTPUT):
        plt.plot(y_star[:,i])    
    plt.show()

# Shift each spike time by group start
in_spike_times = np.array(pos)*DT-2

# Create matching array of IDs
in_spike_ids = np.arange(3)

# Pre-process spikes
in_spikes = preprocess_spikes(in_spike_times.flatten(), in_spike_ids, NUM_INPUT)

in_spikes = [in_spikes] * NUM_TRIALS
y_star = [y_star] * NUM_TRIALS

i2h_weight = [[ 5.5, 0 ,0 ], [ 0, 5, 0 ], [0, 0, 5]]
h2o_weight = [[ 5, 0 ,0 ], [ 0, -5, 0 ], [0, 0, 5]]

network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=len(in_spike_ids)),
                                  NUM_INPUT, record_spikes=True)
    hidden = Population(LeakyIntegrateFire(v_thresh=0.61, tau_mem=TAU_MEM),
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=TAU_MEM, readout="var"),
                        NUM_OUTPUT)
    
    # Connections
    in_hid = Connection(input, hidden, Dense(i2h_weight), Exponential(TAU_SYN))
    #Connection(hidden, hidden, Dense(Normal(sd=0.5 / np.sqrt(NUM_HIDDEN))), Exponential(TAU_SYN))
    hid_out = Connection(hidden, output, Dense(h2o_weight), Exponential(TAU_SYN))

compiler = EventPropCompiler(example_timesteps=trial_steps, losses="mean_square_error", max_spikes=1500, dt=DT, optimiser= Adam(LR),batch_size=1)
compiled_net = compiler.compile(network, 
                                #regularisers={"all_hidden_populations": SpikeCount(1e-8, 10)}
)

with compiled_net:
    #def alpha_schedule(epoch, alpha):
    #    if (epoch % 2) == 0 and epoch != 0:
    #        return alpha * 0.7
    #    else:
    #        return alpha

    # Evaluate model on numpy dataset
    start_time = perf_counter()
    callbacks = ["batch_progress_bar", 
                 VarRecorder(output, "v", key="output_v"),
                 VarRecorder(hidden, genn_var="LambdaV", key="lambda_vh"),
                 VarRecorder(output, genn_var="LambdaI", key="lambda_io"),
                 VarRecorder(output, genn_var="LambdaV", key="lambda_vo"),
                 SpikeRecorder(input, key="input_spikes"),
                 SpikeRecorder(hidden, key="hidden_spikes"),
                 #OptimiserParamSchedule("alpha", alpha_schedule)
    ]
    metrics, cb_data  = compiled_net.train({input: in_spikes},
                                           {output: y_star},
                                           num_epochs=NUM_EPOCHS,
                                           callbacks=callbacks)
    end_time = perf_counter()
    print(f"Time = {end_time - start_time}s")

    #
    report_trials = [ 120, 121, 122, 123, 124, 125, 5*NUM_TRIALS ]
    report_trials = base + np.asarray(report_trials)
    plotcols = len(report_trials)
    fig, axes = plt.subplots(NUM_OUTPUT + 4, plotcols, sharex=True, sharey="row")
    t = np.arange(trial_steps)*DT
    base = 7*NUM_TRIALS
    for i in range(plotcols):
        error = []
        for c in range(NUM_OUTPUT):
            y = cb_data["output_v"][report_trials[i]][:,c]
            error.append(y - y_star[0][:,c])
            mse = np.sum(error[-1][50:trial_steps] * error[-1][50:trial_steps]) / len(error[-1][50:trial_steps])
            axes[c,i].set_title(f"Y{c} (MSE={mse:.2f})")
            axes[c,i].plot(t,y)
            axes[c,i].plot(t,y_star[0][:,c], linestyle="--")
            axes[c,i].set_title(f"trial {report_trials[i]}")
        axes[NUM_OUTPUT,i].scatter(cb_data["input_spikes"][0][report_trials[i]],
                                      cb_data["input_spikes"][1][report_trials[i]], s=1)
        axes[NUM_OUTPUT + 1,i].scatter(cb_data["hidden_spikes"][0][report_trials[i]],
                                          cb_data["hidden_spikes"][1][report_trials[i]], s=1)
        axes[NUM_OUTPUT + 2,i].plot(TRIAL_TIME-t,cb_data["lambda_vh"][report_trials[i]][:,c])
        
        axes[NUM_OUTPUT + 3,i].plot(TRIAL_TIME-t,cb_data["lambda_io"][report_trials[i]][:,c])
        
        axes[NUM_OUTPUT + 3,i].plot(TRIAL_TIME-t,cb_data["lambda_vo"][report_trials[i]][:,c])
        error = np.hstack(error)
        total_mse = np.sum(error * error) / len(error)
        print(f"{i}: Total MSE: {total_mse}")
    axes[NUM_OUTPUT,0].set_ylabel("Input spikes")
    axes[NUM_OUTPUT+1,0].set_ylabel("Hidden spikes")
    plt.show()
