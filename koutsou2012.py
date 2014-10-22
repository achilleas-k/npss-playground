'''
Reproduces the results of Koutsou, Christodoulou, Bugmann and Kanev (2012).
This is not the code that produced the original results. That code now exists
somewhere in some archive I'm not willing to dig through.
'''
import brian
from brian import (Network, NeuronGroup, Equations, Connection,
                   StateMonitor, SpikeMonitor,
                   second, ms, volt, mV, Hz)
import gc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spikerlib as sl


def setup_sims(neuron_params, input_params, duration):
    fin = input_params.get("fin")
    fout = input_params.get("fout")
    weight = input_params.get("weight")
    num_inp = input_params.get("num_inp")
    input_configs = [(sync, jitter*ms) for sync in np.arange(0, 1.1, 0.1)
                     for jitter in np.arange(0, 4.1, 0.5)]
    if fin is None:
        fin = sl.tools.calibrate_frequencies(neuron_params,
                                             N_in=num_inp, w_in=weight,
                                             f_out=fout,
                                             input_configs=input_configs)


    brian.clear(True)
    gc.collect()
    brian.defaultclock.reinit()
    neurons = NeuronGroup(N=len(input_configs), **neuron_params)
    simulation = Network(neurons)
    input_groups = []
    for idx, (inrate, (sync, jitter)) in enumerate(zip(fin, input_configs)):
        inp_grp = sl.tools.fast_synchronous_input_gen(num_inp,
                                                      inrate*Hz,
                                                      sync, jitter,
                                                      duration)
        simulation.add(inp_grp)
        inp_conn = Connection(inp_grp, neurons[idx], state='V', weight=weight)
        input_groups.append(inp_grp)
        simulation.add(inp_conn)
    tracemon = StateMonitor(neurons, 'V', record=True)
    spikemon = SpikeMonitor(neurons)
    inputmons = [SpikeMonitor(igrp) for igrp in input_groups]
    simulation.add(tracemon, spikemon, inputmons)
    monitors = {"inputs": inputmons, "outputs": spikemon, "traces": tracemon}
    return simulation, monitors


duration = 5*second

neuron_params = {}
neuron_params['model'] = Equations("dV/dt = -V/(10*ms) : volt")
neuron_params['threshold'] = "V>(15*mvolt)"
neuron_params['reset'] = "V=(0*mvolt)"
neuron_params['refractory'] = 1*ms

input_params = {}
input_params['fout'] = 50*Hz
input_params['weight'] = 0.5*mV
input_params['num_inp'] = 50
#input_params['fin'] = [10]*11

simulation, monitors = setup_sims(neuron_params, input_params, duration)


print("Running simulation for %.2f seconds ..." % duration)
simulation.run(duration)
print("Simulation finished")

tracemon = monitors["traces"]
spikemon = monitors["outputs"]
inputmon = monitors["inputs"]

mean_slopes = []
for trace, spikes in zip(tracemon.values, spikemon.spiketimes.values()):
    slopes = sl.tools.npss(trace, spikes, 0*mV, 15*mV, 10*ms, 2*ms)
    mslope = np.mean(slopes)
    mean_slopes.append(mslope)
plt.plot(mean_slopes, '.', markersize=15)


input_configs = [(sync, jitter*ms) for sync in np.arange(0, 1.1, 0.1)
                 for jitter in np.arange(0, 4.1, 0.5)]
sync, jitter = zip(*input_configs)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter3D(sync, jitter, mean_slopes, c=mean_slopes)
