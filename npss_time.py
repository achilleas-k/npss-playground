from __future__ import print_function
import brian
from brian import (Network, NeuronGroup, Equations, Connection,
                   StateMonitor, SpikeMonitor,
                   second, ms, volt, mV, Hz)
import gc
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import spikerlib as sl
import sys


def setup_sims(neuron_params, input_params, duration):
    fin = input_params.get("fin")
    fout = input_params.get("fout")
    weight = input_params.get("weight")
    num_inp = input_params.get("num_inp")
    sync_configs = input_params.get("sync")
    if fin is None:
        fin = sl.tools.calibrate_frequencies(neuron_params,
                                             N_in=num_inp, w_in=weight,
                                             f_out=fout,
                                             synchrony_conf=sync_configs)


    brian.clear(True)
    gc.collect()
    brian.defaultclock.reinit()
    neurons = NeuronGroup(N=len(sync_configs), **neuron_params)
    simulation = Network(neurons)
    input_groups = []
    for idx, (inrate, (sync, jitter)) in enumerate(zip(fin, sync_configs)):
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


def plot_figure(neuron_params, input_params):
    duration = 10*second
    simulation, monitors = setup_sims(neuron_params, input_params, duration)
    print("Running simulation for %.2f seconds ..." % duration)
    simulation.run(duration, report="stderr")
    print("Simulation finished")

    tracemon = monitors["traces"]
    spikemon = monitors["outputs"]
    inputmon = monitors["inputs"]

    all_slopes = []
    for trace, spikes in zip(tracemon.values, spikemon.spiketimes.values()):
        if len(spikes) < 2:
            all_slopes.append([])
            continue
        slopes = sl.tools.npss(trace, spikes, 0*mV, 15*mV, 10*ms, 2*ms)
        all_slopes.append(slopes)

    sync_conf = input_params["sync"]
    for sc, inp, out, slopes in zip(sync_conf, inputmon,
                                    spikemon.spiketimes.values(), all_slopes):
        inspikes = inp.spiketimes.values()
        # convolve or low-pass filter
        psth = sl.tools.PSTH(inspikes, bin=1*ms, duration=duration)
        plt.figure()
        plt.subplot(2,1,1)
        t = np.arange(0*second, duration, 1*ms)
        plt.plot(t, psth)
        plt.subplot(2,1,2)
        plt.plot(out, slopes)
        file_params = input_params.copy()
        file_params["sync"] = sc[0]
        file_params["jitter"] = sc[1]
        plt.savefig("npss_{num_inp}_{fout}_{weight}"
                    "_{sync}_{jitter}_img.png".format(**file_params))



neuron_params = {}
neuron_params['model'] = Equations("dV/dt = -V/(10*ms) : volt")
neuron_params['threshold'] = "V>(15*mvolt)"
neuron_params['reset'] = "V=(0*mvolt)"
neuron_params['refractory'] = 1*ms

sync_conf = [(sync, jitter*ms) for sync in np.arange(0, 1.1, 0.25)
                 for jitter in np.arange(0, 4.1, 1.0)]
input_params = {}
input_params['fout'] = 5*Hz
input_params['weight'] = 0.1*mV
input_params['num_inp'] = 100
input_params['sync'] = sync_conf
input_params['fin'] = [10]*len(sync_conf)
plot_figure(neuron_params, input_params)
