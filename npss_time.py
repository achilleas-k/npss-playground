from __future__ import print_function
import brian
from brian import (Network, NeuronGroup, Equations, Connection,
                   StateMonitor, SpikeMonitor, SpikeGeneratorGroup,
                   raster_plot,
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

def simple_profiles(neuron_params, input_params, duration):
    fin = input_params.get("fin")
    fout = input_params.get("fout")
    weight = input_params.get("weight")
    num_inp = input_params.get("num_inp")
    sync_configs = input_params.get("sync")
    if fin is None:
        fin = [fout]*len(sync_configs)
        print("Warning: simple_profiles called with fin = None. "
              "Setting fin = fout.")
    brian.clear(True)
    gc.collect()
    brian.defaultclock.reinit()
    neurons = NeuronGroup(N=len(sync_configs), **neuron_params)
    simulation = Network(neurons)
    input_groups = []
    for idx, (inrate, (sync, jitter)) in enumerate(zip(fin, sync_configs)):
        input_trains = []
        shift_train = np.arange(0*second, duration, 1/inrate)
        Nsync = int(num_inp*sync)
        Nrand = num_inp-Nsync
        for irnd in range(Nrand):
            shift = irnd/inrate/Nrand
            shifted = shift_train+shift
            input_trains.append(shifted)
        sync_train = np.arange(1/inrate, duration, 1/inrate)
        for isyn in range(Nsync):
            input_trains.append(sync_train)
        # make tuples
        input_tuples = []
        for in_idx, train in enumerate(input_trains):
            input_tuples.extend([(in_idx, st) for st in train])
        input_tuples = sorted(input_tuples, key=lambda pair: pair[1])
        inp_grp = SpikeGeneratorGroup(N=num_inp, spiketimes=input_tuples)
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


def compare_temporal_profiles(neuron_params, input_params):
    duration = 1*second
    #simulation, monitors = setup_sims(neuron_params, input_params, duration)
    simulation, monitors = simple_profiles(neuron_params, input_params, duration)
    print("Running simulation for %.2f seconds ..." % duration)
    simulation.run(duration, report="stderr")
    print("Simulation finished")

    print("Creating figures ...")
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
    for sc, trace in zip(sync_conf, tracemon.values):
        plt.figure()
        plt.title("S: %.2f  J: %.2f" % (sc[0], sc[1]))
        plt.plot(tracemon.times, trace)
        file_params = input_params.copy()
        file_params["sync"] = sc[0]
        file_params["jitter"] = sc[1]
        plt.savefig("{num_inp}_{fout}_{weight}"
                    "_{sync}_{jitter}_trace.png".format(**file_params))
        plt.close()
    for sc, inp, out, slopes in zip(sync_conf, inputmon,
                                    spikemon.spiketimes.values(), all_slopes):
        inspikes = inp.spiketimes.values()
        # convolve or low-pass filter
        psth_bin = 2*ms
        psth = sl.tools.PSTH(inspikes, bin=psth_bin, duration=duration)
        plt.figure()
        plt.title("S: %.2f  J: %.2f" % (sc[0], sc[1]))
        plt.subplot(2,1,1)
        t = np.arange(0*second, duration, psth_bin)
        plt.plot(t, psth)
        plt.subplot(2,1,2)
        plt.plot(out, slopes)
        file_params = input_params.copy()
        file_params["sync"] = sc[0]
        file_params["jitter"] = sc[1]
        plt.savefig("{num_inp}_{fout}_{weight}"
                    "_{sync}_{jitter}_npss.png".format(**file_params))
        plt.close()
        plt.figure()
        plt.title("S: %.2f  J: %.2f" % (sc[0], sc[1]))
        for height, spiketrain in enumerate(inspikes):
            plt.plot(spiketrain, np.zeros_like(spiketrain)+height, 'b.')
        plt.savefig("{num_inp}_{fout}_{weight}"
                    "_{sync}_{jitter}_inp.png".format(**file_params))
        plt.close()
    print("Done!")


neuron_params = {}
neuron_params['model'] = Equations("dV/dt = -V/(10*ms) : volt")
neuron_params['threshold'] = "V>(15*mvolt)"
neuron_params['reset'] = "V=(0*mvolt)"
neuron_params['refractory'] = 1*ms

sync_conf = [(sync, jitter*ms) for sync in np.arange(0, 1.1, 0.25)
                 for jitter in np.arange(0, 4.1, 1.0)]
sync_conf = [(sync, 0*ms) for sync in np.arange(0, 1.1, 0.1)]
input_params = {}
input_params['fout'] = 5*Hz
input_params['weight'] = 0.5*mV
input_params['num_inp'] = 50
input_params['sync'] = sync_conf
input_params['fin'] = [10*Hz]*len(sync_conf)
compare_temporal_profiles(neuron_params, input_params)
