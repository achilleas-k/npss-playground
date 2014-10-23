'''

Reproduces the main results (figure 3) of Koutsou, Christodoulou, Bugmann and
Kanev (2012).  This is not the code that produced the original results. That
code now exists somewhere in some archive I'm not willing to dig through.
The original simulations were not written in brian, so it is possible the
results produced from this file do not match the published results exactly.

WARNING: Running this for all (6) figures can take a while.

'''
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
#plt.plot(mean_slopes, '.', markersize=15)

    sync_conf = input_params["sync"]
    sync, jitter = zip(*sync_conf)
    jitter = [j*1000 for j in jitter]
    shape_ms2d = (len(np.unique(sync)), len(np.unique(jitter)))
    mean_slopes_2d = np.reshape(mean_slopes, (shape_ms2d)).transpose()
    plt.figure()
    extent = (min(sync), max(sync), min(jitter), max(jitter))
    plt.imshow(mean_slopes_2d, origin="lower", extent=extent, aspect="auto", 
               vmin=0, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label("$NPSS$")
    plt.xlabel("$S_{in}$")
    plt.ylabel("$\sigma_{in} (ms)$")
    plt.title("NPSS   $N_in$: {num_inp}, "
              "$f_{{out}}$: {fout} Hz, $w_{{in}}$: {weight} V".format(**input_params))
    plt.savefig("npss_{num_inp}_{fout}_{weight}_img.png".format(**input_params))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(sync, jitter, mean_slopes, c=mean_slopes)
    ax.set_xlim((min(sync), max(sync)))
    ax.set_xlabel("$S_{in}$")
    ax.set_ylim((min(jitter), max(jitter)))
    ax.set_ylabel("$\sigma_{in} (ms)$")
    ax.set_zlim((0, 1))
    ax.set_zlabel("$NPSS$")
    plt.title("NPSS   $N_in$: {num_inp}, "
              "$f_{{out}}$: {fout} Hz, $w_{{in}}$: {weight} V".format(**input_params))
    def rotate_scatterplot(nframe):
        ax.view_init(elev=10, azim=nframe)
        print("\rfinished frame %i" % (nframe), end="")
        sys.stdout.flush()
        return ax

    anim = animation.FuncAnimation(fig, rotate_scatterplot, frames=range(0, 360, 1))
    anim.save("npss_{num_inp}_{fout}_{weight}_3d.gif".format(**input_params),
              writer="imagemagick", fps=30)
    print()



neuron_params = {}
neuron_params['model'] = Equations("dV/dt = -V/(10*ms) : volt")
neuron_params['threshold'] = "V>(15*mvolt)"
neuron_params['reset'] = "V=(0*mvolt)"
neuron_params['refractory'] = 1*ms

sync_conf = [(sync, jitter*ms) for sync in np.arange(0, 1.1, 0.1)
                 for jitter in np.arange(0, 4.1, 0.5)]
input_params = {}
input_params['fout'] = 5*Hz
input_params['weight'] = 0.1*mV
input_params['num_inp'] = 100
input_params['sync'] = sync_conf
#input_params['fin'] = [10]*11
plot_figure(neuron_params, input_params)

input_params['fout'] = 100*Hz
input_params['weight'] = 0.2*mV
input_params['num_inp'] = 50
input_params['sync'] = sync_conf

plot_figure(neuron_params, input_params)
input_params['fout'] = 10*Hz
input_params['weight'] = 0.3*mV
input_params['num_inp'] = 60
input_params['sync'] = sync_conf
plot_figure(neuron_params, input_params)

input_params['fout'] = 70*Hz
input_params['weight'] = 0.5*mV
input_params['num_inp'] = 60
input_params['sync'] = sync_conf
plot_figure(neuron_params, input_params)

input_params['fout'] = 10*Hz
input_params['weight'] = 0.1*mV
input_params['num_inp'] = 200
input_params['sync'] = sync_conf
plot_figure(neuron_params, input_params)

input_params['fout'] = 400*Hz
input_params['weight'] = 0.5*mV
input_params['num_inp'] = 60
input_params['sync'] = sync_conf
plot_figure(neuron_params, input_params)
