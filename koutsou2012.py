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
    input_configs = input_params.get("configs")
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

input_configs = [(sync, jitter*ms) for sync in np.arange(0, 1.1, 0.1)
                 for jitter in np.arange(0, 4.1, 0.5)]
input_configs = [(sync, jitter*ms) for sync in np.arange(0, 1.1, 0.5)
                 for jitter in np.arange(0, 4.1, 2.0)]
input_params = {}
input_params['fout'] = 50*Hz
input_params['weight'] = 0.5*mV
input_params['num_inp'] = 50
input_params['configs'] = input_configs
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
#plt.plot(mean_slopes, '.', markersize=15)

sync, jitter = zip(*input_configs)
shape_ms2d = (len(np.unique(sync)), len(np.unique(jitter)))
mean_slopes_2d = np.reshape(mean_slopes, (shape_ms2d)).transpose()
plt.figure()
extent = (min(sync), max(sync), min(jitter), max(jitter))
plt.imshow(mean_slopes_2d, origin="lower", extent=extent, aspect="auto")
cbar = plt.colorbar()
cbar.set_label("$NPSS$")
plt.xlabel("$S_{in}$")
plt.ylabel("$\sigma_{in}$")
plt.savefig("npss_{num_inp}_{fout}_{weight}_img.png".format(**input_params))


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter3D(sync, jitter, mean_slopes, c=mean_slopes)
#ax.set_xlim( # TODO: Set limits
ax.set_xlabel("$S_{in}$")
ax.set_ylabel("$\sigma_{in}$")
ax.set_zlabel("$NPSS$")
def rotate_scatterplot(nframe):
    ax.view_init(elev=10, azim=nframe)
    print("\rfinished frame %i" % (nframe))
    sys.stdout.flush()
    return ax

anim = animation.FuncAnimation(fig, rotate_scatterplot, frames=range(0, 360, 1))
anim.save("npss_{num_inp}_{fout}_{weight}_3d.gif".format(**input_params),
          writer="imagemagick", fps=30)
