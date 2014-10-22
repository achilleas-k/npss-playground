'''
Reproduces the results of Koutsou, Christodoulou, Bugmann and Kanev (2012).
This is not the code that produced the original results. That code now exists
somewhere in some archive I'm not willing to dig through.
'''
from brian import (Network, NeuronGroup, Equations, Connection,
                   StateMonitor, SpikeMonitor,
                   second, ms, volt, mV, Hz)
import numpy as np
import matplotlib.pyplot as plt
import spikerlib as sl


def setup_sims(neuron_params, input_params, duration):
    fin = input_params.get("fin")
    fout = input_params.get("fout")
    weight = input_params.get("weight")
    num_inp = input_params.get("num_inp")
    input_configs = [(s, 0*ms) for s in np.arange(0, 1.1, 0.1)]
    if fin is None:
        fin = sl.tools.calibrate_frequencies(neuron_params,
                                             N_in=num_inp, w_in=weight,
                                             f_out=fout,
                                             input_configs=input_configs)


    neurons = NeuronGroup(N=len(input_configs), **neuron_params)
    simulation = Network(neurons)
    for idx, (inrate, (sync, jitter)) in enumerate(zip(fin, input_configs)):
        inp_grp = sl.tools.fast_synchronous_input_gen(num_inp,
                                                      inrate*Hz,
                                                      sync, jitter,
                                                      duration)
        simulation.add(inp_grp)
        inp_conn = Connection(inp_grp, neurons[idx], state='V', weight=weight)
        simulation.add(inp_conn)
    tracemon = StateMonitor(neurons, 'V', record=True)
    spikemon = SpikeMonitor(neurons)
    simulation.add(tracemon, spikemon)
    return simulation

duration = 10*second

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

simulation = setup_sims(neuron_params, input_params, duration)

print("Running simulation for %.2f seconds ...")
simulation.run(duration)


