import numpy as np
import os

import nest


def create_poisson_pool(net_dict, poisson_pool_size, data_path):
    if nest.Rank() == 0:
        print('Poisson background pool created')
    poisson = []

    poisson_ex = nest.Create('poisson_generator', 1, {'rate': net_dict['network_rate']})
    parrots_ex = nest.Create('parrot_neuron', int(net_dict['input_conn_params']['gamma'] * np.max(poisson_pool_size)))  # use parrots to emulate multiple Poisson sources
    nest.Connect(poisson_ex, parrots_ex, {'rule': 'all_to_all'}, {'weight': 1., 'delay': 1.})
    poisson.append(parrots_ex)

    poisson_in = nest.Create('poisson_generator', 1, {'rate': net_dict['network_rate']})
    parrots_in = nest.Create('parrot_neuron', int((1. - net_dict['input_conn_params']['gamma']) * np.max(poisson_pool_size)))  # use parrots to emulate multiple Poisson sources
    nest.Connect(poisson_in, parrots_in, {'rule': 'all_to_all'}, {'weight': 1., 'delay': 1.})
    poisson.append(parrots_in)

    sd = nest.Create('spike_detector', 1, {'label': os.path.join(data_path, 'noise'), 'to_file': True})
    nest.Connect(parrots_ex, sd)
    nest.Connect(parrots_in, sd)
    net_dict['sd_noise'] = sd

    return poisson


def connect_poisson_pool(net_dict, pops, poisson, K_ext, w_input):
    if nest.Rank() == 0:
        print('Poisson pool background is connected')
    for i, target_pop in enumerate(pops):
        conn_dict_poisson = {
            'rule': 'fixed_indegree',
            'indegree': int(net_dict['input_conn_params']['gamma'] * K_ext[i]),
            'multapses': False,
        }
        syn_dict_poisson = {
            'model': 'static_synapse',
            'weight': w_input,
            'delay': net_dict['poisson_delay'] - 1.,
        }
        nest.Connect(
            poisson[0], target_pop,
            conn_spec=conn_dict_poisson,
            syn_spec=syn_dict_poisson
        )
        conn_dict_poisson['indegree'] = int((1. - net_dict['input_conn_params']['gamma']) * K_ext[i])
        syn_dict_poisson['weight'] = -1. * net_dict['input_conn_params']['g'] * w_input
        nest.Connect(
            poisson[1], target_pop,
            conn_spec=conn_dict_poisson,
            syn_spec=syn_dict_poisson
        )
