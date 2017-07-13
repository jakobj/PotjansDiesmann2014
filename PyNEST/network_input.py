import numpy as np
import os

import nest


def create_network_input(net_dict, network_size, K, w_input, data_path):
    if nest.Rank() == 0:
        print('Background network created')

    # neuron_params = {
    #     'C_m': 200.0,
    #     'E_L': -52.0,
    #     'I_e': 0.0,
    #     'V_reset': -80.0,
    #     'V_th': -62.0,
    #     't_ref': 0.1,
    #     'tau_m': 20.,
    #     'tau_syn_ex': 5.,
    #     'tau_syn_in': 5.,
    # }

    neuron_params = {
        'C_m': 200.0,
        'E_L': -70.0,
        'I_e': 461.,
        'V_reset': -70.0,
        'V_th': -62.0,
        't_ref': 2.0,
        'tau_m': 20.,
        'tau_syn_ex': .5,
        'tau_syn_in': .5,
    }

    w_input_loc = net_dict['input_conn_params']['JE'] * neuron_params['tau_syn_ex'] / neuron_params['C_m'] * 1e6

    # compute derived params
    NE = int(net_dict['input_conn_params']['gamma'] * np.max(network_size))
    NI = int((1. - net_dict['input_conn_params']['gamma']) * np.max(network_size))
    KE = int(net_dict['input_conn_params']['gamma'] * np.max(K))
    KI = int((1. - net_dict['input_conn_params']['gamma']) * np.max(K))

    # I_rel = 3.5
    # I_th = ((net_dict['neuron_params']['V_th'] - net_dict['neuron_params']['E_L']) / net_dict['neuron_params']['tau_m'] * net_dict['neuron_params']['C_m'])

    pop_ex = nest.Create(net_dict['neuron_model'], NE)
    nest.SetStatus(
        pop_ex,
        neuron_params
        # {
        #     'tau_syn_ex': net_dict['neuron_params']['tau_syn_ex'],
        #     'tau_syn_in': net_dict['neuron_params']['tau_syn_in'],
        #     'E_L': net_dict['neuron_params']['E_L'],
        #     # 'E_L': net_dict['neuron_params']['V_th'] + 10.,
        #     'V_th': net_dict['neuron_params']['V_th'],
        #     'V_reset': net_dict['neuron_params']['V_reset'],
        #     't_ref': net_dict['neuron_params']['t_ref'],
        #     'I_e': I_rel * I_th,
        # }
    )

    pop_in = nest.Create(net_dict['neuron_model'], NI)
    nest.SetStatus(
        pop_in,
        neuron_params
        # {
        #     'tau_syn_ex': net_dict['neuron_params']['tau_syn_ex'],
        #     'tau_syn_in': net_dict['neuron_params']['tau_syn_in'],
        #     'E_L': net_dict['neuron_params']['E_L'],
        #     # 'E_L': net_dict['neuron_params']['V_th'] + 10.,
        #     'V_th': net_dict['neuron_params']['V_th'],
        #     'V_reset': net_dict['neuron_params']['V_reset'],
        #     't_ref': net_dict['neuron_params']['t_ref'],
        #     'I_e': I_rel * I_th,
        # }
    )

    for idx in pop_ex + pop_in:
        nest.SetStatus([idx], {'V_m': np.random.uniform(net_dict['neuron_params']['V_reset'] - 30., net_dict['neuron_params']['V_th'] + 1.)})

    conn_params_ex = {
        'rule': 'fixed_indegree',
        'indegree': KE,
        'autapses': False,
        'multapses': False,
    }
    syn_params_ex = {
        'model': 'static_synapse',
        'weight': w_input_loc,
        'delay': net_dict['input_conn_params']['delay'],
    }

    conn_params_in = {
        'rule': 'fixed_indegree',
        'indegree': KI,
        'autapses': False,
        'multapses': False,
    }
    syn_params_in = {
        'model': 'static_synapse',
        'weight': -1. * net_dict['input_conn_params']['g'] * w_input_loc,
        'delay': net_dict['input_conn_params']['delay'],
    }

    nest.Connect(pop_ex, pop_ex, conn_params_ex, syn_params_ex)
    nest.Connect(pop_ex, pop_in, conn_params_ex, syn_params_ex)
    nest.Connect(pop_in, pop_ex, conn_params_in, syn_params_in)
    nest.Connect(pop_in, pop_in, conn_params_in, syn_params_in)

    sd = nest.Create('spike_detector', 1, {'label': os.path.join(data_path, 'noise'), 'to_file': True})
    nest.Connect(pop_ex, sd)
    nest.Connect(pop_in, sd)
    net_dict['sd_noise'] = sd

    m = nest.Create('multimeter', 1, {'label': 'noise_mem', 'to_file': True, 'record_from': ['V_m']})
    nest.Connect(m, pop_ex)

    return [pop_ex, pop_in]


def connect_network_input(net_dict, pops, poisson, K_ext, w_input):
    if nest.Rank() == 0:
        print('Background network is connected')
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
