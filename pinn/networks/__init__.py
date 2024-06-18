# -*- coding: utf-8 -*-

def get(network_spec):
    """Retrieve a PiNN network

    Args:
       network_spec: serialized specification of network, or a Keras model.
    """
    import tensorflow as tf
    from pinn.networks.pinet import PiNet
    from pinn.networks.bpnn import BPNN
    from pinn.networks.lj import LJ
    from pinn.networks.pinet2 import PiNet2
    from pinn.networks.pinet2_modularized import PiNet2_module
    from pinn.networks.pinet2_p5_dot import PiNet2P5Dot
    from pinn.networks.pinet2_p5_dot_i import PiNet2P5Dot_i
    from pinn.networks.pinet2_simple import PiNet2_simple
    from pinn.networks.pinet2_norm import PiNet2_norm
    from pinn.networks.pinet2_norm_simple import PiNet2_norm_simple
    from pinn.networks.pinet2_norm_simple_nosum import PiNet2_norm_simple_nosum
    from pinn.networks.pinet2_norm_p3 import PiNet2_norm_p3
    from pinn.networks.pinet2_norm_p3_i import PiNet2_norm_p3_i
    from pinn.networks.pinet2_energy import PiNet2_energy
    from pinn.networks.pinet2_energy_norm import PiNet2_energy_norm 
    from pinn.networks.pinet2_energy_minmax import PiNet2_energy_minmax
    from pinn.networks.pinet2_minmax import PiNet2_minmax
    from pinn.networks.pinet2_minmax_i3 import PiNet2_minmax_i3
    from pinn.networks.pinet2_minmax_i3_simple import PiNet2_minmax_i3_simple
    implemented_networks = {
        'PiNet': PiNet,
        'BPNN': BPNN,
        'LJ': LJ,
        'PiNet2': PiNet2,
        'PiNet2_module': PiNet2_module,
        'PiNet2P5Dot': PiNet2P5Dot,
        'PiNet2P5Dot_i': PiNet2P5Dot_i,
        'PiNet2_simple': PiNet2_simple,
        'PiNet2_norm': PiNet2_norm,
        'PiNet2_norm_simple': PiNet2_norm_simple,
        'PiNet2_norm_simple_nosum': PiNet2_norm_simple_nosum,
        'PiNet2_norm_p3': PiNet2_norm_p3,
        'PiNet2_norm_p3_i': PiNet2_norm_p3_i,
        'PiNet2_minmax': PiNet2_minmax,
        'PiNet2_minmax_i3': PiNet2_minmax_i3,
        'PiNet2_minmax_i3_simple': PiNet2_minmax_i3_simple,
        'PiNet2_energy': PiNet2_energy,
        'PiNet2_energy_norm': PiNet2_energy_norm,
        'PiNet2_energy_minmax': PiNet2_energy_minmax
    }
    if isinstance(network_spec, tf.keras.Model):
        return network_spec
    else:
        return  implemented_networks[network_spec['name']](
            **network_spec['params'])
