# -*- coding: utf-8 -*-

def get(model_spec, **kwargs):
    import yaml, os
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.lib.io.file_io import FileIO
    from datetime import datetime
    from pinn.models.potential import potential_model
    from pinn.models.dipole import dipole_model
    from pinn.models.AC_neutral_QM9 import neutral_AC_dipole_model_QM9
    from pinn.models.AC_neutral_water import neutral_AC_dipole_model_water
    from pinn.models.AD_QM9 import AD_dipole_model_QM9 
    from pinn.models.AD_water import AD_dipole_model_water
    from pinn.models.AD_OS_water import AD_OS_dipole_model_water 
    from pinn.models.BC import BC_dipole_model
    from pinn.models.BC_R_QM9 import BC_R_dipole_model_QM9
    from pinn.models.BC_R_QM9_i1 import BC_R_dipole_model_QM9_i1
    from pinn.models.BC_R_water import BC_R_dipole_model_water
    from pinn.models.BC_OS_water import BC_OS_dipole_model_water
    from pinn.models.BC_OS_R_water import BC_OS_R_dipole_model_water
    from pinn.models.AC_AD import AC_AD_dipole_model
    from pinn.models.AC_AD_QM9 import AC_AD_dipole_model_QM9
    from pinn.models.AC_AD_water import AC_AD_dipole_model_water
    from pinn.models.AC_AD_neutral_QM9 import neutral_AC_AD_dipole_model_QM9 
    from pinn.models.AC_AD_neutral_water import neutral_AC_AD_dipole_model_water
    from pinn.models.AC_BC import AC_BC_dipole_model
    from pinn.models.AC_BC_neutral_QM9 import neutral_AC_BC_dipole_model_QM9
    from pinn.models.AC_BC_neutral_water import neutral_AC_BC_dipole_model_water
    from pinn.models.AC_BC_R_neutral_water import neutral_AC_BC_R_dipole_model_water
    from pinn.models.AD_BC_QM9 import AD_BC_dipole_model_QM9
    from pinn.models.AD_BC_water import AD_BC_dipole_model_water

    implemented_models = {
        'potential_model': potential_model,
        'dipole_model': dipole_model,
        'neutral_AC_dipole_model_QM9': neutral_AC_dipole_model_QM9,
        'neutral_AC_dipole_model_water': neutral_AC_dipole_model_water,
        'AD_dipole_model_QM9': AD_dipole_model_QM9,
        'AD_dipole_model_water': AD_dipole_model_water,
        'AD_OS_dipole_model_water': AD_OS_dipole_model_water, 
        'BC_dipole_model': BC_dipole_model,
        'BC_R_dipole_model_QM9': BC_R_dipole_model_QM9,
        'BC_R_dipole_model_QM9_i1': BC_R_dipole_model_QM9_i1,
        'BC_R_dipole_model_water': BC_R_dipole_model_water,
        'BC_OS_dipole_model_water': BC_OS_dipole_model_water,
        'BC_OS_R_dipole_model_water': BC_OS_R_dipole_model_water,
        'AC_AD_dipole_model': AC_AD_dipole_model,
        'AC_AD_dipole_model_QM9': AC_AD_dipole_model_QM9,
        'AC_AD_dipole_model_water':  AC_AD_dipole_model_water,
        'neutral_AC_AD_dipole_model_QM9': neutral_AC_AD_dipole_model_QM9,
        'neutral_AC_AD_dipole_model_water': neutral_AC_AD_dipole_model_water,
        'AC_BC_dipole_model': AC_BC_dipole_model,
        'neutral_AC_BC_dipole_model_QM9': neutral_AC_BC_dipole_model_QM9,
        'neutral_AC_BC_dipole_model_water': neutral_AC_BC_dipole_model_water,
        'neutral_AC_BC_R_dipole_model_water': neutral_AC_BC_R_dipole_model_water,
        'AD_BC_dipole_model_QM9': AD_BC_dipole_model_QM9,
        'AD_BC_dipole_model_water': AD_BC_dipole_model_water}

    if isinstance(model_spec, str):
        if tf.io.gfile.exists('{}/params.yml'.format(model_spec)):
            params_file = os.path.join(model_spec, 'params.yml')
            with FileIO(params_file, 'r') as f:
                model_spec = dict(yaml.load(f, Loader=yaml.Loader),
                                  model_dir=model_spec)
        elif tf.io.gfile.exists(model_spec):
            params_file = model_spec
            with FileIO(params_file, 'r') as f:
                model_spec = yaml.load(f, Loader=yaml.Loader)
        else:
            raise ValueError(f'{model_spec} does not seem to be a parameter file or model_dir')
    else:
        # we have a dictionary, write the model parameter
        model_dir = model_spec['model_dir']
        yaml.Dumper.ignore_aliases = lambda *args: True
        to_write = yaml.dump(model_spec)
        params_file = os.path.join(model_dir, 'params.yml')
        if not tf.io.gfile.isdir(model_dir):
            tf.io.gfile.makedirs(model_dir)
        if tf.io.gfile.exists(params_file):
            original = FileIO(params_file, 'r').read()
            if original != to_write:
                tf.io.gfile.rename(params_file, params_file+'.' +
                                   datetime.now().strftime('%y%m%d%H%M'))
        FileIO(params_file, 'w').write(to_write)
    model = implemented_models[model_spec['model']['name']](model_spec, **kwargs)
    return model
