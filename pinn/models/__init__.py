# -*- coding: utf-8 -*-

def get(model_spec, **kwargs):
    import yaml, os
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.lib.io.file_io import FileIO
    from datetime import datetime
    from pinn.models.potential import potential_model
    from pinn.models.dipole import dipole_model
    from pinn.models.atomic_dipole_reg import atomic_dipole_model_reg
    from pinn.models.atomic_dipole import atomic_dipole_model
    from pinn.models.atomic_dipole_QM9 import atomic_dipole_model_QM9
    from pinn.models.dipole_neutral_QM9 import neutral_dipole_model_QM9
    from pinn.models.dipole_neutral_water import neutral_dipole_model_water
    from pinn.models.combined_dipole import combined_dipole_model
    from pinn.models.combined_dipole_neutral_QM9 import neutral_combined_dipole_model_QM9
    from pinn.models.combined_dipole_neutral_water import neutral_combined_dipole_model_water
    from pinn.models.combined_dipole_neutral_reg_water import neutral_combined_dipole_model_reg_water
    from pinn.models.PaiNN_dipole import PaiNN_dipole_model
    from pinn.models.PaiNN_dipole_QM9 import PaiNN_dipole_model_QM9
    from pinn.models.PaiNN_dipole_neutral_QM9 import neutral_PaiNN_dipole_model_QM9
    from pinn.models.PaiNN_dipole_water import PaiNN_dipole_model_water
    from pinn.models.PaiNN_dipole_neutral_water import neutral_PaiNN_dipole_model_water
    from pinn.models.combined_dipole_oxidation_water import oxidation_combined_dipole_model_water
    from pinn.models.PaiNN_dipole_oxidation_water import oxidation_PaiNN_dipole_model_water
    from pinn.models.p3_dipole_QM9 import p3_dipole_model_QM9
    from pinn.models.p3_dipole_water import p3_dipole_model_water
    from pinn.models.AD_BC_dipole_water import AD_BC_dipole_model_water
    from pinn.models.AD_BC_dipole_QM9 import AD_BC_dipole_model_QM9
    implemented_models = {
        'potential_model': potential_model,
        'dipole_model': dipole_model,
        'neutral_dipole_model_QM9': neutral_dipole_model_QM9,
        'neutral_dipole_model_water': neutral_dipole_model_water,
        'atomic_dipole_model': atomic_dipole_model,
        'atomic_dipole_model_QM9': atomic_dipole_model_QM9,
        'atomic_dipole_model_reg': atomic_dipole_model_reg,
        'combined_dipole_model': combined_dipole_model,
        'neutral_combined_dipole_model_QM9': neutral_combined_dipole_model_QM9,
        'neutral_combined_dipole_model_water': neutral_combined_dipole_model_water,
        'neutral_combined_dipole_model_reg_water': neutral_combined_dipole_model_reg_water,
        'PaiNN_dipole_model': PaiNN_dipole_model,
        'PaiNN_dipole_model_QM9': PaiNN_dipole_model_QM9,
        'neutral_PaiNN_dipole_model_QM9': neutral_PaiNN_dipole_model_QM9,
        'PaiNN_dipole_model_water': PaiNN_dipole_model_water,
        'neutral_PaiNN_dipole_model_water': neutral_PaiNN_dipole_model_water,
        'oxidation_combined_dipole_model_water': oxidation_combined_dipole_model_water,
        'oxidation_PaiNN_dipole_model_water': oxidation_PaiNN_dipole_model_water,
        'p3_dipole_model_QM9': p3_dipole_model_QM9,
        'p3_dipole_model_water': p3_dipole_model_water,
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
