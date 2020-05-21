# -*- coding: utf-8 -*-
"""Basic functions for PiNN models"""
import tensorflow as tf
from pinn.utils import pi_named

def export_model(model_fn):
    # default parameters for all models
    from pinn.optimizers import default_adam
    default_params = {'optimizer': default_adam}

    def pinn_model(params, **kwargs):
        import yaml, os
        import numpy as np
        from tensorflow.python.lib.io.file_io import FileIO
        from datetime import datetime

        if isinstance(params, str):
            model_dir = params
            assert tf.io.gfile.exists('{}/params.yml'.format(model_dir)),\
                "Parameters files not found."
            with FileIO(os.path.join(model_dir, 'params.yml'), 'r') as f:
                params = yaml.load(f, Loader=yaml.Loader)
        else:
            model_dir = params['model_dir']
            yaml.Dumper.ignore_aliases = lambda *args: True
            to_write = yaml.dump(params)
            params_path = os.path.join(model_dir, 'params.yml')
            if not tf.io.gfile.isdir(model_dir):
                tf.io.gfile.makedirs(model_dir)
            if tf.io.gfile.exists(params_path):
                original = FileIO(params_path, 'r').read()
                if original != to_write:
                    tf.io.gfile.rename(params_path, params_path+'.' +
                              datetime.now().strftime('%y%m%d%H%M'))
            FileIO(params_path, 'w').write(to_write)

        params_tmp = default_params.copy()
        params_tmp.update(params)
        params = params_tmp
        model = tf.estimator.Estimator(model_fn=model_fn, params=params,
                                       model_dir=model_dir, **kwargs)
        return model
    return pinn_model

class MetricsCollector():
    def __init__(self, mode):
        self.mode = mode
        self.LOSS = 0
        self.ERROR = []
        self.METRICS = {}

    def add_error(self, tag, data, pred, mask=None, weight=None,
                  use_error=True, log_error=True, log_hist=True):
        """Add the error

        Args:
            tag (str): name of the error.
            data (tensor): data label tensor.
            pred (tensor): prediction tensor.
            mask (tensor): default to None (no mask).
            weight (tensor): default to None (no weight).
            mode: ModeKeys.TRAIN or ModeKeys.EVAL.
            return_error (bool): return error vector (for usage with Kalman Filter).
            log_loss (bool): log the error and loss function.
            log_hist (bool): add data and predicition histogram to log.
            log_mae (bool): add the mean absolute error to log.
            log_rmse (bool): add the root mean squared error to log.
        """
        error = data - pred
        loss = tf.reduce_mean(error**2 * weight)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            if log_hist:
                tf.compat.v1.summary.histogram(f'{tag}_DATA', data)
                tf.compat.v1.summary.histogram(f'{tag}_PRED', pred)
                tf.compat.v1.summary.histogram(f'{tag}_ERROR', error)
            if log_error:
                mae = tf.reduce_mean(tf.abs(error))
                rmse = tf.sqrt(tf.reduce_mean(error**2))
                tf.compat.v1.summary.scalar(f'{tag}_MAE', mae)
                tf.compat.v1.summary.scalar(f'{tag}_RMSE', rmse)
            if use_error:
                tf.compat.v1.summary.scalar(f'{tag}_LOSS', loss)
                self.ERROR.append(error)
                self.LOSS += loss

        if self.mode == tf.estimator.ModeKeys.EVAL:
            if log_error:
                self.METRICS[f'METRICS/{tag}_MAE'] = tf.compat.v1.metrics.mean_absolute_error(data, pred)
                self.METRICS[f'METRICS/{tag}_RMSE'] = tf.compat.v1.metrics.root_mean_squared_error(data, pred)
            if use_error:
                self.METRICS[f'METRICS/{tag}_LOSS'] = tf.compat.v1.metrics.mean(loss)
                self.LOSS += loss


@pi_named('TRAIN_OP')
def get_train_op(optimizer, loss, error, network):
    """
    Args:
        optimizer: a PiNN optimizer config.
        params: optimizer parameters.
        loss: scalar loss function.
        error: a list of error vectors (reserved for EKF).
        network: a PiNN network instance.
    """
    from tensorflow.keras.optimizers import get

    optimizer = get(optimizer)
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    tvars = network.trainable_variables
    grads = tf.gradients(loss, tvars)

    return optimizer.apply_gradients(zip(grads, tvars))