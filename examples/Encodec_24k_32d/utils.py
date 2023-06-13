import importlib
import random
import numpy as np
import torch
import warnings
import os
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import time
import torch.utils.tensorboard as tensorboard
from torch import distributed as dist
import sys
import yaml
import json
def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

def is_primary():
    return get_rank() == 0

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0

    return dist.get_rank()

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def save_config_to_yaml(config, path):
    assert path.endswith('.yaml')
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
        f.close()

def save_dict_to_json(d, path, indent=None):
    json.dump(d, open(path, 'w'), indent=indent)

def load_dict_from_json(path):
    return json.load(open(path, 'r'))

def write_args(args, path):
    args_dict = dict((name, getattr(args, name)) for name in dir(args)if not name.startswith('_'))
    with open(path, 'a') as args_file:
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        args_file.write('\n==> args:\n')
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
        args_file.close()

class Logger(object):
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.is_primary = is_primary()
        
        if self.is_primary:
            os.makedirs(self.save_dir, exist_ok=True)
            
            # save the args and config
            self.config_dir = os.path.join(self.save_dir, 'configs')
            os.makedirs(self.config_dir, exist_ok=True)
            file_name = os.path.join(self.config_dir, 'args.txt')
            write_args(args, file_name)

            log_dir = os.path.join(self.save_dir, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self.text_writer = open(os.path.join(log_dir, 'log.txt'), 'a') # 'w')
            if args.tensorboard:
                self.log_info('using tensorboard')
                self.tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) # tensorboard.SummaryWriter(log_dir=log_dir)
            else:
                self.tb_writer = None
            

    def save_config(self, config):
        if self.is_primary:
            save_config_to_yaml(config, os.path.join(self.config_dir, 'config.yaml'))

    def log_info(self, info, check_primary=True):
        if self.is_primary or (not check_primary):
            print(info)
            if self.is_primary:
                info = str(info)
                time_str = time.strftime('%Y-%m-%d-%H-%M')
                info = '{}: {}'.format(time_str, info)
                if not info.endswith('\n'):
                    info += '\n'
                self.text_writer.write(info)
                self.text_writer.flush()

    def add_scalar(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(**kargs)

    def add_scalars(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_scalars(**kargs)

    def add_image(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_image(**kargs)

    def add_images(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_images(**kargs)


    def close(self):
        if self.is_primary:
            self.text_writer.close()
            self.tb_writer.close()

