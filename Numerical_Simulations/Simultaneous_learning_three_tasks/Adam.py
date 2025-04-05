import numpy as np
import os
import pickle
import dill
import time


class pickle_depository:
    def __init__(self, save_path):
        self.save_path = save_path

    def _load(self, default=None):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = {} if default is None else default
        return data

    def _save(self, data):
        while True:
            try:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(data, f)
                break
            except Exception as e:
                print(e)
                time.sleep(1)

    def _clear(self, default=None):
        data = {} if default is None else default
        self._save(data)


class dill_depository:
    def __init__(self, save_path):
        self.save_path = save_path

    def _load(self, default=None):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as f:
                data = dill.load(f)
        else:
            data = {} if default is None else default
        return data

    def _save(self, data):
        with open(self.save_path, 'wb') as f:
            dill.dump(data, f)

    def _clear(self, default=None):
        data = {} if default is None else default
        self._save(data)


class DiskDict(pickle_depository):
    def __init__(self, save_path):
        super().__init__(save_path)

    def __setitem__(self, key, value):
        data = self._load()
        data[key] = value
        self._save(data)

    def __getitem__(self, key):
        return self._load()[key]

    def pop(self, key):
        assert key in self.keys(), f'{key} not existed'
        data = self._load()
        value = data.pop(key)
        self._save(data)
        return value

    def keys(self):
        return self._load().keys()

class adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        '''
        lr: learning rate
        '''
        self.data = {'gradient': {}}
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def record_gradient(self, gs, iter_idx, key='default'):
        '''
        record gradient
        '''
        gs = np.array(gs)
        if key not in self.data['gradient'].keys():
            self.data['gradient'][key] = {}
        self.data['gradient'][key][iter_idx] = gs

    def delta_theta(self, iter_idx, key='default'):
        if 'm' not in self.data:
            self.data['m'] = {}
        if key not in self.data['m']:
            self.data['m'][key] = {}
            self.data['m'][key][-1] = 0
        if 'v' not in self.data:
            self.data['v'] = {}
        if key not in self.data['v']:
            self.data['v'][key] = {}
            self.data['v'][key][-1] = 0
        m = self.beta_1 * self.data['m'][key][iter_idx - 1] + (
            1 - self.beta_1) * self.data['gradient'][key][iter_idx]
        v = self.beta_2 * self.data['v'][key][iter_idx - 1] + (
            1 - self.beta_2) * (self.data['gradient'][key][iter_idx]**2)
        self.data['m'][key][iter_idx] = m
        self.data['v'][key][iter_idx] = v
        m_cap = m / (1 - (self.beta_1**(iter_idx + 1)))
        v_cap = v / (1 - (self.beta_2**(iter_idx + 1)))
        return -(self.lr * m_cap) / (np.sqrt(v_cap) + 1e-8)

    def next(self, gs, iter_idx, key='default'):
        '''
        record gradient and return delta_theta
        theta[iter_idx + 1] = theta[iter_idx] + delta_theta
        '''
        self.record_gradient(gs, iter_idx, key=key)
        return self.delta_theta(iter_idx, key=key)


class adam_log(pickle_depository):
    def __init__(self, save_path, lr=0.1, beta_1=0.9, beta_2=0.999):
        super().__init__(save_path)
        data = self._load()
        data['lr'] = lr
        self._save(data)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def record_gradient(self, gs, key, iter_idx):
        '''
        record gradient
        '''
        gs = np.array(gs)
        data = self._load()
        if 'gradient' not in data.keys():
            data['gradient'] = {}
        if key not in data['gradient'].keys():
            data['gradient'][key] = {}
        if iter_idx is None:
            iter_idx = len(data['gradient'][key])
        data['gradient'][key][iter_idx] = gs
        self._save(data)

    def delta_theta(self, key, iter_idx):
        '''
        lr: learning rate
        '''
        data = self._load()
        lr = data['lr']
        if 'm' not in data:
            data['m'] = {}
        if key not in data['m']:
            data['m'][key] = {}
            data['m'][key][-1] = 0
        if 'v' not in data:
            data['v'] = {}
        if key not in data['v']:
            data['v'][key] = {}
            data['v'][key][-1] = 0
        if iter_idx is None:
            iter_idx = len(data['gradient'][key]) - 1
        m = self.beta_1 * data['m'][key][iter_idx - 1] + (
            1 - self.beta_1) * data['gradient'][key][iter_idx]
        v = self.beta_2 * data['v'][key][iter_idx - 1] + (1 - self.beta_2) * (
            data['gradient'][key][iter_idx]**2)
        data['m'][key][iter_idx] = m
        data['v'][key][iter_idx] = v
        self._save(data)
        m_cap = m / (1 - (self.beta_1**(iter_idx + 1)))
        v_cap = v / (1 - (self.beta_2**(iter_idx + 1)))
        return -(lr * m_cap) / (np.sqrt(v_cap) + 1e-8)

    def log(self, gs, key, iter_idx):
        '''
        record gradient and return delta_theta
        theta[iter_idx + 1] = theta[iter_idx] + delta_theta
        '''
        self.record_gradient(gs, key, iter_idx)
        return self.delta_theta(key, iter_idx)
