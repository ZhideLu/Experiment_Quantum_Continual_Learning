import numpy as np
from . import pickle_depository


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


class Nadam_log(pickle_depository):
    '''Nadam iter_idx must start from 1'''
    def __init__(self,
                 save_path,
                 lr=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 momentum_decay=0.004):
        super().__init__(save_path)
        data = self._load()
        data['lr'] = lr
        self._save(data)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.momentum_decay = momentum_decay

    def ut(self, iter_idx):
        return self.beta_1 * (1 - 0.5 * 0.96**(iter_idx * self.momentum_decay))

    def product_ut(self, iter_idx):
        x = np.arange(1, iter_idx + 1)
        return np.prod(self.ut(x))

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
            data['m'][key][0] = 0
        if 'v' not in data:
            data['v'] = {}
        if key not in data['v']:
            data['v'][key] = {}
            data['v'][key][0] = 0
        m = self.beta_1 * data['m'][key][iter_idx - 1] + (
            1 - self.beta_1) * data['gradient'][key][iter_idx]
        v = self.beta_2 * data['v'][key][iter_idx - 1] + (1 - self.beta_2) * (
            data['gradient'][key][iter_idx]**2)

        data['m'][key][iter_idx] = m
        data['v'][key][iter_idx] = v
        self._save(data)

        m_cap = self.ut(iter_idx + 1) * m / (
            1 - self.product_ut(iter_idx + 1)) + (
                1 - self.ut(iter_idx)) * data['gradient'][key][iter_idx] / (
                    1 - self.product_ut(iter_idx))

        v_cap = v / (1 - (self.beta_2**(iter_idx)))

        delta_theta = -(lr * m_cap) / (np.sqrt(v_cap) + 1e-8)

        return delta_theta

    def log(self, gs, key, iter_idx):
        '''
        record gradient and return delta_theta
        theta[iter_idx + 1] = theta[iter_idx] + delta_theta
        '''
        self.record_gradient(gs, key, iter_idx)
        return self.delta_theta(key, iter_idx)
