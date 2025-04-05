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
