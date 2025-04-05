import copy
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pickle_depository import *
import jax
import copy
import scipy

pi = np.pi


class circuit:
    """
    circuit layout

    """
    def __init__(self,
                 qnum=18,
                 block_num=4,
                 measure_qnum=8,
                 sq_num_per_block=3,
                 state_block_num=5):
        self.dev = qml.device(name="default.qubit",
                              wires=range(qnum),
                              shots=None)

        self.qnum = qnum
        self.block_num = block_num
        self.sq_num_per_block = sq_num_per_block
        self.measure_qnum = measure_qnum
        self.state_block_num=state_block_num

    def circuit(self):
        @qml.qnode(self.dev)
        def _circuit(pic_params, params):
            counter = 0
            params = params + pic_params
            params = params.reshape(self.sq_num_per_block * self.block_num, self.qnum)
            for _b in range(self.block_num):
                for qidx in range(self.qnum):
                    qml.RX(params[counter, qidx], qidx)
                    qml.RZ(params[counter + 1, qidx], qidx)
                    qml.RX(params[counter + 2, qidx], qidx)
                # if (_b+1)%2:
                for qidx in range(int(self.qnum / 2)):
                    qml.CNOT(wires=[qidx * 2, qidx * 2+1])
                for qidx in range(int((self.qnum - 1) / 2)):
                    qml.CNOT(wires=[qidx * 2 + 1, qidx * 2 + 2])
                # else:
                #     for qidx in range(int(self.qnum / 2)):
                #         qml.CNOT(wires=[qidx * 2+1, qidx * 2])
                #     for qidx in range(int((self.qnum - 1) / 2)):
                #         qml.CNOT(wires=[qidx * 2 + 2, qidx * 2 + 1])
                counter = counter + 3
                qml.Barrier()
            return qml.expval(op=qml.Hermitian(np.array([[1, 0], [0, 0]]),
                                               wires=self.measure_qnum))

        return _circuit



    def circuit_phase(self):
        @qml.qnode(self.dev)
        def _circuit(
            state_params,
            params,
        ):
            counter = 0
            state_params = state_params.reshape(3 * self.state_block_num, self.qnum)
            params = params.reshape(self.sq_num_per_block * self.block_num, self.qnum)
            ##
            for _b in range(self.state_block_num):
                for qidx in range(self.qnum):
                    qml.RX(state_params[counter, qidx], qidx)
                    qml.RZ(state_params[counter + 1, qidx], qidx)
                    qml.RX(state_params[counter + 2, qidx], qidx)
                for qidx in range(int(self.qnum / 2)):
                    qml.CZ(wires=[qidx * 2, qidx * 2 + 1])
                for qidx in range(int((self.qnum - 1) / 2)):
                    qml.CZ(wires=[qidx * 2 + 1, qidx * 2 + 2])
                counter = counter + 3
                qml.Barrier()

            counter = 0
            for _b in range(self.block_num):
                for qidx in range(self.qnum):
                    qml.RX(params[counter, qidx], qidx)
                    qml.RZ(params[counter + 1, qidx], qidx)
                    qml.RX(params[counter + 2, qidx], qidx)
                for qidx in range(int(self.qnum / 2)):
                    qml.CNOT(wires=[qidx * 2, qidx * 2 + 1])
                for qidx in range(int((self.qnum - 1) / 2)):
                    qml.CNOT(wires=[qidx * 2 + 1, qidx * 2 + 2])
                counter = counter + 3
                qml.Barrier()
            return qml.expval(op=qml.Hermitian(np.array([[1, 0], [0, 0]]),
                                               wires=self.measure_qnum))

        return _circuit


    def circuit_amplitude(self):
        # @qml.qnode(self.dev, interface="autograd", diff_method="backprop")
        @qml.qnode(self.dev)
        def _circuit(pic_params, params):
            # breakpoint()
            qml.StatePrep(np.array(pic_params), wires=range(self.qnum))
            counter = 0
            params = params.reshape(3 * self.block_num, self.qnum)
            for _b in range(self.block_num):
                for qidx in range(self.qnum):
                    qml.RX(params[counter, qidx], qidx)
                    qml.RZ(params[counter + 1, qidx], qidx)
                    qml.RX(params[counter + 2, qidx], qidx)
                for qidx in range(int(self.qnum / 2)):
                    qml.CNOT(wires=[qidx * 2, qidx * 2 + 1])
                if self.qnum % 2:
                    for qidx in range(int(self.qnum / 2)):
                        qml.CNOT(wires=[qidx * 2 + 1, qidx * 2 + 2])
                else:
                    for qidx in range(int(self.qnum / 2) - 1):
                        qml.CNOT(wires=[qidx * 2 + 1, qidx * 2 + 2])
                counter = counter + 3
                qml.Barrier()
            return qml.expval(op=qml.Hermitian(np.array([[1, 0], [0, 0]]),
                                               wires=self.measure_qnum))

        return _circuit


class cml(pickle_depository):
    """ """
    def __init__(
        self,
        measure_qnum=8,
        qnum=18,
        block_num=4,
        sq_num_per_block=3,
        param_num=None,
        state_block_num=5,
        seed_num=0,
        save_paht=''
    ):
       
        self.measure_qnum = measure_qnum
        self.qnum = qnum
        self.block_num = block_num
        self.sq_num_per_block = sq_num_per_block
        self.sq_block_num = self.block_num * sq_num_per_block
        self.state_block_num=state_block_num
        self.jit_circuit = jax.jit(
            circuit(qnum=qnum, block_num=block_num,
                    measure_qnum=measure_qnum,sq_num_per_block=sq_num_per_block).circuit())
        self.jit_circuit_amp = jax.jit(circuit(
            qnum=qnum, block_num=block_num,
            measure_qnum=measure_qnum,sq_num_per_block=sq_num_per_block,state_block_num=state_block_num).circuit_phase())
        
        self.load_train_datasets()
        param_num=self.qnum*self.sq_num_per_block*self.block_num if param_num is None else param_num
        self.param_num=param_num
        np.random.seed(1)
        x = 2 * np.pi * (np.random.rand(param_num)-0.5)
        self.x = x
        self.params = jax.numpy.array(x)
        print("use random params!!!")
        self.SAVE_PATH = save_paht 
        self.seed_num=seed_num

    def save_params(self, task, train_num, measure_qnum):
        save_path = "/training_params/train_" + task + '_' + "params_num=" + str(
            train_num) + "_readq" + str(measure_qnum)
        with open(self.SAVE_PATH + save_path, "wb") as f:
            pickle.dump(self.params, f)

    def load_params(self, task, train_num, measure_qnum):
        save_path = "/training_params/train_" + task + "_params_num=" + str(
            train_num) + "_readq" + str(measure_qnum)
        with open(self.SAVE_PATH + save_path, "rb") as f:
            data = pickle.load(f)
        return data

    def update_params(self, delta_theta):
        self.params += delta_theta

    def load_train_datasets(self,tasks = ['FashionMNIST_09', 'medical', 'spt']):
        self.x_train = {}
        self.y_train = {}
        self.x_test = {}
        self.y_test = {}
        with open('./dataset/Data.pkl', 'rb') as f:
            data=pickle.load(f)
        for task in tasks:
            self.x_train[task] = data[task]['x_train']
            self.y_train[task] = data[task]['y_train']
            self.x_test[task] = data[task]['x_test']
            self.y_test[task] = data[task]['y_test']
    def encode_pic_data(self, x_train, theta2data):
        x_pad_zeros = np.zeros([x_train.shape[0], self.param_num])
        for idx in range(x_train.shape[0]):
            for index,jdx in enumerate(np.arange(0,x_train[:,:256].shape[1],2)):
                x_pad_zeros[idx, index] = (x_train[idx,jdx]+x_train[idx,jdx+1]) * theta2data
        self.pic_params = x_pad_zeros

    def encode_pic_data_random_perm(self, x_train, theta2data):
        x_pad_zeros = np.zeros([x_train.shape[0], self.param_num])
        for idx in range(x_train.shape[0]):
            for index,jdx in enumerate(np.arange(0,x_train[:,:256].shape[1],2)):
                x_pad_zeros[idx, index] = (x_train[idx,jdx]+x_train[idx,jdx+1]) * theta2data

        np.random.seed(self.seed_num)
        per_index=np.random.permutation(np.arange(0,int(x_pad_zeros.shape[1])))
        
        x_pad_zeros[:,:]=x_pad_zeros[:,per_index]
        self.pic_params = x_pad_zeros
        return per_index

    def encode_state_data(self, x_train):
        x_pad_zeros = np.zeros(
        [x_train.shape[0], self.qnum * self.state_block_num * self.sq_num_per_block])
        for idx in range(x_train.shape[0]):
            x_pad_zeros[idx, :x_train.shape[1]] = x_train[idx]
        
        self.pic_params = x_pad_zeros

    def loss_and_accuracy(
            self,
            params,
            theta2data=2,
            amp=False,
            test=False,
            task='FashionMNIST_17',
            random_perm=False,
            batch_index=None):
        x_train=self.x_test[task] if test else self.x_train[task]
        y_train=self.y_test[task] if test else self.y_train[task]
        if batch_index is None:
            batch_index = np.arange(x_train.shape[0]) if test else self.train_idx
        if amp:
            jit_circuit=self.jit_circuit_amp
        else:
            jit_circuit=self.jit_circuit
        losses = 0
        li = 0
        if amp:
            self.encode_state_data(x_train=x_train)
        elif random_perm:
            self.encode_pic_data_random_perm(x_train, theta2data=theta2data)
        else:
            self.encode_pic_data(x_train, theta2data=theta2data)
        line=0.5
        for index, idx in enumerate(batch_index):
            p0 = jit_circuit(self.pic_params[idx], params)
            p1 = 1 - p0
            label = y_train[:, idx]
            p = label[0] * p0 + label[1] * p1
            if p >line:
                li += 1
            sigle_loss = -label[0] * np.log(p0) - label[1] * np.log(p1)
            losses += sigle_loss
        accuracy_rate = li / len(batch_index)
        return losses / len(batch_index), accuracy_rate



    def gradient(self, params, task=None,theta2data=2, amp=False,noise_level=0,random_perm=False,classical=False):
        gradients = 0
        x_train=self.x_train[task]
        y_train=self.y_train[task]
        if amp :
            self.encode_state_data(x_train=x_train)
        elif random_perm:
            self.encode_pic_data_random_perm(x_train, theta2data=theta2data)
        else:
            self.encode_pic_data(x_train, theta2data=theta2data)
        if amp:
            jit_circuit=self.jit_circuit_amp
        else:
            jit_circuit=self.jit_circuit
        for idx in self.train_idx:
            def loss(para):
                p0 = jit_circuit(self.pic_params[idx], para)
                p1 = 1 - p0

                sigle_loss = -y_train[0, idx] * jax.numpy.log(p0) - y_train[
                    1, idx] * jax.numpy.log(p1)
                return sigle_loss
            gradients += jax.grad(loss)(params)
            gradients += np.random.normal(loc=0,scale=noise_level,
                                           size=len(gradients))
        return gradients / len(self.train_idx)
