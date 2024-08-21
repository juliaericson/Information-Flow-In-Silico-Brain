from scipy.fft import fft, fftfreq
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import numpy as np


class KuramotoNetwork:
    def __init__(self, itt, network, distances, w_dist=0.5, omega=0.04*np.ones(214), k=0.22, sigma=0):  
        self.itt = itt
        self.network = network
        self.distances = (distances/w_dist).astype(int)
        self.size = network.shape[0]
        self.omega = omega*np.pi*2 
        self.k = k 
        self.theta = np.zeros((itt + 1, self.size))
        self.theta[:200] = np.sin(np.random.rand(200, network.shape[0])*2*np.pi)
        self.E_delay = np.zeros(self.size)
        self.sigma = sigma

    def update_current(self, t, dt=0.1):
        for i in range(self.size):
            delays = np.sin(self.theta[t - self.distances[i], np.arange(self.size)] - self.theta[t, i])
            self.E_delay[i] = np.dot(self.network[i], delays)
        self.theta[t+1] = self.theta[t] + dt * (self.k*self.E_delay + self.omega) \
                          + np.random.normal(0, self.sigma, self.size)

    def run_simulation(self):
        for t in range(self.itt-199):
            if t % 5000 == 0:
                print(t)
            self.update_current(t+199)

    @staticmethod
    def calculate_frequency(current, N, T, label=None, plot_amp=False, norm=True):
        freq = fft(current)
        xf = fftfreq(N, T)[:N // 2]
        amp = (2.0 / N) * np.abs(freq[0:N // 2])
        x = np.where(xf < 101)[0]
        frequencies = xf[1:x.shape[0]][::10]
        if norm:
            amplitude = (amp[1:x.shape[0]]/np.sum(amp))[::10]
        else:
            amplitude = (amp[1:x.shape[0]])[::10]    
        if plot_amp:
            plt.plot(frequencies, amplitude, label=label)
            plt.xlabel('frequency')
            plt.ylabel('amplitude')
        return frequencies, amplitude

    @staticmethod
    def plot_current(current, start, stop, labels):
        for n in range(current.shape[1]):
            plt.plot(np.arange(stop-start)*0.1, current[start: stop, n], label=labels[n])
        plt.xlabel('time [ms]')
        plt.ylabel('current')
        plt.legend()
        plt.show()

    @staticmethod
    def create_heatmap(heatmap_array, f, alpha_i):
        ax = sns.heatmap(heatmap_array)
        f_idx = np.round(np.linspace(1, len(f) - 1, num=20)).astype(int)
        ax.set_xticks(f_idx)
        ax.set_xticklabels(f[f_idx])
        ax.set_yticklabels(alpha_i)
        plt.show()

    @staticmethod
    def calculate_synchrony(phase, imag=True):
        phase = phase[1000:]
        N_timepoints = phase.shape[0]

        pl_part1 = np.exp(1j * phase).T
        pl_part2 = np.exp(-1j * phase)
        synch_matrix = (1 / N_timepoints) * np.dot(pl_part1, pl_part2)
        if imag == True: 
            synch_matrix = np.imag(synch_matrix)
        synch_matrix = np.abs(synch_matrix)
        np.fill_diagonal(synch_matrix, 0)
        return synch_matrix
