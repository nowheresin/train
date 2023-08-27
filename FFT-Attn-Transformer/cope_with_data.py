import struct
import numpy as np
import math
from scipy.signal import hilbert

# import matplotlib.pyplot as plt
# filepath='0007_bad_L8LB_S009_20220414_050_2.bin'

evelope_rete = 1.0

def get_envelope(x, n=None):
    """use the Hilbert transform to determine the amplitude envelope.
    Parameters:
    x : ndarray
        Real sequence to compute  amplitude envelope.
    N : {None, int}, optional, Number of Fourier components. Default: x.shape[axis]
        Length of the hilbert.

    Returns:
    amplitude_envelope: ndarray
        The amplitude envelope.

    """

    analytic_signal = hilbert(x, N=n)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def xshow(filename, nx, nz):
    '''
    filename:文件路径
    nx:行
    nz：列
    '''
    f = open(filename, "rb")
    figure = []
    for i in range(nx):
        for j in range(nz):
            for k in range(10):   # 6
                data = f.read(4)
            elem = struct.unpack("f", data)[0]
            k = math.floor(10*elem)
            figure = figure + [k]
    f.close()

    amplitude_envelope = get_envelope(figure)
    figure = [math.floor(evelope_rete * i + (1.0 - evelope_rete) * j) for i, j in zip(amplitude_envelope, figure)]

    min_figure = min(figure)

    for x in range(len(figure)):
        figure[x] = figure[x] - min_figure
    return [figure]

# data_arry = xshow(filepath,1,12000)
# plt.show()
# np.savetxt("001.csv", data_arry, delimiter=",")

'''
1 is Start Str
2 is Stop Str
0 is Occupy Str
3 is good signal
4 is abnormal/bad signal
5 is disease signal
'''
def show_good_signal(filename, nx, nz):
    enc_input = xshow(filename, nx, nz)
    dec_input = [[1, 3, 3, 3, 2]]
    dec_output = [[3, 3, 3, 0, 2]]
    return enc_input, dec_input, dec_output

def show_disease_signal(filename, nx, nz):
    enc_input = xshow(filename, nx, nz)
    dec_input = [[1, 5, 5, 5, 2]]
    dec_output = [[5, 5, 5, 0, 2]]
    return enc_input, dec_input, dec_output

def show_bad_signal(filename, nx, nz):
    enc_input = xshow(filename, nx, nz)
    dec_input = [[1, 4, 4, 4, 2]]
    dec_output = [[4, 4, 4, 0, 2]]
    return enc_input, dec_input, dec_output