import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import numpy.linalg as la
#from scipy.linalg import eigh
import matplotlib.pyplot as plt
import math as math

"#Set_x_axis_limit_and_steps"

XLENG = 30.0
XMIN = 0.0

xleng = int(XLENG)
N = 300.0
n = int(N)
d = XLENG/N

"time_evolution"
TMAX = 25.0
tmax = int(TMAX)
TN = 30.0
tn = int(TN)
tanal = TMAX/TN

def potential2(k, step_d):
	n = int(N)
	V = np.zeros((k, k), dtype=float)
	for i in range(n):
		V[i, i] = -0.08*i + 0.5*np.sin(0.2*math.pi*i)
	return V 


def potential(k, step_d):
	n = int(N)
	V = np.zeros((k, k), dtype=float)
	for i in range(n):
		V[i, i] = 0
	return V 


def binding(n, g):
    t = np.zeros(N, dtype=float)
    for i in range(0, n):
        t[i] = - 1
    return t


def hamiltonian(K, V, t):
    k = int(K)
    HO = np.zeros((k, k), dtype=float)
    ham = np.zeros((k, k), dtype=float)
    for i in range(0, k):
        for j in range(0, k):
            if abs(i-j) == 1 or (i == (n-1) and j == 0) or (j == (n-1) and i == 0):
                HO[i, j] = -1
    ham = HO + V
    return ham, HO


def wavepacket(k, step_d):
    CR = np.zeros(n, dtype=float)
    CI = np.zeros(n, dtype=float)
    for i in range(0, n):
        CR[i] = np.exp((i*step_d-15)**2)
        CI[i] = 0
    return CR, CI


def iterations(v, w, CR, CI, n, tn, tanal):
    """real part of wavefunction"""
    YR = np.empty((tn, n), dtype=float)
    """imagin part of wavefunction"""
    YI = np.empty((tn, n), dtype=float)
    for q in range(0, tn):
        print(q)
        for k in range(0, n):
            sum1 = 0
            sum2 = 0
            for i in range(0, n):
                for m in range(0, n):
                    sum1 = sum1 + w[i, m] * w[k, m] * (CR[i]*np.cos(v[m] * (q*tanal)) + CI[i]*np.sin(v[m] * (q*tanal)))
                    sum2 = sum2 + w[i, m] * w[k, m] * (-CR[i]*np.sin(v[m] * (q*tanal)) + CI[i]*np.cos(v[m] * (q*tanal)))
            YR[q, k] = sum1
            YI[q, k] = sum2
    return YR, YI


def normalization(n, tn, YR, YI):
    """real part of probability"""
    PIT = np.zeros((tn, n), dtype=float)
    for q in range(0, tn):
        test = 0
        for i in range(0, n):
            PIT[q, i] = (YR[q, i]**2 + YI[q, i]**2)
        for i in range(0, n):
            test = test + PIT[q, i] 
        PIT[q] = PIT[q]/test
    return PIT


def average_pos(tn, n, step_d, PIT):
    xav = np.empty(tn, dtype=float)
    xsq = np.empty(tn, dtype=float)
    ds = np.empty(tn, dtype=float)
    for q in range(0, tn):
        temp = 0
        for i in range(0, n):
            temp = temp + PIT[q, i] * (i*d)
        xav[q] = temp
        print(xav[q])
    for q in range(0, tn):
        tempo = 0
        for i in range(0, n):
            tempo = tempo + PIT[q, i] * ((i*d)**2)
        xsq[q] = tempo
    for q in range(0, tn):
        ds[q] = np.sqrt(xsq[q]-(xav[q])**2)
    return xav, xsq, ds


def main():
    t = -1
    CR, CI = wavepacket(n, d)
    V = potential(N, d)
    H, HO = hamiltonian(N, V, t)
    print('V=', V)
    #print('HO=', HO)

    v, w = la.eig(H)
    YR, YI = iterations(v, w, CR, CI, n, tn, tanal)
    PIT = normalization(n, tn, YR, YI)

    plt.imshow(np.flipud(PIT), extent=[int(XMIN), int(XMIN+XLENG), 0, tmax], aspect='auto')
    plt.colorbar()
    plt.set_cmap('Blues')
    plt.title('Probability density ')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.savefig('thesed1.png')
    plt.clf()

    #T = np.linspace(0, TMAX, tn)

    #plt.plot(T, xav, color="blue")
    #plt.title("Mean position of the wavepacket")
    #plt.xlabel("t")
    #plt.ylabel("$\overline{x}$")
    #plt.savefig('mocked1.png')
    #plt.clf()

    #X = np.linspace(0, XMAX, n)
    #plt.title("Time evolution of the probability distribution")
    #plt.xlabel("x")
    #plt.ylabel("P")
    #plt.plot(X, PIT[0], color="black", label="t=0")
    #plt.plot(X, PIT[int(tn/6)], color="blue", label="t=10")
    #plt.plot(X, PIT[int(2*tn/6)], color="red", label="t=20")
    #plt.plot(X, PIT[int(4*tn/6)], color="yellow", label="t=40")
    #plt.plot(X, PIT[int(4*tn/5)], color="magenta", label="t=40")
    #plt.plot(X, PIT[tn-1], color="green", label="t=60")
    #plt.legend()
    #plt.savefig('mocked4.png')
    #plt.clf()
main()

