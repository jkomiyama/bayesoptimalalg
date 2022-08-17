# -*- coding: utf-8 -*-

import sys, math, copy
import numpy as np
from scipy import stats
from scipy.stats import truncnorm
from scipy.stats import norm

def approximateSimpleRegret(hatmus, Ns):

    K = len(hatmus)
    sorted_hatmus = sorted(hatmus)[::-1]
    j = np.argmax(hatmus) # recommendation arm
    N = Ns[j]
    # approximated simple regret
    asr = 0
    asr_part = []
    for i in range(K):
        if i == j: #max
            continue
            #gap = sorted_hatmus[0] - sorted_hatmus[1]
        else:
            gap = sorted_hatmus[0] - hatmus[i]
        val = truncnorm.stats(gap, gap+10000, moments='mvsk')[0] #mvsk = mean,variance,stddev,kurtosis
        stddev = math.sqrt( 1./Ns[i] + 1./Ns[j] )
        v = stats.norm.cdf(-gap/stddev) * (val - gap)
        asr += v 
        asr_part.append( v )
    # print(f"asr_part = {asr_part}")
    return asr

# single run, Ninit*K is initial exploration
def run(initial_minus, Ninit, mus, T = 100, alg = "SR", rs = None):
    # print(f"initial_minus = {initial_minus}")
    # generate rewards
    def draw(i, mus, rng, initial_minus, T):
        if initial_minus > 0:
            #b = norm.ppf(1.0/T)
            #r = truncnorm.rvs(-100, b) #lower 1/T quantile
            val = rng.normal(0, 1) + mus[i] -  np.sqrt(initial_minus * np.log(T))
            #print(f"val = {val}")
            #sys.stdout.flush()
            return val
        else:
            return rng.normal(0, 1) + mus[i]

    if rs != None:
        rng = np.random.RandomState(rs)
    else:
        rng = np.random
    K = len(mus)
    Xs = np.zeros(K) # rewards
    ist = np.argmax(mus)
    #Xs[ist] -= initial_minus 
    Ns = np.zeros(K) # 
    #regret_run = np.zeros(T)
    #def get_regret(t, Ns):
    #    return max(mus)*(t+1) - Ns[0] * mus[0] - Ns[1] * mus[1]
    if alg == "ABO": # Bayes Optimal, Myopic
        # initial sample 
        for t in range(K*Ninit):
            i = t % K #for each arm
            Ns[i] += 1
            Xs[i] += draw(i, mus, rng, initial_minus, T)
            initial_minus = 0
            #regret_run[t] = get_regret(t, Ns)        
        # after initial samples: 
        for t in range(K*Ninit, T):
            hatmus = Xs / Ns
            # posterior N(mu, 1/N)
            asrs = []
            for i in range(K):
                Nbak = Ns[i]
                Ns[i] += 1
                vd = approximateSimpleRegret(hatmus, Ns)
                Ns[i] = Nbak
                v = approximateSimpleRegret(hatmus, Ns)
                asrs.append(v - vd)
            #print(f"hatmus = {hatmus}, Ns = {Ns}, asrs = {asrs}")
            I = np.argmax(asrs)
            Ns[I] += 1
            Xs[I] += draw(I, mus, rng, initial_minus, T)
            initial_minus = 0
        #print(f"ABO Ns = {Ns}")
        JT = np.argmax(hatmus)
    elif alg == "SR":
        def ulogk(k):
            v = 0.5
            for i in range(2,k+1):
                v += 1./i
            return v
        def SR_nk(k): #segment size
            if k==0: return 0
            return math.ceil(   (1/ulogk(K)) * (T-K)/(K+1-k)   )
        #print(f"n0 = {SR_nk(0)}, n1 = {SR_nk(1)}, n2 = {SR_nk(2)}, n3 = {SR_nk(3)} ulogk = {ulogk(K)} test = {(T-K)/(K+1-2)}")
        A = [i for i in range(K)]
        t = 0
        for k_p in range(1, K): # k=1,2,...,K-1
            hatmus = []
            size = SR_nk(k_p) - SR_nk(k_p-1)
            #print(f"k_p = {k_p}, size = {size}")
            if k_p+1 < K:
                for i in A:
                    for j in range(size):
                        Ns[i] += 1
                        Xs[i] += draw(i, mus, rng, initial_minus, T)
                        initial_minus = 0
                        t += 1
                    hatmus.append( Xs[i]/Ns[i] )
            else: #k_p = K-1
                while t < T:
                    I = A[t%2]
                    Ns[I] += 1
                    Xs[I] += draw(I, mus, rng, initial_minus, T)
                    initial_minus = 0
                    t += 1
                hatmus = [Xs[i]/Ns[i] for i in A]
            imin = np.argmin(hatmus)
            A = A[:imin]+A[imin+1:]
            #print(f"A = {A} T={t}")
        JT = A[0] #unique left arm
    else:
        print("Unknown algorithm")
        assert(False)
    return max(mus) - mus[JT], Ns # simple regret, Ns

from joblib import Parallel, delayed

import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
NUM_PARALLEL = int(mp.cpu_count() * 0.8)

#Runnum = 100
Runnum = 1000
#Ts = [100, 300, 1000]
Ts = [30, 100, 300, 1000, 3000, 10000] 

def runbatch(mode = 1, initial_minus = 0):
    SplReg_SR = []
    SplReg_ABO = []
    largeMisestABO = [] # prob that only <5% of samples are allocated to arm 3
    if mode == 1:
        mus = np.array([0.5, 0, 0])
    if mode == 2:
        mus = np.array([1.0, 0.8, 0.0])
    for T in Ts:
        sr_list = []
#       # single_thread
#        for r in range(Runnum): # run for Runnum times
#            sr, Ns = run(initial_minus = initial_minus, Ninit = 1, mus = mus, T = T, alg = "SR")
#            sr_list.append(sr)
        rss = np.random.randint(np.iinfo(np.int32).max, size=Runnum)
        print(f"rss = {rss}")
        results = Parallel(n_jobs=NUM_PARALLEL)( [delayed(run)(initial_minus = initial_minus, Ninit = 1, mus = mus, T = T, alg = "SR", rs=rss[r]) for r in range(Runnum)] ) 
        sr_list = [r[0] for r in results]
        print(f"results = {results}")
        SplReg_SR.append(copy.deepcopy(sr_list))
        print(f"T = {T} simple regret of SR = {np.mean(sr_list)}")

        sr_list = []
#        for r in range(Runnum): # run for Runnum times
#            sr, Ns = run( initial_minus = initial_minus, Ninit = 1, mus = mus, T = T, alg = "ABO")
#            sr_list.append(sr)
        rss = np.random.randint(np.iinfo(np.int32).max, size=Runnum)
        results = Parallel(n_jobs=NUM_PARALLEL)( [delayed(run)(initial_minus = initial_minus, Ninit = 1, mus = mus, T = T, alg = "ABO", rs=rss[r]) for r in range(Runnum)] ) 
        sr_list = [r[0] for r in results]
        Ns_list = [r[1] for r in results]
        lm_count = 0
        for Ns in Ns_list:
            if Ns[np.argmax(mus)] * 20 <= T:
                lm_count += 1
        largeMisestABO.append( lm_count / Runnum )
        SplReg_ABO.append(sr_list)
        print(f"results = {results}")
        print(f"T = {T} simple regret of ABO = {np.mean(sr_list)}")

        sys.stdout.flush()
    return (SplReg_SR, SplReg_ABO, largeMisestABO)

import pickle

def colab_save(filename):
    try:
        from google.colab import files
        files.download(filename)  
    except:
        pass


# plotting results 
import matplotlib
try:
    from google.colab import files
except: # https://stackoverflow.com/questions/35737116/runtimeerror-invalid-display-variable
    matplotlib.use('agg')
import matplotlib.pyplot as plt
def my_show():
    try:
        from google.colab import files
        plt.show()
    except:
        pass
def colab_save(filename):
    try:
        from google.colab import files
        files.download(filename)  
    except:
        pass

Figsize = (6,4)
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.subplot.bottom"] = 0.14
confidence_bound = True

COLOR_SR = "tab:blue"
COLOR_ABO = "tab:red"
COLOR_ABOE = "black"

def my_plot(results, figname):
    SplReg_SR, SplReg_ABO, ratioLargeMiss = results
    SplReg_SR_mean   = np.array([np.mean(d) for d in SplReg_SR])
    SplReg_ABO_mean = np.array([np.mean(d) for d in SplReg_ABO])
    SplReg_SR_std    = np.array([np.std(d) for d in SplReg_SR])
    SplReg_ABO_std  = np.array([np.std(d) for d in SplReg_ABO])
    fig = plt.figure(figsize=Figsize)
    if confidence_bound:
        #SplReg_ABO_lower = SplReg_ABO_mean - SplReg_ABO_std*2
        #SplReg_ABO_upper = SplReg_ABO_mean + SplReg_ABO_std*2
        plt.errorbar(Ts, SplReg_SR_mean, yerr=2*SplReg_SR_std/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_SR) #2 sigma
        plt.errorbar(Ts, SplReg_ABO_mean, yerr=2*SplReg_ABO_std/np.sqrt(Runnum), fmt='o', capsize = 5, color = COLOR_ABO) #2 sigma
    plt.plot(Ts, SplReg_SR_mean, label = "SR", marker = "o", markersize = 10, color = COLOR_SR) #, linestyle = LINESTYLE_MODE2)
    plt.plot(Ts, SplReg_ABO_mean, label = "ABO", marker = "s", markersize = 10, color = COLOR_ABO) #, color = COLOR_MODE2, linestyle = LINESTYLE_MODE2)
    plt.legend()
    plt.ylabel("Simple Regret")
    plt.xlabel("Round (T)")
    plt.xscale("log")
    print("Saving fig")
    my_show()
    fig.savefig(figname+".pdf", dpi=fig.dpi, bbox_inches='tight')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    if confidence_bound:
        # binomial confidence region p  +/-  z*(√p(1-p) / n) z=2
        confregion = np.array([np.sqrt(ratioLargeMiss[i]*(1-ratioLargeMiss[i])/Runnum) for i,T in enumerate(Ts)])
        print(f"confregion = {confregion}")
        plt.errorbar(Ts, ratioLargeMiss, yerr=2*confregion, fmt='o', capsize = 5, color = COLOR_ABOE) #2 sigma
    plt.plot(Ts, ratioLargeMiss, marker = "s", markersize = 10, color = COLOR_ABOE)
    #plt.legend()
    plt.ylabel("Ratio of N3(T)<0.05T")
    plt.xlabel("Round (T)")
    plt.xscale("log")
    print("Saving fig")
    my_show()
    fig.savefig(figname+"_largemiss.pdf", dpi=fig.dpi, bbox_inches='tight')
    plt.clf()

np.random.seed(1)
results_m1_initminus = runbatch(initial_minus = 4, mode = 1)
pickle.dump( results_m1_initminus, open( "results_m1_initminus.pickle", "wb" ) )
colab_save("results_m1_initminus.pickle")
my_plot(results_m1_initminus, "M1_init")
