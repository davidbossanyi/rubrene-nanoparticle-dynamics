from model import trRubreneModel
import matplotlib.pyplot as plt


# set model parameters
m = trRubreneModel()
m.kSF = 104
m.kTF = 118
m.kHOP = 7
m.k_HOP = 1.2
m.kSPIN = 0.25
m.kFRET = 20
m.kSSA = 3e-17
m.kR = 0.0625
m.kDBP = 0.25
m.kT = 1e-5
m.initial_weighting = {'S': 1, 'DBP': 0.1}

tscale = 0.36  # scale the triplet population
dbpscale = 0.7  # scale the DBP singlet population

powers = [0.3, 0.7, 1.6]
alphas = [0.2, 0.4, 0.8]
fig, ax = plt.subplots()

for i, power in enumerate(powers):
    
    N0 = 6e17*power
    m.G = N0
    m.simulate()
        
    ax.plot(m.t*1000, m.S/max(m.S), color='darkorange', alpha=alphas[i])
    ax.plot(m.t*1000, tscale*(m.TT+m.T_T+m.T_Tm)/max(m.S), color='darkblue', alpha=alphas[i])
    ax.plot(m.t*1000, dbpscale*m.DBP/max(m.S), color='darkred', alpha=alphas[i])

ax.set_xlim([0.1, 7000])

ax.set_xscale('log')

ax.axhline(0, linewidth=1, color='0.5')   

ax.set_ylabel(r'$\Delta$A (normalised)')
ax.set_xlabel('Time (ps)')
