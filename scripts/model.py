import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class RateModel:

    def __init__(self):
        self._number_of_states = 2
        self.states = ['S', 'T']
        self.rates = []
        self.model_name = 'base'
        self._time_resolved = True
        self.G = 1e17
        self._allowed_initial_states = {'S', 'T'}
        self._initial_state_mapping = {'S': 0, 'T': -1}
        self.initial_weighting = {'S': 1}
        
    def _check_initial_weighting(self):
        for starting_state in self.initial_weighting.keys():
            if starting_state not in self._allowed_initial_states:
                raise ValueError('invalid state {0} in initial_weighting'.format(starting_state))
            if self.initial_weighting[starting_state] < 0:
                raise ValueError('weightings must be positive')
        return
            
    def _set_initial_condition(self):
        self._y0 = np.zeros(self._number_of_states)
        total_weights = np.sum(np.array(list(self.initial_weighting.values())))
        for key in self.initial_weighting.keys():
            idx = self._initial_state_mapping[key]
            weight = self.initial_weighting[key]/total_weights
            self._y0[idx] = weight*self.G
        return


class TimeResolvedModel(RateModel):

    def __init__(self):
        super().__init__()
        self.t_step = 0.0052391092278624
        self.t_end = 1e6
        self.num_points = 10000
        return
    
    def _calculate_time_axis(self):
        self.t = np.geomspace(self.t_step, self.t_end+self.t_step, self.num_points)-self.t_step
        self.t[0] = 0
        return
    
    def view_timepoints(self):
        self._calculate_time_axis()
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.semilogx(self.t, np.ones_like(self.t), 'bx')
        plt.show()
        print('\n')
        for t in self.t[0:5]:
            print(t)
        print('\n')
        for t in self.t[-5:]:
            print(t)
        return
    
    def _rate_equations(self, y, t):
        return np.ones(self._number_of_states+1)
        
    def _initialise_simulation(self):
        self._calculate_time_axis()
        self._check_initial_weighting()
        self._set_initial_condition()
        return
    
    def simulate(self):
        self._initialise_simulation()
        y = odeint(lambda y, t: self._rate_equations(y, t), self._y0, self.t)
        self._unpack_simulation(y)
        return
    

class trRubreneModel(TimeResolvedModel):

    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'Rubrene Model'
        self._number_of_states = 5
        self.states = ['S', 'TT1', 'T_T1', 'T_Tm' 'DBP']
        self.rates = ['kSF', 'kTF', 'kHOP', 'k_HOP', 'kSPIN', 'kFRET', 'kR', 'kT', 'kDBP']
        # rates between excited states
        self.kSF = 100
        self.kTF = 100
        self.kHOP = 10
        self.k_HOP = 10
        self.kSPIN = 0.25
        self.kFRET = 10
        # annihilation rate
        self.kSSA = 3e-17
        # rates of decay
        self.kR = 0.0625
        self.kDBP = 0.25
        self.kT = 1e-5
        # initial stuff
        self._allowed_initial_states = {'S', 'DBP'}
        self._initial_state_mapping = {'S': 0, 'DBP': -1}
        self.initial_weighting = {'S': 1}
        self.cslsq = np.ones(9)/9
        self.ctlsq = np.zeros(9)

    def _rate_equations(self, y, t):
        S, TT1, T_T1, T_Tm, DBP = y
        dydt = np.zeros(self._number_of_states)
        # S1
        dydt[0] = -(self.kR+self.kSF+self.kFRET)*S - self.kSSA*S*S + self.kTF*TT1
        # TT1
        dydt[1] = self.kSF*S - (self.kHOP+self.kTF)*TT1 + self.k_HOP*T_T1# + 0.3*self.k_HOP*T_Tm
        # T_T1
        dydt[2] = self.kHOP*TT1 - (self.k_HOP+self.kSPIN+self.kT)*T_T1
        # T_Tm
        dydt[3] = self.kSPIN*T_T1 - self.kT*T_Tm# - self.k_HOP*T_Tm
        # DBP
        dydt[4] = self.kFRET*S - self.kDBP*DBP
        #
        return dydt

    def _unpack_simulation(self, y):
        self.S = y[:, 0]
        self.TT = y[:, 1]
        self.T_T = y[:, 2]
        self.T_Tm = y[:, 3]
        self.DBP = y[:, 4]
        self._wrap_simulation_results()
        return
    
    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S, self.TT, self.T_T, self.T_Tm, self.DBP]))
        return
