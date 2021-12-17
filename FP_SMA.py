from SMA import BaseSMA
from numpy.random import uniform, choice
from numpy import abs, zeros, log10, where, arctanh, tanh, round, cos
import numpy as np
from sklearn.cluster import KMeans
from Utilities import *
import warnings
from sklearn.exceptions import ConvergenceWarning
import time

class FP_SMA(BaseSMA):

    def __init__(self, obj_func=None, lb=None, ub=None, problem_size=50, verbose=True, epoch=750, pop_size=100, z=0.03, min_pop_siz=3):
        BaseSMA.__init__(self, obj_func, lb, ub, problem_size, verbose, epoch, pop_size, z)
        self.init_pop_size = pop_size # \bar{NP}
        self.D = problem_size
        self.faction = 0.01
        self.M = self.init_pop_size / self.D * self.faction
        self.min_pop_size = min_pop_siz
        self.max_pop_size = 10 * self.init_pop_size
        self.T = epoch
        self.groupB = []

        self.Num_Fluctuation_Period = 10
        self.Lambda = self.T / self.Num_Fluctuation_Period
        self.min_pop_size_exploration = self.init_pop_size / 2
        self.min_RD = 0.1

    def population_diversity2(self, pop):

        pop_pos = [pop[i][self.ID_POS] for i in range(len(pop))]
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=ConvergenceWarning)
            kmean = KMeans(algorithm="elkan")
            X_new = kmean.fit_transform(pop_pos)
            distance = 0
            for i in range(len(pop)):
                distance += min(X_new[i])
            return distance / len(pop)

    def reference_diversity(self, x):
        beta = 10
        offset = 0.5
        return 0.5 * (1 - arctanh_scaled_offset(x, beta, offset))

    def plot_clustering(self, pop):
        pop_pos = [pop[i][self.ID_POS] for i in range(len(pop))]
        #plot_clustering(pop_pos)

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)      # Eq.(2.6)
        DI_init = self.population_diversity2(pop)
        for epoch in range(self.epoch):

            s = pop[0][self.ID_FIT] - pop[-1][self.ID_FIT] + self.EPSILON  # plus eps to avoid denominator zero

            # calculate the fitness weight of each slime mold
            for i in range(0, self.pop_size):
                # Eq.(2.5)
                if i <= int(self.pop_size / 2):
                    pop[i][self.ID_WEI] = 1 + uniform(0, 1, self.problem_size) * log10((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)
                else:
                    pop[i][self.ID_WEI] = 1 - uniform(0, 1, self.problem_size) * log10((pop[0][self.ID_FIT] - pop[i][self.ID_FIT]) / s + 1)

            a = arctanh(-((epoch + 1) / self.epoch) + 1)                        # Eq.(2.4)
            b = 1 - (epoch + 1) / self.epoch

            # Update the Position of search agents
            for i in range(0, self.pop_size):
                if uniform() < self.z:  # Eq.(2.7)
                    pos_new = uniform(self.lb, self.ub)
                else:
                    p = tanh(abs(pop[i][self.ID_FIT] - g_best[self.ID_FIT]))    # Eq.(2.2)
                    vb = uniform(-a, a, self.problem_size)                      # Eq.(2.3)
                    vc = uniform(-b, b, self.problem_size)

                    # two positions randomly selected from population, apply for the whole problem size instead of 1 variable
                    id_a, id_b = choice(list(set(range(0, self.pop_size)) - {i}), 2, replace=False)

                    pos_1 = g_best[self.ID_POS] + vb * (pop[i][self.ID_WEI] * pop[id_a][self.ID_POS] - pop[id_b][self.ID_POS])
                    pos_2 = vc * pop[i][self.ID_POS]
                    pos_new = where(uniform(0, 1, self.problem_size) < p, pos_1, pos_2)

                # Check bound and re-calculate fitness after each individual move
                pos_new = self.amend_position(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                pop[i][self.ID_POS] = pos_new
                pop[i][self.ID_FIT] = fit_new

            # # Sorted population and update the global best
            # pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            # self.loss_train.append(g_best[self.ID_FIT])
            # if self.verbose:
            #     print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

            # Fluctuate the population size
            t = epoch
            D = self.D
            T = self.T
            try:
                DI_t = self.population_diversity2(pop)
            except ValueError:
                print("> Epoch {}: population is converged, break".format(epoch))
                break

            RD_t = DI_t / DI_init
            REP_t = (t / T)
            Ref_RD_t = self.reference_diversity(REP_t)
            M = self.M
            M_t = 1.0 * (M - 1) * (T - t) / T + 1
            Delta_t = int(max(round(M_t * D / 2 * np.sin(t * np.pi / self.Lambda)**2 + 1), 1))
            Threshold_low = 0.9 * Ref_RD_t
            Threshold_high = 1.1 * Ref_RD_t

            old_pop_size = self.pop_size
            if RD_t < self.min_RD and len(pop) > self.min_pop_size_exploration:
                self.plot_clustering(pop)
                pop = pop[0: len(pop) - Delta_t]
                self.pop_size -= Delta_t
            elif RD_t < self.min_RD and len(pop) < self.min_pop_size_exploration:
                self.pop_size = self.init_pop_size
                pop = [self.create_solution() for _ in range(self.pop_size)]
                DI_init = self.population_diversity2(pop)
            elif self.min_RD < RD_t < Threshold_low and len(pop) < self.init_pop_size:
                pop_delta = [self.create_solution() for _ in range(Delta_t)]
                pop = pop + pop_delta
                self.pop_size += Delta_t
            elif RD_t > Threshold_high and len(pop) > self.min_pop_size_exploration:
                pop = pop[0: len(pop)-Delta_t]
                self.pop_size -= Delta_t

            #if self.verbose:
                #print("FP-SMA: Epoch {}, RD_n = {}, threshold_low={}, threshold_high={}, pop size = {} -> {}".format(
                #    t, RD_t, Threshold_low, Threshold_high, old_pop_size, self.pop_size))
            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train



