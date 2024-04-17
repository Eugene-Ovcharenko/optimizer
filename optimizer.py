import os
import numpy as np
import matplotlib.pyplot as plt

from pymoo.problems import get_problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.visualization.scatter import Scatter
from pymoo.decomposition.asf import ASF
from test_problem import optimization_problem_test

# General settings
folder_path = 'results'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Problem implementation
class Problem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=4,  # number of parameters
                         n_obj=2,  # number of objectives
                         n_constr=4,  # number of constraints
                         xl=np.array([0.01, 0.01, 0.01, 0.01]),  # lower bounds
                         xu=np.array([10.0, 10.0, 10.0, 10.0]))  # upper bounds

    def _evaluate(self, x, out, *args, **kwargs):
        result = optimization_problem_test({
            'param1': x[0],
            'param2': x[1],
            'param3': x[2],
            'param4': x[3]
        })
        out["F"] = np.array(
            [result['objectives']['objective1'],
             result['objectives']['objective2']
             ])
        out["G"] = np.array([
            result['constraints']['constraint1'],
            result['constraints']['constraint2'],
            result['constraints']['constraint3'],
            result['constraints']['constraint4']
        ])


problem = Problem()

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.7, eta=30),
    mutation=PM(prob=0.2, eta=25),
    eliminate_duplicates=True
)


termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=30,
    n_max_gen=100,
    n_max_evals=100000
)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

npfile_path = os.path.join(folder_path, 'history.npy')
np.save(npfile_path, res.history)
X = res.X  # Design space values are
F = res.F  # Objective spaces values
G = res.G  # Constraint values
CV = res.CV  # Aggregated constraint violation
opt = res.opt  # The solutions as a Population object.
pop = res.pop  # The final Population

# Pareto front for "welded beam" problem
pymoo_scatter_plot = Scatter(title="Pareto front for welded beam problem")
pymoo_scatter_plot.add(get_problem("welded_beam").pareto_front(use_cache=False), plot_type="line", color="black")
pymoo_scatter_plot.add(F, facecolor="none", edgecolor="red", alpha=0.8, s=20)
pymoo_scatter_plot.save(os.path.join(folder_path, 'pareto_front.png'))

# # Objective Minimization Over Generations evaluation
# history = np.load('history_nsga2.npy', allow_pickle=True)
# objectives_over_time = [np.min(h.pop.get("F")) for h in history]
# plt.figure()
# plt.plot(objectives_over_time)
# plt.title('Objective Minimization Over Generations')
# plt.xlabel('Generation')
# plt.ylabel('Objective Value')
#
# # Find the best trade-off between two objectives F1 and F2 using Augmented Scalarization Function (ASF)
# approx_ideal = F.min(axis=0)
# approx_nadir = F.max(axis=0)
# nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
# weights = np.array([0.5, 0.5])
# decomp = ASF()
# i = decomp.do(nF, 1 / weights).argmin()
# print("Best regarding ASF: Point \ni = %s\nF = %s" % (i, F[i]))
# plt.figure(figsize=(7, 5))
# plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
# plt.title("Objective Space")
#
# plt.show()

    # TODO: save npy in results
    # TODO: git
    # TODO: SAVE G1
    # TODO: SAVE G2
    # TODO: SAVE G3
    # TODO: paraneters F G X names
    # TODO convergence