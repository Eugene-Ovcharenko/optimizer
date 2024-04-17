import numpy as np
import matplotlib.pyplot as plt
from test_problem import optimization_problem_test


# Determine non-dominated (Pareto optimal) points
def is_non_dominated(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


params = [
    {'param1': 0.1 + i * 0.01,
     'param2': 0.1 + i * 0.02,
     'param3': 0.1 + i * 0.03,
     'param4': 0.1 + i * 0.04}
    for i in range(100)
]

results = [optimization_problem_test(p) for p in params]

# Extracting data for plotting
objective1 = [r['objectives']['objective1'] for r in results]
objective2 = [r['objectives']['objective2'] for r in results]
constraints = [sum([r['constraints'][k] for k in r['constraints']]) for r in results]

scores = np.array(list(zip(objective1, objective2)))
non_dominated = is_non_dominated(scores)

# Plotting objectives
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(objective1, objective2, c='gray', label='Dominated')
plt.scatter(np.array(objective1)[non_dominated], np.array(objective2)[non_dominated], c='red',
            label='Non-Dominated (Pareto Front)')
plt.title('Objective Space')
plt.xlabel('Objective 1 (Cost)')
plt.ylabel('Objective 2 (Deflection)')
plt.legend()

# Plotting constraints
plt.subplot(1, 2, 2)
plt.plot(constraints, label='Total Constraint Violations')
plt.title('Constraint Violations Over Parameters')
plt.xlabel('Parameter Set Index')
plt.ylabel('Total Constraint Violation')
plt.legend()

plt.tight_layout()
plt.show()
