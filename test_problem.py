import numpy as np
from pymoo.problems import get_problem

problem = get_problem("welded_beam")


def optimization_problem_test(
        parameters: dict,
) -> dict:
    """
        Evaluates the 'welded_beam' optimization problem using given parameters and returns the
        objective values and constraint violations.

        Args:
            parameters (dict): A dictionary of parameter names and their respective values.
                Expected keys are 'param1', 'param2', 'param3', and 'param4'.

        Returns:
            A dictionary with two keys:
                - 'objectives': A sub-dictionary containing:
                    - 'objective1' (float): The first objective value.
                    - 'objective2' (float): The second objective value.
                - 'constraints': A sub-dictionary containing:
                    - 'constraint1' (float): The first constraint violation.
                    - 'constraint2' (float): The second constraint violation.
                    - 'constraint3' (float): The third constraint violation.
                    - 'constraint4' (float): The fourth constraint violation.

    """
    param_array = np.array(list(parameters.values()))
    result = problem.evaluate(param_array)
    objective_values = result[0]
    objectives_dict = {
        "objective1": objective_values[0],
        "objective2": objective_values[1]
    }
    constraint_values = result[1]
    constraints_dict = {
        "constraint1": constraint_values[0],
        "constraint2": constraint_values[1],
        "constraint3": constraint_values[2],
        "constraint4": constraint_values[3]
    }

    return {"objectives": objectives_dict, "constraints": constraints_dict}
