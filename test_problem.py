import numpy as np
from pymoo.problems import get_problem
from utils.problem import init_procedure, Procedure

problem = init_procedure(cpus=6, baseName='Beam')


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
    result = Procedure.run_procedure(self=problem, params=param_array)
    objective_values = result.get('objectives')
    objectives_dict = {
        "Displacement": objective_values.get('Displacement'),
        "Mass": objective_values.get('Mass')
    }
    constraint_values = result.get('constraints')
    constraints_dict = {
        "THK_constr": constraint_values.get('THK_constr'),
        "Width_constr": constraint_values.get('Width_constr'),
        "Smax_constr": constraint_values.get('Smax_constr')
    }

    return {"objectives": objectives_dict, "constraints": constraints_dict}
