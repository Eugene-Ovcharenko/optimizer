import numpy as np
from pymoo.problems import get_problem
from utils.problem import init_procedure, Procedure
from random import random
from utils.global_variable import get_percent, get_problem_name


def optimization_problem(
        parameters: dict,
) -> dict:
    def optimization_problem_beam(
            problem,
            parameters: dict,
    ) -> dict:
        """
            Evaluates the 'welded_beam' optimization problem using given parameters and returns the
            objective values and constraint violations.

            Args:
                problem:
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

    def optimization_problem_test(
            problem,
            parameters: dict,
    ) -> dict:
        """
            Evaluates the 'welded_beam' optimization problem using given parameters and returns the
            objective values and constraint violations.

            Args:
                problem:
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
        curr_rand = random() * 100
        if curr_rand > get_percent():
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
        else:
            # param_array = np.array(list(parameters.values()))
            # result = problem.evaluate(param_array)
            # objective_values = result[0]
            # objectives_dict = {
            #     "objective1": objective_values[0],
            #     "objective2": objective_values[1]
            # }
            objectives_dict = {
                "objective1": 1000,
                "objective2": 1000
            }
            # constraint_values = result[1]
            # constraints_dict = {
            #     "constraint1": constraint_values[0],
            #     "constraint2": constraint_values[1],
            #     "constraint3": constraint_values[2],
            #     "constraint4": constraint_values[3]
            # }
            constraints_dict = {
                "constraint1": 100,
                "constraint2": 100,
                "constraint3": 100,
                "constraint4": 100
            }
            print('value losted')
        return {"objectives": objectives_dict, "constraints": constraints_dict}

    problem_name = get_problem_name()
    # print(f'Wrong optimization problem name: {problem_name}')
    if problem_name.lower() == 'test':
        problem = get_problem("welded_beam")
        return optimization_problem_test(problem=problem, parameters=parameters)
    elif problem_name.lower() == 'beam':
        problem = init_procedure(cpus=6, baseName='Beam')
        return optimization_problem_beam(problem=problem, parameters=parameters)
