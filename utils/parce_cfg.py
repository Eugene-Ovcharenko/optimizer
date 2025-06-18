import pathlib
import sys
from typing import Tuple, Any, List

from omegaconf import DictConfig
from utils.global_variable import *


def parce_cfg(cfg:DictConfig, globalPath=None) -> tuple[dict, list[str], list[str]]:
    parameters = dict()
    objectives = list()
    error_count = 0

    print(f'Cfg parsing:')
    # check parameters, objectives and constraints
    if hasattr(cfg, 'parameters'):
        parameters = {k: tuple(v) for k, v in cfg.parameters.items()}
    else:
        print('No attr \'parameters\'. Exit')
        error_count += 1

    if hasattr(cfg, 'objectives'):
        objectives = list(cfg.objectives)
    else:
        print('No attr \'objectives\'. Exit')
        error_count += 1

    if hasattr(cfg, 'constraints'):
        constraints = cfg.constraints
    else:
        print('No attr \'constraints\'. Set empty')
        constraints = []

    # check FEA-related parameters
    if hasattr(cfg.Abaqus, 'abq_cpus'):
        set_cpus(cfg.Abaqus.abq_cpus)
    else:
        print('No attr \'Abaqus.abq_cpus\'. Set default - 1')
        set_cpus(1)

    if hasattr(cfg.Abaqus, 'tangent_behavior'):
        set_tangent_behavior(cfg.Abaqus.tangent_behavior)
    else:
        print('No attr \'Abaqus.tangent_behavior\'. Set default - 0.2')
        set_tangent_behavior(0.2)

    if hasattr(cfg.Abaqus, 'normal_behavior'):
        set_normal_behavior(cfg.Abaqus.normal_behavior)
    else:
        print('No attr \'Abaqus.normal_behavior\'. Set default - 0.2')
        set_normal_behavior(0.2)

    # check problem-related parameters
    if hasattr(cfg.problem_definition, 'DIA'):
        set_DIA(float(cfg.problem_definition.DIA))
    else:
        print('No attr \'problem_definition.DIA\'. Set default - 23 mm')
        set_DIA(23)

    if hasattr(cfg.problem_definition, 'Lift'):
        set_Lift(float(cfg.problem_definition.Lift))
    else:
        print('No attr \'problem_definition.DIA\'. Set default - 0 mm')
        set_Lift(0.)

    if hasattr(cfg.problem_definition, 'SEC'):
        set_SEC(float(cfg.problem_definition.SEC))
    else:
        print('No attr \'problem_definition.SEC\'. Set default - 119')
        set_SEC(119)

    if hasattr(cfg.problem_definition, 'mesh_step'):
        set_mesh_step(float(cfg.problem_definition.mesh_step))
    else:
        print('No attr \'problem_definition.mesh_step\'. Set default - 0.35')
        set_mesh_step(0.35)

    if hasattr(cfg.problem_definition, 'position'):
        set_valve_position(cfg.problem_definition.position)
    else:
        print('No attr \'problem_definition.position\'. Set default - \'ao\'')
        set_valve_position('ao')

    if hasattr(cfg.problem_definition, 'problem_name'):
        set_problem_name(cfg.problem_definition.problem_name)
    else:
        print('No attr \'problem_definition.problem_name\'. Set default - \'leaflet_contact\'')
        set_problem_name('leaflet_contact')

    if hasattr(cfg.problem_definition, 'name'):
        set_base_name(cfg.problem_definition.name)
    else:
        print('No attr \'problem_definition.name\'. Set default - \'Forgotten name\'')
        set_base_name('Forgotten name')


    # check material-related parameters
    if hasattr(cfg.problem_definition.material, 'material_definition_type'):
        if cfg.problem_definition.material.material_definition_type.lower() in ['linear', 'polynomial', 'ortho']:
            set_material_type(cfg.problem_definition.material.material_definition_type)
        else:
            print('No attr \'problem_definition.material.material_definition_type\'. Set default - \'linear\'')
            set_material_type('linear')

    if cfg.problem_definition.material.material_definition_type.lower() in ['linear', 'polynomial']:
        if hasattr(cfg.problem_definition.material, 'EM') \
                and not hasattr(cfg.problem_definition.material, 'material_csv_path'):
            set_EM(float(cfg.problem_definition.material.EM))
            set_material_csv_path(str(None))
        elif not hasattr(cfg.problem_definition.material, 'EM') \
                and hasattr(cfg.problem_definition.material, 'material_csv_path'):
            set_EM(-1)
            set_material_csv_path(str(cfg.problem_definition.material.material_csv_path))
        elif hasattr(cfg.problem_definition.material, 'EM') \
                and hasattr(cfg.problem_definition.material, 'material_csv_path'):
            if cfg.problem_definition.material.EM > 0:
                print(f"\t\tYou entered positive EM = {cfg.problem_definition.material.EM}."
                      f" Using linear model. Either delete this row or set -1")
                set_EM(float(cfg.problem_definition.material.EM))
                set_material_csv_path(str(None))
            else:
                set_EM(-1)
                set_material_csv_path(str(cfg.problem_definition.material.material_csv_path))
        else:
            print('No attr \'problem_definition.material.SEC\'. Set default - 2 MPa')
            set_EM(2)
            set_material_csv_path(str(None))
    else:
        if cfg.problem_definition.material.material_definition_type.lower() in ['ortho']:
            if hasattr(cfg.problem_definition.material, 'ortho_coeffs_E'):
                set_e_coeffs(list(cfg.problem_definition.material.ortho_coeffs_E))
            else:
                print('No attr \'problem_definition.material.ortho_coeffs_E\'. Exit')
                error_count += 1
            if hasattr(cfg.problem_definition.material, 'ortho_coeffs_poisson'):
                set_poisson_coeffs(list(cfg.problem_definition.material.ortho_coeffs_poisson))
            else:
                print('No attr \'problem_definition.material.ortho_coeffs_poisson\'. Exit')
                error_count += 1

    if hasattr(cfg.problem_definition.material, 'poisson_coeff') \
            and cfg.problem_definition.material.material_definition_type.lower() in ['linear', 'polynomial']:
        set_poisson_coeffs(cfg.problem_definition.material.poisson_coeff)
    elif cfg.problem_definition.material.material_definition_type.lower() in ['linear', 'polynomial']:
        print('No attr \'problem_definition.material.poisson_coeff\'. Set default - 0.495 ')
        set_poisson_coeffs(0.495)


    if hasattr(cfg.problem_definition.material, 'Dens'):
        set_density(float(cfg.problem_definition.material.Dens))
    else:
        print('No attr \'problem_definition.material.Dens\'. Set default - 1e-9 Tonn/mm3')
        set_density(float(1e-9))

    if hasattr(cfg.problem_definition.material, 'material_name'):
        set_material_name(cfg.problem_definition.material.material_name)
    else:
        print('No attr \'problem_definition.material.material_name\'. Set default - \'Forgotten material name\'')
        set_material_name('Forgotten material name')

    if hasattr(cfg.problem_definition.material, 's_lim'):
        set_s_lim(float(cfg.problem_definition.material.s_lim))
    else:
        print('No attr \'problem_definition.material.s_lim\'. Set default - 5 MPa')
        set_s_lim(5)


    # check globalPath
    if globalPath:
        set_global_path(globalPath)
    else:
        print('Forgot to set globalPath in this function. Exit program')
        error_count += 1

    # check optimizer settings
    if not hasattr(cfg.optimizer, 'pop_size'):
        print('Forgot to set \'optimizer.pop_size\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer, 'offsprings'):
        print('Forgot to set \'optimizer.offsprings\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer, 'crossover_chance'):
        print('Forgot to set \'optimizer.crossover_chance\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer, 'mutation_chance'):
        print('Forgot to set \'optimizer.mutation_chance\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer, 'mutation_eta'):
        print('Forgot to set \'optimizer.mutation_eta\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer.termination_parameters, 'xtol'):
        print('Forgot to set \'optimizer.termination_parameters.xtol\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer.termination_parameters, 'cvtol'):
        print('Forgot to set \'optimizer.termination_parameters.cvtol\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer.termination_parameters, 'ftol'):
        print('Forgot to set \'optimizer.termination_parameters.ftol\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer.termination_parameters, 'period'):
        print('Forgot to set \'optimizer.termination_parameters.period\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer.termination_parameters, 'n_max_gen'):
        print('Forgot to set \'optimizer.termination_parameters.n_max_gen\'. Exit program')
        error_count += 1
    if not hasattr(cfg.optimizer.termination_parameters, 'n_max_evals'):
        print('Forgot to set \'optimizer.termination_parameters.n_max_evals\'. Exit program')
        error_count += 1

    if error_count > 0:
        sys.exit()

    return parameters, objectives, constraints