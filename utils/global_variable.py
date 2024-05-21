class Params:
    percent = 0
    problem_name = ''

    def __init__(self, percent, problem_name):
        self.percent = percent
        self.problem_name = problem_name

params = Params(0, '')

def set_percent(val):
    params.percent = val

def set_problem_name(val):
    params.problem_name = val

def get_percent():
    return params.percent

def get_problem_name():
    return params.problem_name