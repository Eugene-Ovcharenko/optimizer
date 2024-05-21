import os
from .global_variable import get_problem_name, get_cpus


# по туториалам из гугла: в папке с файлами расчета пишем в файл что мы хотим дважды (не знаю почему, так надо),
# через абакус открываем скрипт питона odbHistoryOutput_4perField, который использует внутренние пакеты абакуса
# (поэтому и открываем через него)) и парсим .odb
# скрипт лежит в папке с расчетами абакуса, не удалять
def get_history_output_beam(pathName=None, odbFileName=None):
    # prepare result folder
    if not os.path.exists(pathName + 'results/'):
        os.makedirs(pathName + 'results/')

    reqFile = str(pathName) + '/req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    reqFile = './req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    consoleCommand = 'abaqus cae noGUI=' + str(pathName) + 'odbHistoryOutput.py'
    os.system(consoleCommand)


def get_history_output_single(pathName=None, odbFileName=None, cpus=-1):
    # prepare result folder
    if not os.path.exists(pathName + 'results/'):
        os.makedirs(pathName + 'results/')

    reqFile = str(pathName) + '/req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    reqFile = './req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    consoleCommand = 'abaqus cae noGUI=' + str(pathName) + 'odbHistoryOutput_4perField.py'
    os.system(consoleCommand)


def get_history_output_contact(pathName=None, odbFileName=None, cpu=-1):
    # prepare result folder
    if not os.path.exists(pathName + 'results/'):
        os.makedirs(pathName + 'results/')
    reqFile = str(pathName) + '/req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    reqFile = './req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    if cpu == -1:
        consoleCommand = 'abaqus cae noGUI=' + str(pathName) + 'odbHistoryOutput_ShellContact.py'
    else:
        consoleCommand = 'abaqus cae noGUI=' + str(pathName) + 'odbHistoryOutput_ShellContact_' + str(cpu) + '.py'
    os.system(consoleCommand)


def get_history_output(pathName=None, odbFileName=None, cpus=-1):
    problem_name = get_problem_name()
    if problem_name.lower() == 'leaflet_single':
        get_history_output_single(pathName=pathName, odbFileName=odbFileName)
    elif problem_name.lower() == 'leaflet_contact':
        get_history_output_contact(pathName=pathName, odbFileName=odbFileName, cpu=cpus)
    elif problem_name.lower() == 'beam':
        get_history_output_beam(pathName=pathName, odbFileName=odbFileName)
