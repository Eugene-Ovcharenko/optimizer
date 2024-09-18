import numpy as np
import os
import datetime
import time
import sys
import psutil as psu
from .logger_leaflet import log_message


# простой запуск консолькой команды.
# проверка на существование файла .lck - если есть, значит расчет еще идет.
def runabaqus(Path=None, jobName=None, InpFile=None, cpus=None):
    inputFile = 'abaqus job=' + str(jobName) + ' inp=' + str(InpFile) + ' cpus=' + str(cpus) + ' mp_mode=threads'
    log_message("InpFile CL is: >> " + inputFile)
    t0 = datetime.datetime.now()
    MatlabPath = os.getcwd()
    os.chdir(Path)
    outputs_args = os.system(inputFile)
    time.sleep(10)
    os.chdir(MatlabPath)
    message = 'ABAQUS complete'
    checked = False
    current_user = psu.users()[0].name
    if (os.path.exists(Path + '/' + jobName + '.lck')):
        log_message('-------------ABAQUS calculating-------------')
        while (os.path.exists(Path + '/' + jobName + '.lck')):

            t = datetime.datetime.now() - t0
            sec = t.seconds
            m = int(sec / 60) % 60
            h = int(sec / 3600)
            if m > 1 and not checked:
                for proc in psu.process_iter(['name', 'username']):
                    if 'pre' in proc.name() and current_user in proc.username():
                        os.system('pkill -n -9 pre')
                        message = 'ABAQUS terminated with error in pre'
                        break
                    if 'package' in proc.name() and current_user in proc.username():
                        os.system('pkill -n -9 package')
                        message = 'ABAQUS terminated with error in package'
                        break
                checked = True
            if m < 60:
                time.sleep(30)
            else:
                os.system('pkill -n -9 explicit')
                message = 'ABAQUS terminated due time'
                log_message('\t time costed: %3d:%2d:%2d\n-------------ABAQUS terminate-------------' % (h, m, np.mod(sec, 60)))
                break
        log_message('\n-------------ABAQUS complete-------------' )
    else:
        message = 'runanaqus error: InpFile submit failed'
        log_message('\n runanaqus error: InpFile submit failed\n')

    return message
    

def runabaqus_no_walltime(Path=None, jobName=None, InpFile=None, cpus=None):
    inputFile = 'abaqus job=' + str(jobName) + ' inp=' + str(InpFile) + ' cpus=' + str(cpus) + ' mp_mode=threads'
    log_message("InpFile CL is: >> " + inputFile)
    t0 = datetime.datetime.now()
    MatlabPath = os.getcwd()
    os.chdir(Path)
    outputs_args = os.system(inputFile)
    time.sleep(5)
    os.chdir(MatlabPath)
    if (os.path.exists(Path + '/' + jobName + '.lck')):
        log_message('-------------ABAQUS calculating-------------' % ())
        while (os.path.exists(Path + '/' + jobName + '.lck')):

            t = datetime.datetime.now() - t0
            sec = t.seconds
            m = int(sec / 60) % 60
            h = int(sec / 3600)

        message = 'ABAQUS complete'
        log_message('\r\t time costed: %3d:%2d:%2d\n-------------ABAQUS complete-------------' % (h, m, np.mod(sec, 60)))
    else:
        message = 'runanaqus error: InpFile submit failed'
        log_message('\n runanaqus error: InpFile submit failed\n' % ())

    return message


def runabaqus_minute_walltime(Path=None, jobName=None, InpFile=None, cpus=None, minutes=5):
    inputFile = 'abaqus job=' + str(jobName) + ' inp=' + str(InpFile) + ' cpus=' + str(cpus) + ' mp_mode=threads'
    log_message("InpFile CL is: >> " + inputFile)
    t0 = datetime.datetime.now()
    MatlabPath = os.getcwd()
    os.chdir(Path)
    primary_stdout = sys.stdout
    primary_stderr = sys.stderr
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull

        outputs_args = os.system(inputFile)

    sys.stdout = primary_stdout
    sys.stderr = primary_stderr
    time.sleep(0.5)
    os.chdir(MatlabPath)
    if (os.path.exists(Path + '/' + jobName + '.lck')):
        log_message('-------------ABAQUS calculating-------------' % ())
        while (os.path.exists(Path + '/' + jobName + '.lck')):

            t = datetime.datetime.now() - t0
            sec = t.seconds
            m = int(sec / 60) % 60
            h = int(sec / 3600)
            if m < minutes:
                time.sleep(0.5)
                log_message('\r\t time costed: %3d:%2d:%2d' % (h, m, np.mod(sec, 60)))
            else:
                os.system('pkill -9 standard')
                message = 'ABAQUS terminate'
                log_message('\t time costed: %3d:%2d:%2d\n-------------ABAQUS terminate-------------' % (h, m, np.mod(sec, 60)))
                break

        message = 'ABAQUS complete'
        log_message('\r\t-------------ABAQUS complete-------------')
    else:
        message = 'runanaqus error: InpFile submit failed'
        log_message('\n runanaqus error: InpFile submit failed')

    return message
