import os
import sys
from glob2 import glob


def purgeFiles(endPath: str, partName: str, pathToAbaqus: str, jobName: str):
    if str.upper(sys.platform) == 'WIN32':
        rm_com = 'del '
        os.system('rmdir /s /q ' + endPath + partName[0:-5].upper())
        os.system(pathToAbaqus + jobName + ".sim* ")
    else:
        rm_com = 'rm -r '
        os.system(rm_com + endPath + partName[0:-5].upper())
        if os.path.exists(pathToAbaqus + jobName + ".simdir"):
            os.system(rm_com + pathToAbaqus + jobName + ".simdir")

    # Get all files matching the pattern
    all_files = glob(os.path.join(pathToAbaqus, jobName) + '*')

    # Filter out files with extensions you want to exclude
    files_to_delete = [
        f for f in all_files
        if not (f.endswith(".py") or f.endswith(".odb") or f.endswith(".txt"))
    ]

    # Delete the filtered files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except Exception as e:
            pass
    all_files = glob(os.path.join(pathToAbaqus, 'explicit*'))
    for file_path in all_files:
        try:
            os.remove(file_path)
        except Exception as e:
            pass
    # comLine = rm_com + pathToAbaqus + jobName + "* " + pathToAbaqus + "explicit*"
    # os.system(comLine)
    # for files in glob(pathToAbaqus + '/' + jobName + '*'):
    #    if files != pathToAbaqus + jobName + '.odb':
    #        os.remove(files)
    # for file in glob('./abaqus.rp*'):
    #     os.remove(file)

    # + jobName + ".msg " + jobName + ".sta "