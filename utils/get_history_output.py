import os


# по туториалам из гугла: в папке с файлами расчета пишем в файл что мы хотим дважды (не знаю почему, так надо),
# через абакус открываем скрипт питона odbHistoryOutput_4perField, который использует внутренние пакеты абакуса
# (поэтому и открываем через него)) и парсим .odb
# скрипт лежит в папке с расчетами абакуса, не удалять
def get_history_output(pathName = None, odbFileName = None, cpu=-1):
    # prepare result folder
    if not os.path.exists(pathName + 'results3/'):
        os.makedirs(pathName + 'results3/')

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
    return


def get_history_output_contact(pathName = None, odbFileName = None, cpu=-1):
    # prepare result folder
    if not os.path.exists(pathName + 'result/'):
        os.makedirs(pathName + 'result/')
    reqFile = str(pathName) + '/req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    reqFile = './req.txt'
    fid = open(reqFile, 'wt')
    fid.write('%s,%s' % (pathName, odbFileName))
    fid.close()
    if cpu == -1:
        consoleCommand = 'abaqus cae noGUI=' + str(pathName) + '/odbHistoryOutput_ShellContact.py'
    else:
        consoleCommand = 'abaqus cae noGUI=' + str(pathName) + '/odbHistoryOutput_ShellContact_'+str(cpu)+'.py'
    os.system(consoleCommand)
    return