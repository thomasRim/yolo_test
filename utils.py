import os


def fileExist(filePath):
    if not os.path.isfile(filePath):
        print("Input file ", filePath, " doesn't exist")
        return False
    else:
        return True


def folderExist(filePath):
    if not os.path.isdir(filePath):
        print("Input folder ", filePath, " doesn't exist")
        return False
    else:
        return True
