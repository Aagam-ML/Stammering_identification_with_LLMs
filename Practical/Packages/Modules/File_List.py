import os
import re

def natural_key(s):
    # Split the filename into text and numbers for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# List all files in the folder and filter only files
def file_List(path):
    file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print(file_names)



    clom = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
    clom = sorted(clom, key=natural_key)

    file_dict = {}

    for cl in clom:
        inside_clom = []
        inside_clom = [entry for entry in os.listdir(path + '/' + cl) if
                       os.path.isdir(os.path.join(path + '/' + cl, entry))]
        inside_clom = sorted(inside_clom, key=natural_key)
        semiDic = {}

        for ins in inside_clom:
            file_names = [f for f in os.listdir(path + '/' + cl + '/' + ins + '/') if
                          os.path.isfile(os.path.join(path + '/' + cl + '/' + ins + '/', f))]
            sorted_natural = sorted(file_names, key=natural_key)
            semiDic[ins] = sorted_natural
        file_dict[cl] = semiDic
    return file_dict