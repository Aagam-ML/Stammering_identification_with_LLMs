import os
import re
import json

def natural_key(s):
    # Split the filename into text and numbers for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# List all files in the folder and filter only files
def file_List(Path , File_Path):
    clom = [entry for entry in os.listdir(Path) if os.path.isdir(os.path.join(Path, entry))]
    clom = sorted(clom, key=natural_key)

    file_dict = {}

    for cl in clom:
        inside_clom = [entry for entry in os.listdir(Path + '/' + cl) if
                       os.path.isdir(os.path.join(Path + '/' + cl, entry))]
        inside_clom = sorted(inside_clom, key=natural_key)
        semiDic = {}

        for ins in inside_clom:
            file_names = [
                os.path.join(Path, cl, ins, f)
                for f in os.listdir(os.path.join(Path, cl, ins))
                if os.path.isfile(os.path.join(Path, cl, ins, f)) and not f.startswith("._")
                # Filter out files starting with "._"
            ]


            sorted_natural = sorted(file_names, key=natural_key)
            semiDic[ins] = sorted_natural
        file_dict[cl] = semiDic

    if not os.path.exists(File_Path):
        # Create and write the dictionary to the file
        with open(File_Path, "w") as file:
            json.dump(file_dict, file, indent=4)  # Use JSON for a reloadable format
        print(f"File created and dictionary saved to {File_Path}")
    else:
        print(f"File already exists at {File_Path}")







