import time
import tensorflow as tf
import pandas as pd
import whisper
import asyncio
import json
import os
import warnings
import sys
import os
import nest_asyncio

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


async def call_whisper2(File_List:str,Transcription_File_Path:str,Model_Name:str):

    file_path = File_List
    model = whisper.load_model(Model_Name)
    counter = 0

    try:
        transcription_information={}
        with open(file_path, "r") as file:
            Folders = json.load(file)
            for Folder in Folders:
                for key, path_list in Folders[Folder].items():

                    for path in path_list:
                        counter = counter +1
                        result = await asyncio.to_thread(model.transcribe, path)
                        transcription = f'{result["text"]}'  # Ensure correct string formatting
                        key = path
                        transcription_information[key]=transcription
                        if counter %100==0:
                            if not os.path.exists(Transcription_File_Path):
                                # Create and write the dictionary to the file
                                with open(Transcription_File_Path, "w") as file:
                                    json.dump(transcription_information, file, indent=4)  # Use JSON for a reloadable format
                                print(f"File created and dictionary saved to {Transcription_File_Path}    {counter/100}/210")
                                transcription_information = {}
                                time.sleep(10)
                            else:
                                with open(Transcription_File_Path, 'r') as file:
                                    data = json.load(file)

                                # Append transcription_information to the data
                                data.update(transcription_information)  # Use extend to append the contents of the list

                                # Write the updated data back to the file
                                with open(Transcription_File_Path, 'w') as file:
                                    json.dump(data, file, indent=4)  # Write the updated list to the file

                                print(f"File updated at {Transcription_File_Path}  {counter/100}/214")

    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}.")

async def call_whisper3(File_List:str,Transcription_File_Path:str,Model_Name:str):


    file_path = File_List
    model = whisper.load_model(Model_Name)
    counter = 0

    try:
        transcription_information={}
        with open(file_path, "r") as file:
            Folders = json.load(file)
            ac = 0
            for Folder in Folders:
                for key, path_list in Folders[Folder].items():

                    for path in path_list:
                        ac+=1

                        if ac>8297:
                          print(ac)
                          counter += 1
                          result = await asyncio.to_thread(model.transcribe, path)
                          transcription = f'{result["text"]}'  # Ensure correct string formatting
                          key = path
                          transcription_information[key]=transcription
                          if counter %100==0:

                              if not os.path.exists(Transcription_File_Path):
                                  # Create and write the dictionary to the file
                                  with open(Transcription_File_Path, "w") as file:
                                      json.dump(transcription_information, file, indent=4)  # Use JSON for a reloadable format
                                  print(f"File created and dictionary saved to {Transcription_File_Path}    {counter/100}/1200")
                                  transcription_information = {}
                                  time.sleep(10)
                              else:
                                  with open(Transcription_File_Path, 'r') as file:
                                      data = json.load(file)

                                  # Append transcription_information to the data
                                  data.update(transcription_information)  # Use extend to append the contents of the list

                                  # Write the updated data back to the file
                                  with open(Transcription_File_Path, 'w') as file:
                                      json.dump(data, file, indent=4)  # Write the updated list to the file

                                  print(f"File updated at {Transcription_File_Path}  {counter/100}/214")

    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}.")


def getList(Transcription_File_Path:str):
    if os.path.exists(Transcription_File_Path):
        # Create and write the dictionary to the file
        with open(Transcription_File_Path, 'r') as file:
            Transcription = json.load(file)
    keys = Transcription.keys()
    keys_list_transcribed = list(keys)
    return keys_list_transcribed




def data_cleaning(Transcription_File_Path_tiny:str,Transcription_File_Path_Base:str,Transcription_File_Path_Small:str,Transcription_File_Path_Medium:str,Transcription_File_Path_Large:str,File_List_Path:str,Data_Cleaning_File_Path:str):

    keys_list_transcribed_tiny = getList(Transcription_File_Path_tiny)
    keys_list_transcribed_base = getList(Transcription_File_Path_Base)
    keys_list_transcribed_small = getList(Transcription_File_Path_Small)
    keys_list_transcribed_medium = getList(Transcription_File_Path_Medium)
    keys_list_transcribed_large = getList(Transcription_File_Path_Large)

    if os.path.exists(Transcription_File_Path_tiny):
        File_List = []
        with open(File_List_Path, "r") as file:
            Folders = json.load(file)
            for Folder in Folders:
                for key, path_list in Folders[Folder].items():
                    for path in path_list:
                        File_List.append(path)
        print(File_List)


    # Find the maximum length of the two columns
    max_length = max(len(File_List), len(keys_list_transcribed_tiny),len(keys_list_transcribed_base),len(keys_list_transcribed_small),len(keys_list_transcribed_medium),len(keys_list_transcribed_large))

    # Pad the shorter column with None
    File_List += [None] * (max_length - len(File_List))
    keys_list_transcribed_tiny += [None] * (max_length - len(keys_list_transcribed_tiny))
    keys_list_transcribed_base += [None] * (max_length - len(keys_list_transcribed_base))
    keys_list_transcribed_small += [None] * (max_length - len(keys_list_transcribed_small))
    keys_list_transcribed_medium += [None] * (max_length - len(keys_list_transcribed_medium))
    keys_list_transcribed_large += [None] * (max_length - len(keys_list_transcribed_large))

    # Create a DataFrame
    data = {
        "Files": File_List,
        "Model (Tiny)": keys_list_transcribed_tiny,
        "Model (Base)": keys_list_transcribed_base,
        "Model (Small)": keys_list_transcribed_small,
        "Model (Medium)": keys_list_transcribed_medium,
        "Model (Large)": keys_list_transcribed_large,
        "condition":"=IF(AND(A2=B2, A2=C2, A2=D2, A2=E2, A2=F2), TRUE, FALSE)"


    }
    df = pd.DataFrame(data)
    df.to_excel(Data_Cleaning_File_Path, index=False)  # index=False avoids writing row indices

    print("Excel file created using pandas successfully!")


File_List_Path = "../File_List/File_List.json"
Transcription_File_Path = "../Transcrition/Transcrition"
Data_Cleaning_File_Path = "../Data_Cleaning/Data_Cleaning.xlsx"
data_cleaning(Transcription_File_Path+"_tiny.json",Transcription_File_Path+"_base.json",Transcription_File_Path+"_small.json",Transcription_File_Path+"_medium.json",Transcription_File_Path+"_large.json",File_List_Path,Data_Cleaning_File_Path)



