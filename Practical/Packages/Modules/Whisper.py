import whisper
import asyncio
import json
import os
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


async def call_whisper(file_dict:dict , path:str):

    print("entered in whisper model")
    model = whisper.load_model("tiny")
    counter = 0
    ModelInformation = {}  # Initialize the dictionary here
    for key1, value1 in file_dict.items():
        for key2, value2 in value1.items():
            for i, file in enumerate(value2, start=0):
                counter += 1
                if counter <= 30:
                        # Use asyncio.to_thread to run the synchronous function in a thread
                    result = await asyncio.to_thread(model.transcribe, f"{path}/{key1}/{key2}/{file}")
                    result = f'"""{result["text"]}"""'  # Ensure correct string formatting
                    key = file
                    print(result)
                    ModelInformation[key] = result
        return ModelInformation

    # Entry point for running the function
async def call_whisper2(File_List:dict,Transcription_File_Path:str):

    file_path = File_List
    model = whisper.load_model("tiny")
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
                        if counter %10==0 or counter ==9:
                            if not os.path.exists(Transcription_File_Path):
                                # Create and write the dictionary to the file
                                with open(Transcription_File_Path, "w") as file:
                                    json.dump(transcription_information, file, indent=4)  # Use JSON for a reloadable format
                                print(f"File created and dictionary saved to {Transcription_File_Path}")
                                transcription_information = {}
                            else:
                                with open(Transcription_File_Path, 'r') as file:
                                    data = json.load(file)

                                # Append transcription_information to the data
                                data.update(transcription_information)  # Use extend to append the contents of the list

                                # Write the updated data back to the file
                                with open(Transcription_File_Path, 'w') as file:
                                    json.dump(data, file, indent=4)  # Write the updated list to the file

                                print(f"File already exists at {Transcription_File_Path}")

    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in {file_path}.")


