import whisper
import asyncio


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


