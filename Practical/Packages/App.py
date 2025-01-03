import os
from dotenv import load_dotenv #for environment veriable
import asyncio
import time
from langchain_cohere.llms import Cohere #cohere is a llm not a open source but provides few free tokens
from langchain_core.prompts import ChatPromptTemplate #prompt needed to run the program
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere #chatting with cohere llm using langchain
import werpy
import whisper
import re

from openpyxl import Workbook, load_workbook
import PyPDF2
import ssl
from dotenv import load_dotenv
load_dotenv()



######################################################
if os.path.exists('Assumption/Assumption.xlsx'):
    os.remove('Assumption/Assumption.xlsx')
if os.path.exists('GroundTruth/GroundTruth.xlsx'):
    os.remove('GroundTruth/GroundTruth.xlsx')

#######################################################



######################################################
ssl._create_default_https_context = ssl._create_unverified_context
#######################################################


######################################################
path = 'clom'
def natural_key(s):
    # Split the filename into text and numbers for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# List all files in the folder and filter only files
file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
print(file_names)
#######################################################



######################################################
path = 'clom'

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

print(file_dict)
#######################################################


######################################################
model = whisper.load_model("tiny")
ModelInformation={}
#######################################################



######################################################
import asyncio

counter = 0
ModelInformation = {}  # Initialize your dictionary



import asyncio

async def process_files():
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

ModelInfor = asyncio.run(process_files())
print(ModelInfor)

# Print or use the ModelInformation dictionary as needed
print(ModelInfor)
print(ModelInformation)

#######################################################


# Print or use the ModelInformation dictionary as needed
file_name = "Assumption/Assumption.xlsx"
file_name2 = "GroundTruth/GroundTruth.xlsx"

#######################################################

#############
def LLM_search(BOKS):
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")
    LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system"," |Prolongation | Block | SoundRep | WordRep | DifficultToUnderstand | Interjection | These all are stammering features. from the data i provide you have to tell me which all features are applies to the text , provide ans in excel format like this |Prolongation: 0|  0 | SoundRep : 0 | WordRep : 0 | DifficultToUnderstand : 0 | Interjection : 1 |"),
                   ("user","Question:{question}")
        ]
    )

    llm = ChatCohere()
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser

    llm_result=chain.invoke({"question":BOKS})
    return llm_result

############


########################################################
# Print or use the ModelInformation dictionary as needed
if os.path.exists(file_name):
    # Load the existing workbook
    workbook = load_workbook(file_name)
    sheet = workbook.active
    print("File exists. Opened in append mode.")
else:
    # Create a new workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"  # Optionally, rename the sheet
    print("File created.")

# Example of adding data to the workbook
# sheet.append(["Prolongation", "Block", "SoundRep", "WordRep", "DifficultToUnderstand", "Interjection"])
sheet.append(["Prolongation", "Block", "SoundRep", "WordRep", "DifficultToUnderstand", "Interjection"])
counter = 0
for itemNum, itemName in ModelInfor.items():
    counter = counter + 1
    if counter % 9 == 0:
        time.sleep(60)

    print(itemName)
    bokl = LLM_search(itemName)
    data = list(map(int, re.findall(r'\b\d+\b', bokl)))
    sheet.append(data)
    print(f'{itemNum} : {bokl}')

# Save the file
workbook.save(file_name)
print(f"Data written to {file_name}")
#######################################################




#######################################################
import csv

# Define the file name
file_name = "Data/SEP-28k_labels.csv"  # Replace with your actual file name

# List to hold the extracted rows
extracted_data = []

# Open the file and read it
with open(file_name, 'r') as file:
    reader = csv.reader(file)  # Create a CSV reader
    headers = next(reader)  # Read the header row (optional)

    # Iterate through each row in the CSV
    for i, row in enumerate(reader):
        if i >=30:  # Stop after 40 rows
            break
        # Fetch columns 3, 4, 5, 6, 7 (indexes 2, 3, 4, 5, 6)
        extracted_row = row[7:13]
        # Access each element separately and convert each to int, if needed.
        Converted_Data = []
        for row in extracted_row:
           Converted_Data.append(int(row))
          # Slicing to get columns 3 to 7
        extracted_data.append(Converted_Data)
#######################################################








#######################################################
if os.path.exists(file_name2):
    # Load the existing workbook
    workbook = load_workbook(file_name2)
    sheet = workbook.active
    print("File exists. Opened in append mode.")
else:
    # Create a new workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"  # Optionally, rename the sheet
    print("File created.")

# Example of adding data to the workbook
sheet.append(["Prolongation", "Block", "SoundRep", "WordRep", "DifficultToUnderstand", "Interjection"])
for item in extracted_data:
    sheet.append(item)

# Save the file
workbook.save(file_name2)
print(f"Data written to {file_name2}")
#######################################################

#######################################################
import pandas as pd

# Load the Excel files
file1 = "Assumption/Assumption.xlsx"  # Replace with the actual file path
file2 = "GroundTruth/GroundTruth.xlsx"  # Replace with the actual file path

# Load the sheets (by default, the first sheet is loaded)
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Define the columns you want to compare (assumes column names are the same)
columns_to_compare = ["Prolongation", "Block", "SoundRep", "WordRep", "DifficultToUnderstand",
                      "Interjection"]  # Replace with your column names

# Check if both files have the same number of rows and columns
if len(df1) != len(df2):
    print("The files have different numbers of rows.")
elif len(df1.columns) != len(df2.columns):
    print("The files have different numbers of columns.")
else:
    # Initialize counters
    total_matches = 0
    total_values = 0

    # Compare each specified column in both dataframes
    for column in columns_to_compare:
        if column in df1.columns and column in df2.columns:
            # Count the number of matching values in this column
            matches = (df1[column] == df2[column]).sum()
            total_matches += matches
            total_values += len(df1[column])

            # Print column-wise accuracy
            column_accuracy = matches / len(df1[column]) * 100
            print(f"Accuracy for {column}: {column_accuracy:.2f}%")
        else:
            print(f"Column '{column}' not found in one of the files.")
    print(total_matches)
    print(total_values)
    # Calculate overall accuracy
    overall_accuracy = total_matches / total_values * 100
    print(f"Overall accuracy: {overall_accuracy:.2f}%")

#######################################################


