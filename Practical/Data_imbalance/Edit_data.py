import json
import pandas as pd


def json_to_excel(json_data, output_file):
    """
    Convert JSON data to Excel format with three columns:
    1. Sequential Number
    2. File Path (key)
    3. Sentence (value)
    """
    # Create lists to store data
    numbers = []
    paths = []
    sentences = []

    # Extract data from JSON
    for i, (path, sentence) in enumerate(json_data.items(), start=1):
        numbers.append(i)
        paths.append(path)
        sentences.append(sentence)

    # Create DataFrame
    df = pd.DataFrame({
        'Number': numbers,
        'Path': paths,
        'Sentence': sentences
    })

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Excel file saved as {output_file}")


def add_counter_column(input_file, output_file):
    """
    Add a counter column starting from the second row and save to a new Excel file.
    """
    # Read the input Excel file
    df = pd.read_excel(input_file)

    # Add a counter column starting from the second row
    df.insert(0, 'Counter', range(1, len(df) + 1))

    # Save to the output Excel file
    df.to_excel(output_file, index=False)
    print(f"File saved as {output_file}")


# Main script
if __name__ == "__main__":
    # Path to your JSON file
    json_file = 'Transcrition_Tiny.json'  # Replace with your JSON file path

    # Load JSON data
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file}' does not exist.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{json_file}' is not a valid JSON file.")
        exit(1)

    # Convert JSON to Excel
    output_file = 'transcription.xlsx'  # Name of the output Excel file
    json_to_excel(data, output_file)

    input_file = 'SEP-28k_label.xlsx'  # Replace with your input file path
    output_file = 'output_data.xlsx'  # Replace with your desired output file path

    # Add counter column and save to new file
    add_counter_column(input_file, output_file)