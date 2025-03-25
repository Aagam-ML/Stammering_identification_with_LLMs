import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load the dataset from an Excel file
file_path = '/Final_approaches/Diagnostic Explanation Generation (DEG)/balanced_dataset_with_all_combinations_wav2vec.xlsx'  # Update with your Excel file path
df = pd.read_excel(file_path)

# Extract the 'Transcription' column
texts = df['Transcription'].tolist()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define batch size
batch_size = 50


# Tokenize the text in batches
def tokenize_batch(batch_texts):
    return tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)


# Process the text in batches
all_pooled_outputs = []
processed_texts = 0

total_texts = len(texts)

for i in range(0, total_texts, batch_size):
    batch_texts = texts[i:i + batch_size]

    # Filter out None or empty strings
    batch_texts = [text for text in batch_texts if isinstance(text, str) and text.strip()]

    if not batch_texts:
        continue  # Skip empty batches

    inputs = tokenize_batch(batch_texts)

    with torch.no_grad():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    all_pooled_outputs.append(pooled_output)

    processed_texts += len(batch_texts)
    percentage = (processed_texts / total_texts) * 100
    print(f'Processing: {percentage:.2f}% completed')

# Concatenate all pooled outputs
if all_pooled_outputs:
    total_pooled_output = torch.cat(all_pooled_outputs, dim=0)
else:
    raise ValueError("No valid text data found in the 'Transcription' column.")

# Save the features to a file
output_file_path = '../DeepNeuralNetworkApproach/bert_features_779.pt'
torch.save(total_pooled_output, output_file_path)
print(f"BERT feature vectors saved to {output_file_path}")