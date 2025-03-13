import torch

# Replace 'path_to_your_file.pt' with the path to your .pt file
feature_vectors1 = torch.load('/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_1.pt',weights_only=True)
feature_vectors2 = torch.load('/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_2.pt',weights_only=True)
feature_vectors3 = torch.load('/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_3.pt',weights_only=True)
feature_vectors4 = torch.load('/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_4.pt',weights_only=True)
feature_vectors5 = torch.load('/Volumes/HDD/Stammering_identification/Stammering_identification_with_LLMs/Practical/BERT/bert_features_5.pt',weights_only=True)

# Assuming your tensor is already in the right shape (samples, features)
# If it's not, you might need to reshape or perform operations to get it into this form.
feature_vectors_np_1 = feature_vectors1.numpy()  # Convert PyTorch tensor to NumPy array
feature_vectors_np_2 = feature_vectors2.numpy()  # Convert PyTorch tensor to NumPy array
feature_vectors_np_3 = feature_vectors3.numpy()  # Convert PyTorch tensor to NumPy array
feature_vectors_np_4 = feature_vectors4.numpy()  # Convert PyTorch tensor to NumPy array
feature_vectors_np_5 = feature_vectors5.numpy()  # Convert PyTorch tensor to NumPy array



from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

pca = PCA(n_components=2)
reduced_vectors1 = pca.fit_transform(feature_vectors_np_1)  # Use the NumPy array here
reduced_vectors2 = pca.fit_transform(feature_vectors_np_2)  # Use the NumPy array here
reduced_vectors3 = pca.fit_transform(feature_vectors_np_3)  # Use the NumPy array here
reduced_vectors4 = pca.fit_transform(feature_vectors_np_4)  # Use the NumPy array here
reduced_vectors5 = pca.fit_transform(feature_vectors_np_5)  # Use the NumPy array here


plt.figure(figsize=(8, 6))
plt.scatter(reduced_vectors1[:, 0], reduced_vectors1[:, 1], alpha=0.5,color="red")
plt.scatter(reduced_vectors2[:, 0], reduced_vectors2[:, 1], alpha=0.5,color="orange")
plt.scatter(reduced_vectors3[:, 0], reduced_vectors2[:, 1], alpha=0.5,color="blue")
plt.scatter(reduced_vectors4[:, 0], reduced_vectors2[:, 1], alpha=0.5,color="lightgreen")



plt.title('PCA of BERT Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
