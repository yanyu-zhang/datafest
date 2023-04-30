import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

# Install necessary libraries
# !pip install sentence-transformers
# !pip install scikit-learn

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your group of questions
questions = [
    "What is your favorite color?",
    "How old are you?",
    "What's your favorite animal?",
    "What is your favorite food?",
    "How tall are you?",
    "What is the color you like the most?",
    "What do you like to eat the most?",
    "What kind of animal do you prefer?",
    "How many years have you lived?",
    "What is your height?"
]

# Encode the questions using the SentenceTransformer model
question_embeddings = model.encode(questions)

# Perform clustering using K-means with the optimal k
optimal_k = 3  # Based on the Elbow Method plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(question_embeddings)

# Find the most representative question for each cluster
closest_questions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, question_embeddings)
representative_questions = [questions[idx] for idx in closest_questions]

# Print the results
print("Clusters and their representative questions:")
for idx, question in enumerate(representative_questions):
    print(f"Cluster {idx + 1}: {question}")

print("\nCluster assignments:")
for question, cluster in zip(questions, cluster_labels):
    print(f"{question} -> Cluster {cluster + 1}")
