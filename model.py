import os
import pandas as pd
from kneed import KneeLocator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Install necessary libraries
# !pip install sentence-transformers
# !pip install scikit-learn

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

question_text = pd.read_csv('data/question_text.csv')

question_by_category = question_text.groupby('SubcategoryUno')
selected_subcategory = "E9F87919-0CD5-4D9D-AAC6-3CBC49132F1D"
questions = question_by_category.get_group(selected_subcategory)['PostText'].values

# Encode the questions using the SentenceTransformer model
question_embeddings = model.encode(questions)

# print the number unique questions
print(f"Number of questions: {len(questions)}")
max_cluster_size = min(30, len(questions) // 50)
min_cluster_size = 5

# Determine the optimal number of clusters using the elbow method
sse = []
range_n_clusters = range(min_cluster_size, max_cluster_size)
for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(question_embeddings)
    sse.append(kmeans.inertia_)

knee_locator = KneeLocator(range_n_clusters, sse, curve='convex', direction='decreasing')
optimal_k = knee_locator.elbow
print(f"The optimal number of clusters is {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(question_embeddings)

# Find the most representative question for each cluster
closest_questions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, question_embeddings)
representative_questions = [questions[idx] for idx in closest_questions]

# store the model
import pickle
if os.path.exists('model.pkl'):
    os.remove('model.pkl')
with open('model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)