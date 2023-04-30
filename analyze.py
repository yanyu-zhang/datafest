import pickle
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

# load model
with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load the questions
question_text = pd.read_csv('data/question_text.csv')

# group the questions by subcategory
question_by_category = question_text.groupby('SubcategoryUno')

# select a subcategory
selected_subcategory = "E9F87919-0CD5-4D9D-AAC6-3CBC49132F1D"

# get the questions for the selected subcategory
questions = question_by_category.get_group(selected_subcategory)['PostText'].values

# encode the questions using the SentenceTransformer model
question_embeddings = model.encode(questions)

# predict the clusters
cluster_labels = kmeans.predict(question_embeddings)

# find the most representative question for each cluster
closest_questions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, question_embeddings)
representative_questions = [questions[idx] for idx in closest_questions]

# print the clusters and question
for i, question in enumerate(representative_questions):
    print(f"Cluster {i}: {question}")
    print()
