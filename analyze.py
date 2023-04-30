import pickle
from datetime import datetime

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

# load model
with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load the questions
question_text = pd.read_csv('data/question_text.csv')

# select a subcategory
selected_subcategory = "E9F87919-0CD5-4D9D-AAC6-3CBC49132F1D"

# get the questions for the selected subcategory
question_text = question_text[question_text['SubcategoryUno'] == selected_subcategory]

questions = question_text['PostText'].tolist()

# encode the questions using the SentenceTransformer model
question_embeddings = model.encode(questions)

# predict the clusters
cluster_labels = kmeans.predict(question_embeddings)

# append the cluster labels to the question text dataframe
question_text['Cluster'] = cluster_labels

# find the most representative question for each cluster
closest_questions, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, question_embeddings)
representative_questions = [questions[idx] for idx in closest_questions]

# print the clusters and question
for i, question in enumerate(representative_questions):
    print(f"Cluster {i}: {question}")
    print()

# assign the cluster labels to the questions


satisfaction_rate = pd.read_csv('data/satisfaction_rate.csv')
satisfaction_rate = satisfaction_rate[satisfaction_rate['SubcategoryUno'] == selected_subcategory]

# we want to make a plot where x is date
# y is satisfaction rate for each question in one cluster
# we want to plot a line for each cluster
merged_data = pd.merge(question_text, satisfaction_rate, on='QuestionUno', how='inner')
merged_data['AskedOnUtc'] = pd.to_datetime(merged_data['AskedOnUtc'])

# for each cluster make the plot
for i, group in merged_data.groupby('Cluster'):
    X = group['AskedOnUtc']
    y = group['SatisfactionRate']
    plt.scatter(X, y, label=f'Cluster {i}')

    plt.legend()
    plt.show()

print(merged_data)
