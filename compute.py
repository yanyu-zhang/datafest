import pandas as pd

questions = pd.read_csv('questions.csv')
# filter out the questions that are not answered by TakenByAttorneyUno since they are not useful
questions = questions[questions['TakenByAttorneyUno'] is not None]

question_posts = pd.read_csv('question_posts_short.csv')
# keep the first 5 columns
question_posts = question_posts.iloc[:, :5]

# merge the two dataframes on QuestionUno, left join
merged_data = pd.merge(questions, question_posts, on='QuestionUno', how='inner')

print("merged_data: ", merged_data.head())

# group by QuestionUno and sort by CreatedUtc
first_posts = merged_data.groupby('QuestionUno').first().sort_values(by='CreatedUtc')

question_text = first_posts['PostText']

for question in question_text:
    print(question)