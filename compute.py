import pandas as pd

questions = pd.read_csv('./data/questions.csv')

question_posts = pd.read_csv('./tmp/questionposts0.csv')
question_posts = question_posts.iloc[:, :5]
question_posts = question_posts[question_posts['PostText'].notna()]

# convert create_utc to datetime, if cannot then drop the row
question_posts['CreatedUtc'] = pd.to_datetime(question_posts['CreatedUtc'], errors='coerce')
question_posts = question_posts[question_posts['CreatedUtc'].notna()]

# merge the two dataframes on QuestionUno, left join
merged_data = pd.merge(questions, question_posts, on='QuestionUno', how='inner')

# group by QuestionUno and sort by CreatedUtc
first_posts = merged_data.groupby('QuestionUno').first().sort_values(by='CreatedUtc')

# create a new df with the question text and question uno
question_text = first_posts['PostText']

# save the question text to a file
question_text.to_csv('./data/question_text.csv', index=False)