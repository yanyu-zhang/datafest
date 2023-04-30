import pandas as pd

questions = pd.read_csv('./data/questions.csv')

question_posts = pd.read_csv('./data/questionposts.csv')
question_posts = question_posts.iloc[:, :5]
question_posts = question_posts[question_posts['PostText'].notna()]

# convert create_utc to datetime, if cannot then drop the row
question_posts['CreatedUtc'] = pd.to_datetime(question_posts['CreatedUtc'], errors='coerce')
question_posts = question_posts[question_posts['CreatedUtc'].notna()]

# merge the two dataframes on QuestionUno, left join
merged_data = pd.merge(questions, question_posts, on='QuestionUno', how='left')
# drop the rows where the question text is null
merged_data = merged_data[merged_data['PostText'].notna()]

# group by QuestionUno and sort by CreatedUtc
first_posts = merged_data.groupby('QuestionUno').first().sort_values(by='CreatedUtc')

# create a new df with the question text and question uno
question_text = first_posts['PostText']
# append SubcategoryUno column to the dataframe
question_text = pd.merge(question_text, questions[['QuestionUno', 'SubcategoryUno']], on='QuestionUno', how='left')
# drop the duplicates
question_text = question_text.drop_duplicates(subset=['QuestionUno'])
# set the index to QuestionUno
question_text = question_text.set_index('QuestionUno')

# save the question text to a file
question_text.to_csv('./data/question_text.csv', index=True)

print(question_text)