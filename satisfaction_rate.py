import pandas as pd

questions = pd.read_csv('./data/questions.csv')
# filter out the questions that are not answered by TakenByAttorneyUno since they are not useful

# set the index to QuestionUno
questions.set_index('QuestionUno', inplace=True)

question_posts = pd.read_csv('./data/questionposts.csv')
# keep the first 5 columns
question_posts = question_posts.iloc[:, :5]
question_posts = question_posts[question_posts['PostText'].notna()]

merged_data = pd.merge(questions, question_posts, on='QuestionUno', how='inner')

key_words = ['thank']

# append two columns to the question dataframe
# one column is the number of key word appearing, the second column is the number of words in the question

questions['KeyWordCount'] = 0
questions['WordCount'] = 0

for question_uno, group in merged_data.groupby('QuestionUno'):
    key_word_count = 0
    word_count = 0
    for post_text in group['PostText']:
        for key_word in key_words:
            if key_word in post_text:
                key_word_count += 1
        word_count += len(post_text.split())
    questions.loc[question_uno, 'KeyWordCount'] = key_word_count
    questions.loc[question_uno, 'WordCount'] = word_count

# print rows with key word count > 0
print(questions[questions['KeyWordCount'] > 0])
