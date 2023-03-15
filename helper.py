import matplotlib.pyplot as plt
from urlextract import URLExtract
from collections import Counter
from wordcloud import WordCloud, STOPWORDS ,ImageColorGenerator
import pandas as pd
import matplotlib.pylab as plt
import emoji

extract=URLExtract()
def fetch_stats(selected_user,df):

    if selected_user!= "Group analysis":
        df=df[df['users']==selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())


    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words),len(links)

def most_busy_users(df):
    x = df['users'].value_counts().head()
    df=round((df['users'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def most_common_words(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]
    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df=pd.DataFrame(Counter(words).most_common(30))
    return most_common_df

def word_cloud(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    stopwords = set('STOPWORDS')

        # wordcloud
    wordcloud = WordCloud(stopwords=stopwords, background_color="Black").generate(''.join(df['message']))
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

    return wordcloud

def emoji_helper(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA.keys()])
    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    timeline = df.groupby(['year', 'Month_name', 'Month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['Month_name'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time

    return timeline
def Daily_timeline(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    daily_timeline = df.groupby('Date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]
    return df['Day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]
    return df['Month_name'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    Activity_heatmap= df.pivot_table(index='Day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return Activity_heatmap

def pos_words(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    pos_word = df[df['vader_Analysis'] == 'Positive']
    pos_word = pos_word.pop('message')
    return pos_word

def neg_words(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    neg_word = df[df['Analysis'] == 'Negative']
    neg_word = neg_word.pop('message')
    return neg_word

def neu_words(selected_user,df):
    if selected_user != "Group analysis":
        df = df[df['users'] == selected_user]

    neu_word = df[df['vader_Analysis'] == 'Neutral']
    neu_word = neu_word.pop('message')
    return neu_word