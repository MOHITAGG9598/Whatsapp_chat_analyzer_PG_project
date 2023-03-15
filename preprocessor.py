import pandas as pd
import re
from textblob import TextBlob
import numpy as np
import nltk
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia=SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')

def preprocess(data):
    pattern ='\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')
    df.rename(columns={'message_date': 'date'}, inplace=True)
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)

        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])

        else:
            users.append('group_notification')
            messages.append(entry[0])
    df['users'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['Day_name'] = df['date'].dt.day_name()
    df['Date']=df['date'].dt.date
    df['Month'] = df['date'].dt.month
    df['Month_name'] = df['date'].dt.month_name()

    period = []
    for hour in df[['Day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period']=period

    temp = df[df['users'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    temp.replace("", np.nan, inplace=True)
    temp = temp.dropna()

    def cleanTxt(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#', '', text)
        text = text.replace('\n', "")
        return text

    temp['message'] = temp['message'].apply(cleanTxt)
    temp['users'] = temp['users'].apply(cleanTxt)

    res = {}
    for i, row in tqdm(temp.iterrows(), total=len(temp)):
        text = row['message']
        myid = row['users']
        res[myid] = sia.polarity_scores(text)

    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'users'})
    vaders = vaders.merge(temp, how="right")
    vaders_new = vaders.pop('message')
    vaders_new = pd.DataFrame(vaders_new)
    vaders.insert(1, "message", vaders_new['message'])

    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    vaders['Subjectivity'] = vaders['message'].apply(getSubjectivity)
    vaders['Polarity'] = vaders['message'].apply(getPolarity)

    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        if score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    vaders['Analysis'] = vaders['Polarity'].apply(getAnalysis)

    def getAnalysis(score):
        if score <= 0:
            return 'Negative'
        if score < 0.2960:
            return 'Neutral'
        else:
            return 'Positive'

    vaders['vader_Analysis'] = vaders['compound'].apply(getAnalysis)

    return vaders