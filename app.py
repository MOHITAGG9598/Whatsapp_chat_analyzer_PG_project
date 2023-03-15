import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import nltk
nltk.download('all')
import matplotlib.pyplot as plt
import helper
import preprocessor
import torch
import seaborn as sns
st.sidebar.title("Whatsapp Chat analyzer")

uploaded_file= st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:

    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    df_new= preprocessor.preprocess(data)

    user_list= df_new['users'].unique().tolist()
    user_list.sort()
    user_list.insert(0,"Group analysis")
    selected_user=st.sidebar.selectbox("show analysis wrt",user_list)
    if st.sidebar.button("Show Analysis"):
        num_messages,words,num_links=helper.fetch_stats(selected_user,df_new)
        st.title("Top Statistics")
        col1,col2,col3=st.columns(3)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Links Shared")
            st.title(num_links)

        st.title("Timeline")
        col1, col2 = st.columns(2)

        with col1:
            st.header("Monthly ")
            timeline = helper.monthly_timeline(selected_user, df_new)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.title("Daily")
            daily_timeline = helper.Daily_timeline(selected_user, df_new)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['Date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Activity Map")
        col1,col2=st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day=helper.week_activity_map(selected_user, df_new)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color=('pink','green','orange','black','blue','yellow','red'))
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.header("Most busy Month")
            busy_day = helper.month_activity_map(selected_user, df_new)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values,color=('violet','indigo','blue','green'))
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        Activity_heatmap=helper.activity_heatmap(selected_user,df_new)
        fig,ax=plt.subplots()
        ax=sns.heatmap(Activity_heatmap,cmap='RdBu',linewidths=1,linecolor='black')
        st.pyplot(fig)

        if selected_user == "Group analysis":
            st.title("Most busy user")
            x,new_df=helper.most_busy_users(df_new)
            fig,ax=plt.subplots()
            col1,col2=st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color=('blue','red','pink','orange','green'))
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        st.title("Chat Sentiment Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("Positive")
            pos_words = helper.pos_words(selected_user, df_new)
            st.dataframe(pos_words)
        with col2:
            st.header("Negative")
            neg_words = helper.neg_words(selected_user, df_new)
            st.dataframe(neg_words)
        with col3:
            st.header("Neutral")
            neu_words = helper.neu_words(selected_user, df_new)
            st.dataframe(neu_words)


        st.title("Word cloud")
        df_wc = helper.word_cloud(selected_user, df_new)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        plt.axis('off')
        st.pyplot(fig)

        st.title("Most Common Words")
        most_common_df=helper.most_common_words(selected_user,df_new)
        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1],color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue')
        plt.ylabel(None)
        sns.despine(left=True)
        ax.grid(False)
        ax.tick_params(bottom=True, left=False)
        st.pyplot(fig)
        st.dataframe(most_common_df.style.set_properties(**{"background-color": "black", "color": "lawngreen"}))

        emoji_df=helper.emoji_helper(selected_user,df_new)
        st.title("Emoji Analysis")
        st.dataframe(emoji_df.style.set_properties(**{"background-color": "black", "color": "lawngreen"}))


st.title("Sentiment Analysis")
@st.cache(allow_output_mutation=True)
def get_model():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer,model


tokenizer, model = get_model()

user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

sent_pipeline = pipeline("sentiment-analysis")
if user_input and button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Prediction: ", sent_pipeline(user_input))
    showWarningOnDirectExecution = False
