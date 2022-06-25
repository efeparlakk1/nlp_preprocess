import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud

def clean_text(df, col):
    df[col] = df[col].str.lower()
    df[col] = df[col].str.replace('[^\w\s]', '')
    df[col] = df[col].str.replace('\d', '')
    df[col] = df[col].str.replace(r'\n', " ")
    return df
def remove_stopwords(df, col):
    sw = stopwords.words('english')
    df[col] = df[col].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    return df
def preprocess_texts(df, col, show=False):
    """

    :param df: dataframe that you will use
    :param col: columns that you will use
    :param bar: to plot barplot
    :param wordcloud: to show image of words on wordcloud
    :return: returns preprocessed df

    """
    print("Process begins...")
    df = clean_text(df, col)
    df = remove_stopwords(df, col)
    freq_count = pd.Series(" ".join(df["text"]).split()).value_counts()
    drops = freq_count[freq_count < 1500]
    df["text"] = df["text"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    tokenized = df["text"].apply(lambda x: TextBlob(x).words)
    df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    freq_df = pd.Series(" ".join(df["text"]).split()).value_counts().reset_index()
    print("Process ongoing...")
    freq_df.columns = ["words", "tf"]
    tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]
    if show == "bar":
        tf[tf["tf"] > 7500].plot.bar(x="words", y="tf")
        plt.show()
    if show == "wordcloud":
        text = " ".join(i for i in df.text)
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    print("Process complete.")
    print(df.head())
    return df

df = pd.read_csv("datasets/text_data.csv", index_col="Unnamed: 0")
df = preprocess_texts(df, "text", show="wordcloud")









