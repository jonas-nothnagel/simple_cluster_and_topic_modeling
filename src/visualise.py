from wordcloud import WordCloud, STOPWORDS
from wordcloud import WordCloud
import wordcloud

def draw_cloud(dataframe, column):
    # Join the different processed titles together.
    long_string = ','.join(list(dataframe[column]))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=6, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    return wordcloud.to_image()