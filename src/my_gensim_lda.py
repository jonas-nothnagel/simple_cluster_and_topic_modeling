from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import CoherenceModel,LsiModel
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import utils
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim  

def build_dict(data, no_below = 0, no_above = 1.0, tf_idf_normalize = True):
    
    '''
    takes in pandas column that contains tokens and has to be named "tokens"
    
    tf_idf_normalize must be bool.
    
    
    '''
    
    dictionary = gensim.corpora.Dictionary(data['tokens'])
    dictionary.filter_extremes(no_below=no_below, no_above = no_above)
    print(len(dictionary))
    bow_corpus = [dictionary.doc2bow(doc) for doc in data['tokens']]
    tfidf = models.TfidfModel(bow_corpus, normalize=True)
    tfidf_corpus = tfidf[bow_corpus]
    
    return dictionary, bow_corpus, tfidf_corpus

def create_LDA_model(data, corpus, number_of_topics, dictionary, alpha=0.1, beta=0.01,
                     random_state=123, passes=20):
    
    '''
    Train the LDA model
    
    '''
    
    lda_model =  gensim.models.LdaModel(corpus, num_topics=number_of_topics, id2word = 
                    dictionary) 

    coherence_lda = CoherenceModel(model=lda_model, texts=data['tokens'],
                              dictionary=dictionary, coherence='c_v')

    return lda_model,coherence_lda.get_coherence()



#create graph
def plot_graph(coherence_values, start, stop, step):
    
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    
    return plt.show()


#plot topics as wordclouds
def topic_wordclouds(lda_model, ):
    
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    #stopwords=stop_words,
    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=1800,
                      max_words=5000,
                      contour_color='steelblue',
                     # colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],prefer_horizontal=1.0)
    #topics = gensimmodel30.get_document_topics(corpus)
    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 3, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud.to_image())
        plt.gca().set_title('Topic ' + str(i),fontdict=dict(size=30))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    
    return plt.show()
