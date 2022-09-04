
from urllib import request
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import sent_tokenize
import re
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
link = "https://www.youtube.com/watch?v=YFhp587hcI0" 
unique_id = link.split("=")[-1]
sub = YouTubeTranscriptApi.get_transcript(unique_id)  
subtitle = " ".join([x['text'] for x in sub])
subtitle = subtitle.replace("n","")
sentences = sent_tokenize(subtitle)
organized_sent = {k:v for v,k in enumerate(sentences)}
tf_idf = TfidfVectorizer(min_df=1, 
                                    strip_accents='unicode',
                                    max_features=None,
                                    lowercase = True,
                                    token_pattern=r'w{1,}',
                                    ngram_range=(1, 3), 
                                    use_idf=1,
                                    smooth_idf=1,
                                    sublinear_tf=1,
                                    stop_words = 'english')
import numpy as np
sentence_vectors = tf_idf.fit_transform(sentences)
sent_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
N = 3
top_n_sentences = [sentences[index] for index in np.argsort(sent_scores, axis=0)[::-1][:N]]
# mapping the scored sentences with their indexes as in the subtitle
mapped_sentences = [(sentence,organized_sent[sentence]) for sentence in top_n_sentences]
# Ordering the top-n sentences in their original order
mapped_sentences = sorted(mapped_sentences, key = lambda x: x[1])
ordered_sentences = [element[0] for element in mapped_sentences]
# joining the ordered sentence
summary = " ".join(ordered_sentences)
print(summary)








