from __future__ import unicode_literals
from bs4 import BeautifulSoup
from flask import Flask, render_template, url_for, request
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
from spacy_summarization import text_summarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from urllib.request import urlopen, Request
from spacy.lang.en import English
from urllib.error import URLError
import flask
import spacy
import spacy_summarization
import time
import urllib

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 '
                         'Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8',
           'Connection': 'keep-alive'}

nlp = spacy.load("en_core_web_sm")
app = Flask(__name__)


# Sumy
def sumy_summary(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result


# Reading Time
def readingtime(mytext):
    total_words = len([token.text for token in nlp(mytext)])
    estimatedTime = total_words / 200.0
    return estimatedTime


# Fetch Text From Url
def get_text(url):
    page = urlopen(url)
    soup = BeautifulSoup(page.content)
    fetched_text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return fetched_text


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    global summary_reading_time_nltk
    start = time.time()
    if flask.request.method == 'POST':
        rawtext = flask.request.form['rawtext']
        final_reading_time = readingtime(rawtext)
        final_summary_spacy = text_summarizer(rawtext)
        summary_reading_time = readingtime(final_summary_spacy)
        # Gensim Summarizer
        final_summary_gensim = summarize(rawtext)
        summary_reading_time_gensim = readingtime(final_summary_gensim)
        # NLTK
        final_summary_nltk = nltk_summarizer(rawtext)
        summary_reading_time_nltk = readingtime(final_summary_nltk)
        # Sumy
        final_summary_sumy = sumy_summary(rawtext)
        summary_reading_time_sumy = readingtime(final_summary_sumy)

        end = time.time()
        final_time = end - start
    return flask.render_template('index.html', ctext=rawtext, final_summary_spacy=final_summary_spacy,
                                 final_summary_gensim=final_summary_gensim, final_summary_nltk=final_summary_nltk,
                                 final_time=final_time, final_reading_time=final_reading_time,
                                 summary_reading_time=summary_reading_time,
                                 summary_reading_time_gensim=summary_reading_time_gensim,
                                 final_summary_sumy=final_summary_sumy,
                                 summary_reading_time_sumy=summary_reading_time_sumy,
                                 summary_reading_time_nltk=summary_reading_time_nltk)


@app.route('/analyze_url', methods=['GET', 'POST'])
def analyze_url():
    start: float = time.time()
    if request.method == 'POST':
        raw_url = request.form['raw_url']
        rawtext: str = get_text(raw_url)
        final_reading_time = readingtime(rawtext)
        final_summary_spacy = text_summarizer(rawtext)
        summary_reading_time = readingtime(final_summary_spacy)
        # Gensim Summarizer
        final_summary_gensim = summarize(rawtext)
        summary_reading_time_gensim = readingtime(final_summary_gensim)
        # NLTK
        final_summary_nltk = nltk_summarizer(rawtext)
        summary_reading_time_nltk = readingtime(final_summary_nltk)
        # Sumy
        final_summary_sumy = sumy_summary(rawtext)
        summary_reading_time_sumy = readingtime(final_summary_sumy)
        final_time = end - start
        end = time.time()
    return flask.render_template('index.html', ctext=rawtext, final_summary_spacy=final_summary_spacy,
                                 final_summary_gensim=final_summary_gensim, final_summary_nltk=final_summary_nltk,
                                 final_time=final_time, final_reading_time=final_reading_time,
                                 summary_reading_time=summary_reading_time,
                                 summary_reading_time_gensim=summary_reading_time_gensim,
                                 final_summary_sumy=final_summary_sumy,
                                 summary_reading_time_sumy=summary_reading_time_sumy,
                                 summary_reading_time_nltk=summary_reading_time_nltk)


@app.route('/about')
def about():
    return flask.render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
