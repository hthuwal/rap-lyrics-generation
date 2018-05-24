import pickle
import os
import numpy as np
from rhyme_analizer import get_lyrics_stat
from lyrics_database import LyricsDatabase
import io
import six
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import dok_matrix
import math
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as pyplot
from scipy.stats import linregress


def clean_out_file(filename):
    print(filename)
    f = open(filename, 'r')
    tokens = []
    f.readline()  # ignore first error file
    for line in f:
        tokens.extend(line.rstrip().split())
    n_tokens = []
    for token in tokens:
        if token == '<startVerse>' or token == '<endVerse>':
            continue
        elif token == '<endLine>':
            n_tokens.append('\n')
        else:
            n_tokens.append(token)
    # print(n_tokens)
    f.close()
    f = open(filename, 'w')
    f.write(" ".join(n_tokens))
    f.close()


def transform_lyric_to_doc(lyric):
    result_text = io.StringIO()

    def is_skip(s): return s == '<startVerse>' or s == '<endVerse>' or s == '<endLine>'

    for token in lyric:
        if is_skip(token):
            continue

        if isinstance(token, six.string_types):
            result_text.write(' ')
            result_text.write(token)
        else:
            for token2 in token:
                if is_skip(token2):
                    continue
                result_text.write(' ')
                result_text.write(token2)

    return result_text.getvalue()


lstm_generated_lyrics_dir = os.getcwd() + '/lstm_output/'
lyrics_dir = '../fabolous_parsed/'
ld = LyricsDatabase(lyrics_dir)

output_files = os.listdir(lstm_generated_lyrics_dir)

# Calculating rhyme densities on data so generated at per 1000 epochs
print("Calculating rhyme densities")
rhyme_densities = {}
for file_n in tqdm(output_files):
    file_n1 = os.path.join(lstm_generated_lyrics_dir, file_n)
    # clean_out_file(file_n1) # call it on first run except on 0
    stat = get_lyrics_stat(file_n1)
    rhyme_densities[int(file_n[8:])] = stat['Rhyme_Density']

# print(rhyme_densities)


# max cosine similarity

lyrics_verses = ld.get_lyrics_from_artist_as_list_of_verses('fabolous')


def create_docs(lyrics_verses, generated_lyrics):
    docs = [generated_lyrics]

    def is_skip(s): return s == '<startVerse>' or s == '<endVerse>' or s == '<endLine>'
    docs.extend([' '.join([t for t in v if not is_skip(t)]) for v in lyrics_verses])

    return docs


def calc_similarity(lyrics_verses, generated_lyrics):
    docs = create_docs(lyrics_verses, generated_lyrics)
    N = len(docs)

    count_vectorizer = CountVectorizer()
    count_data = count_vectorizer.fit_transform(docs).toarray()

    weight_data = dok_matrix(count_data.shape)

    # save nj
    njs = weight_data.shape[1] * [0]
    for j in range(0, weight_data.shape[1]):
        nnz = 0
        for i in range(0, weight_data.shape[0]):
            if count_data[i, j] != 0:
                nnz += 1
        nj = math.log(N / nnz)
        njs[j] = nj

    for i in range(0, weight_data.shape[0]):
        for j in range(0, weight_data.shape[1]):
            w = count_data[i, j] * njs[j]
            if w != 0:
                weight_data[i, j] = w

    similarities = cosine_similarity(weight_data[0, ], weight_data[1:, ])

    return similarities


sim_data_lstm = {}

print("Calculating similarities")
for file_n in tqdm(output_files):
    filename = os.path.join(lstm_generated_lyrics_dir, file_n)
    print("Processing", file_n)
    with open(filename, 'r') as f:
        generated_lyrics = " ".join([i.rstrip() for i in f.readlines()])
        sim_data_lstm[int(file_n[8:])] = calc_similarity(lyrics_verses, generated_lyrics).max()


pyplot.plot(sorted(rhyme_densities.keys()), [rhyme_densities[i] for i in sorted(rhyme_densities.keys())], label='rhyme density')
pyplot.plot(sorted(sim_data_lstm.keys()), [sim_data_lstm[i] for i in sorted(sim_data_lstm.keys())], label='maximum similarity')
pyplot.legend()
pyplot.show()


save_data_f = open('temp_saved.pickle', 'wb')
pickle.dump([rhyme_densities, sim_data_lstm], save_data_f)
save_data_f.close()

# calculating correlation

rhyme_peaks = [rhyme_densities[i] for i in sorted(rhyme_densities.keys())]
sim_peaks = [sim_data_lstm[i] for i in sorted(sim_data_lstm.keys())]

# with erroronous peaks

print('correlation between rhyme density and maximum similarity - ', np.correlate(sim_peaks, rhyme_peaks))

# without erroronous peaks in rhyme density for 1000,2000 epoch model

print('correlation between rhyme density and maximum similarity (without anomalies)- ', np.correlate(sim_peaks[2:], rhyme_peaks[2:]))


# finding maximum similarity at epoch where rhyme density matches that of artist

fabolous_density = 0.34

print('Rhyme density of fabolous ', fabolous_density)

# regression line for rhyme density

slope_rhyme, intercept_rhyme, _, _, _ = linregress(sorted(rhyme_densities.keys())[2:], rhyme_peaks[2:])
slope_sim, intercept_sim, _, _, _ = linregress(sorted(sim_data_lstm.keys())[2:], sim_peaks[2:])

epoch_at_required_rhyme = (fabolous_density - intercept_rhyme) / slope_rhyme

print('Max similarity at', epoch_at_required_rhyme, " : ", slope_sim * epoch_at_required_rhyme + intercept_sim)

# regression line for max similarity
