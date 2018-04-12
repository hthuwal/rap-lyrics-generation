import numpy as np
from common_lib.lyrics_database import LyricsDatabase

lyrics_dir = 'top_9_parsed/'
ld = LyricsDatabase(lyrics_dir)

artists = ld.get_artists_names()
print(len(artists))

stat = {}
for a in artists:
    lyrics = ld.get_lyrics_from_artist(a)
    num_of_lyrics = len(lyrics)

    verses_len = []
    for l in lyrics:
        nov = 0
        count_of_tokens = 0
        for token in l:
            if token == '<startVerse>':
                count_of_tokens = 0
            elif token == '<endVerse>':
                if count_of_tokens >= 20:
                    nov += 1
            else:
                if isinstance(token, list):
                    count_of_tokens += len(token)

        verses_len.append(nov)

    np_arr = np.array(verses_len)
    all_nov = np_arr.sum()
    min_nov = np_arr.min()
    max_nov = np_arr.max()
    mean_nov = np_arr.mean()
    std_nov = np_arr.std()

    all_tokens = ld.get_lyrics_from_artist_as_plain_list(a)
    uniq_vocab = len(set(all_tokens))

    print(a, num_of_lyrics, all_nov, uniq_vocab, max_nov, round(mean_nov, 1), round(std_nov, 1))

l = lpd.get_lyrics_from_artist_as_plain_list(artists[0])
print(len(l), len(set(l)))
