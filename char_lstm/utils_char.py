import torch
import torch.nn as nn
import pandas as pd
import os
import string
import torch.utils.data as data

from lyrics_database import LyricsDatabase
from torch.autograd import Variable
# from util import *

# rhyme density
from rhyme_analizer import get_lyrics_stat


def get_rhyme_density(verse):

    file_n = os.path.join(os.getcwd() + '/_temp_file1')
    f = open(file_n, 'w')
    f.write(verse.replace('<', '').replace('>', '').replace('\n', ' '))
    f.close()
    stat = get_lyrics_stat(file_n)

    if 'Rhyme_Density' in stat:
        return stat['Rhyme_Density']
    return -1


all_characters = string.printable  # list of all characters that can be printed
number_of_characters = len(all_characters)
use_cuda = torch.cuda.is_available()


def character_to_label(character):
    return all_characters.find(character)


def string_to_labels(character_string):
    # for loop is to catch characters that do not belong to string.printables (Took 2 hrs to reach this)
    # for char in character_string:
    #     if character_to_label(char) == -1:
    #         print("Got Ya!", char)

    # remove any character if it is not printable
    return [character_to_label(char) for char in character_string if character_to_label(char) != -1]


def pad_sequence(seq, max_length, pad_label=100):
    """TODO: used pytorch's pad sequence instead"""
    seq += [pad_label for i in range(max_length - len(seq))]
    return seq


def transform_word(w):
    if w == '<startVerse>':
        return '<'
    elif w == '<endVerse>':
        return '>'
    elif w == '<endLine>':
        return '\n'
    else:
        return w


class FabolousDataset(data.Dataset):

    def __init__(self, min_num_words=None):

        myData = LyricsDatabase('../fabolous_parsed/')
        self.verseList = myData.get_lyrics_from_artist_as_list_of_verses('fabolous')

        self.verseList = [[transform_word(word.strip()) for word in verse] for verse in self.verseList]
        #self.verseList = [[word for word in verse if word.strip() not in ["<startVerse>", "<endLine>", "<endVerse>"]] for verse in self.verseList]
        if min_num_words is not None:
            self.verseList = [verse for verse in self.verseList if len(verse) >= min_num_words]

        for i in range(len(self.verseList)):
            self.verseList[i] = " ".join(self.verseList[i])

        self.max_text_len = len(max(self.verseList, key=lambda x: len(x)))
        self.whole_dataset_len = len(self.verseList)

    def __len__(self):
        return self.whole_dataset_len

    def __getitem__(self, index):

        sequence_raw_string = self.verseList[index]

        sequence_string_labels = string_to_labels(sequence_raw_string)

        sequence_length = len(sequence_string_labels) - 1

        input_string_labels = sequence_string_labels[:-1]
        output_string_labels = sequence_string_labels[1:]

        input_string_labels_padded = pad_sequence(input_string_labels, max_length=self.max_text_len)

        output_string_labels_padded = pad_sequence(output_string_labels, max_length=self.max_text_len, pad_label=-100)

        return (torch.LongTensor(input_string_labels_padded),
                torch.LongTensor(output_string_labels_padded),
                torch.LongTensor([sequence_length]))


class LyricsGenerationDataset(data.Dataset):

    def __init__(self, csv_file_path, minimum_song_count=None, artists=None):

        self.lyrics_dataframe = pd.read_csv(csv_file_path)

        if artists:

            self.lyrics_dataframe = self.lyrics_dataframe[self.lyrics_dataframe.artist.isin(artists)]
            self.lyrics_dataframe = self.lyrics_dataframe.reset_index()

        if minimum_song_count:

            # Getting artists that have 70+ songs
            self.lyrics_dataframe = self.lyrics_dataframe.groupby('artist').filter(lambda x: len(x) > minimum_song_count)
            # Reindex .loc after we fetched random songs
            self.lyrics_dataframe = self.lyrics_dataframe.reset_index()

        # Get the length of the biggest lyric text
        # We will need that for padding
        self.max_text_len = self.lyrics_dataframe.text.str.len().max()

        whole_dataset_len = len(self.lyrics_dataframe)

        self.indexes = range(whole_dataset_len)

        self.artists_list = list(self.lyrics_dataframe.artist.unique())

        self.number_of_artists = len(self.artists_list)

    def __len__(self):

        return len(self.indexes)

    def __getitem__(self, index):

        index = self.indexes[index]

        sequence_raw_string = self.lyrics_dataframe.loc[index].text

        sequence_string_labels = string_to_labels(sequence_raw_string)

        sequence_length = len(sequence_string_labels) - 1

        # Shifted by one char
        input_string_labels = sequence_string_labels[:-1]
        output_string_labels = sequence_string_labels[1:]

        # pad sequence so that all of them have the same lenght
        # Otherwise the batching won't work
        input_string_labels_padded = pad_sequence(input_string_labels, max_length=self.max_text_len)

        output_string_labels_padded = pad_sequence(output_string_labels, max_length=self.max_text_len, pad_label=-100)

        return (torch.LongTensor(input_string_labels_padded),
                torch.LongTensor(output_string_labels_padded),
                torch.LongTensor([sequence_length]))


class LG_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, n_layers=2):

        super(LG_LSTM, self).__init__()
        self.epochs = 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size)

        self.net = nn.LSTM(hidden_size, hidden_size, n_layers)

        self.fcc_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_sequences, input_sequences_lengths, hidden=None):
        # print(1)
        # print("Input: ", input_sequences)
        embedded = self.encoder(input_sequences)
        # print(2)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_sequences_lengths)
        # print(3)
        outputs, hidden = self.net(packed, hidden)
        # print(4)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # print(5)
        fcc = self.fcc_fc(outputs)
        # print(6)
        fcc = fcc.transpose(0, 1).contiguous()
        # print(7)
        fcc = fcc.view(-1, self.num_classes)
        # print(8)
        return fcc, hidden


def post_process_sequence_batch(batch_tuple):
    input_sequences, output_sequences, lengths = batch_tuple

    splitted_input_sequence_batch = input_sequences.split(split_size=1)
    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    splitted_lengths_batch = lengths.split(split_size=1)

    training_data_tuples = zip(splitted_input_sequence_batch,
                               splitted_output_sequence_batch,
                               splitted_lengths_batch)

    training_data_tuples_sorted = sorted(training_data_tuples,
                                         key=lambda p: int(p[2]),
                                         reverse=True)

    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(*training_data_tuples_sorted)

    input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
    lengths_batch_sorted = torch.cat(splitted_lengths_batch)

    input_sequence_batch_sorted = input_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0]]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0]]

    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)

    lengths_batch_sorted_list = list(lengths_batch_sorted)
    lengths_batch_sorted_list = list(map(lambda x: int(x), lengths_batch_sorted_list))

    return input_sequence_batch_transposed, output_sequence_batch_sorted, lengths_batch_sorted_list


def sample_from_rnn(rnn, starting_sting="<", sample_length=1000, temperature=1):

    sampled_string = starting_sting
    hidden = None

    first_input = torch.LongTensor(string_to_labels(starting_sting))
    if use_cuda:
        first_input = first_input.cuda()
    first_input = first_input.unsqueeze(1)
    current_input = Variable(first_input)

    output, hidden = rnn(current_input, [len(sampled_string)], hidden=hidden)

    output = output[-1, :].unsqueeze(0)

    for i in range(sample_length):

        output_dist = nn.functional.softmax(output.view(-1).div(temperature), dim=0).data

        predicted_label = torch.multinomial(output_dist, 1)

        sampled_string += all_characters[int(predicted_label[0])]

        if all_characters[int(predicted_label[0])] == '>':
            break
        current_input = Variable(predicted_label.unsqueeze(1))

        output, hidden = rnn(current_input, [1], hidden=hidden)

    return sampled_string
