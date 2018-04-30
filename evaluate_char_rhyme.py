import torch
import os
# from util import *

# rhyme density
from utils_char import FabolousDataset, LG_LSTM, all_characters, use_cuda, sample_from_rnn, get_rhyme_density

print("Loading Dataset")

# trainset1 = LyricsGenerationDataset(csv_file_path='songdata.csv')  # dataset of 55000 songs mixed artist
trainset2 = FabolousDataset(min_num_words=175)  # fabulous dataset


def evaluate_model(model_file):
    rnn = LG_LSTM(input_size=len(all_characters) + 1, hidden_size=128, num_classes=len(all_characters))
    if use_cuda:
        rnn = rnn.cuda()
    if os.path.exists(model_file):
        print("Loading Model")
        rnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    else:
        print('Model doesnt exist')
        return
    avg = []
    lens = []
    sents = []
    for i in range(10):
        sent = sample_from_rnn(rnn)
        lens.append(len(sent.split(' ')))
        _temp_density = get_rhyme_density(sent)
        if _temp_density != -1:
            avg.append(_temp_density)
            sents.append(sent)
    if len(avg) != 0:
        rhyme_density = sum(avg) / len(avg)
        avg_len = sum(lens) / len(lens)
        print('rhyme density for model %s : %f : average length %d' % (model_file, rhyme_density, avg_len))
        f = open('evaluated_transfer2/' + model_file.split('/')[1] + '.txt', 'w')
        f.write("%f\n%s" % (rhyme_density, "\n\n".join(sents)))
        f.close()
    else:
        print('rhyme density for model %s not well formed' % model_file)


models = os.listdir('best_models_retrain/')

for model_file in models:
    print('best_models_retrain/' + model_file)
    evaluate_model('best_models_retrain/' + model_file)
