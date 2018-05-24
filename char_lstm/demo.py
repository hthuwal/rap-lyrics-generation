import torch
import os

from utils_char import LG_LSTM, all_characters, use_cuda, sample_from_rnn


def demo_loop(model_file):
    rnn = LG_LSTM(input_size=len(all_characters) + 1, hidden_size=128, num_classes=len(all_characters))
    if use_cuda:
        rnn = rnn.cuda()
    if os.path.exists(model_file):
        print("Loading Model")
        rnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    else:
        print('Model doesnt exist')
        return
    while True:
        try:
            inp = input('Enter starting phrase:\n')
            temp = float(input("Enter temperature: "))
            generated_sent = sample_from_rnn(rnn, starting_sting='<' + inp, temperature=temp)
            print('Generated verse : \n')
            print(generated_sent.replace('<', '').replace('>', ''))
        except Exception as e:
            print(e)
            break


model_name = 'models/best_models_retrain/43_lg_char_kaggle_retrain.model'
demo_loop(model_name)
