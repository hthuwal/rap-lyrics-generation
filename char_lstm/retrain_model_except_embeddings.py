import torch
import os

from torch.autograd import Variable
from tqdm import tqdm
from utils_char import *

print("Loading Dataset")

trainset1 = LyricsGenerationDataset(csv_file_path='songdata.csv')  # dataset of 55000 songs mixed artist
trainset2 = FabolousDataset(min_num_words=175)  # fabulous dataset


def train(model_file, trainset, out_folder, batch_size=1, epochs=100, bias=0):

    rhyme_target = 0.34
    prev_rhym = 0
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    rnn = LG_LSTM(input_size=len(all_characters) + 1, hidden_size=128, num_classes=len(all_characters))
    model_path = os.path.join("models", model_file)
    out_folder = os.path.join("outputs", out_folder)
    if os.path.exists(model_path):
        print("Loading Model")
        rnn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    print("use_cuda ", use_cuda)
    if use_cuda:
        rnn.cuda()

    print("Training Model")

    learning_rate = 0.001

    # setting the gradient = False for embeddings
    parameters = []
    for param in rnn.parameters():
        parameters.append(param)

    parameters[0].requires_grad = False

    # Optimizing everything except the embeddings
    optimizer = torch.optim.Adam(parameters[1:], lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch_number in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(trainset_loader):

            post_processed_batch_tuple = post_process_sequence_batch(batch)
            input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple
            # print(input_sequences_batch.shape)
            output_sequences_batch_var = Variable(output_sequences_batch.contiguous().view(-1))
            input_sequences_batch_var = Variable(input_sequences_batch)

            if use_cuda:
                output_sequences_batch_var = output_sequences_batch_var.cuda()
                input_sequences_batch_var = input_sequences_batch_var.cuda()

            optimizer.zero_grad()

            logits, _ = rnn(input_sequences_batch_var, sequences_lengths)

            loss = criterion(logits, output_sequences_batch_var)
            epoch_loss += loss.data[0]
            loss.backward()

            optimizer.step()

        print("Epoch %d/%d: Total epochs:%d Loss:%f" % (epoch_number, epochs, epoch_number + bias, epoch_loss))
        save_model(model_path, rnn, "Saving Model %s" % (model_path))

        avg = []
        for i in range(10):
            sent = sample_from_rnn(rnn)
            _temp_density = get_rhyme_density(sent)
            if _temp_density != -1:
                avg.append(_temp_density)

        if len(avg) != 0:
            rhyme_density = sum(avg) / len(avg)
            print('rhyme density', rhyme_density)
            if abs(rhyme_density - rhyme_target) < 0.03:  # < abs(prev_rhym - rhyme_target):
                best_model_path = os.path.join("models", "best_models_retrain", str(epoch_number) + '_' + model_file)
                save_model(best_model_path, rnn, msg="Saving best Model")
                prev_rhym = rhyme_density

        if rnn.epochs % 10 == 0:
            sent = sample_from_rnn(rnn)
            print("Generated\n", sent)
            output_file = os.path.join(out_folder, "%d.txt" % (epoch_number + bias))
            save_text(output_file, "%f\n" % (epoch_loss) + sent)

        rnn.epochs += 1


# if sys.argv[1] == "fab":
#     train("lg_char_fabolous_new2.model", trainset2, "fabolous_out_new2", batch_size=1, epochs=1000, bias=0)
# else:

train("lg_char_kaggle_retrain.model", trainset2, "retrain_out", batch_size=1, epochs=1000, bias=0)
