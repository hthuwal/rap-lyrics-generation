import torch
import os
import sys

from torch.autograd import Variable
from tqdm import tqdm

from utils_char import LyricsGenerationDataset, FabolousDataset, LG_LSTM, all_characters, use_cuda, post_process_sequence_batch, sample_from_rnn
print("Loading Dataset")

trainset1 = LyricsGenerationDataset(csv_file_path='songdata.csv')  # dataset of 55000 songs mixed artist
trainset2 = FabolousDataset(min_num_words=175)  # fabulous dataset


def train(model_file, trainset, out_folder, batch_size=1, epochs=100, bias=0):
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    rnn = LG_LSTM(input_size=len(all_characters) + 1, hidden_size=128, num_classes=len(all_characters))
    if os.path.exists(model_file):
        print("Loading Model")
        rnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    print("use_cuda ", use_cuda)
    if use_cuda:
        rnn.cuda()

    print("Training Model")

    learning_rate = 0.001
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch_number in range(epochs):

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
            loss.backward()

            optimizer.step()

        print("Epoch %d/%d: Total epochs:%d Loss:%f" % (epoch_number, epochs, epoch_number + bias, loss))
        print("Saving Model %s" % (model_file))
        torch.save(rnn.state_dict(), model_file)

        if rnn.epochs % 100 == 0:
            sent = sample_from_rnn(rnn)
            print("Generated\n", sent)
            out_f = open('%s/%d.txt' % (out_folder, epoch_number + bias), 'w')
            out_f.write("%f\n" % (loss) + sent)
            out_f.close()

        rnn.epochs += 1


if sys.argv[1] == "fab":
    train("lg_char_fabolous_orig.model", trainset2, "fabolous_out_orig", batch_size=1, epochs=5000, bias=0)
else:
    train("lg_char_kaggle.model", trainset1, "lg", batch_size=100, epochs=100)
