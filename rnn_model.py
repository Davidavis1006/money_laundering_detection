import torch
import torch.nn as nn
import numpy as np
import RNN
from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch.cuda
import pandas as pd

class MyRNN(nn.Module):

    def __init__(self):
        super(MyRNN, self).__init__()

        self.output_size = 1
        self.n_layers = 2
        self.hidden_dim = 64
        # self.embedding_dim = 50
        self.input_dim = 1

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers,
                            dropout=0.5, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # linear and sigmoid layers
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.actvation = nn.Sigmoid()



    def forward(self, input, hidden):
        # input = input.to(torch.int64)

        # input = input.unsqueeze(2)
        lstm_out, hidden = self.lstm(input, hidden)

        # out = self.dropout(lstm_out)
        fc_out = self.fc(lstm_out)

        out = self.actvation(fc_out)


        return out, hidden

    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden



rnn = MyRNN()
#data_preprossesing
def read_train_data(filepath):
    features = []
    labels = []
    with open(filepath) as f:
        count=0
        for line in f:
            count+=1
            if(count==1): continue
            line = line.strip('\n')
            line = line.split(',')
            line = np.asarray(line[1:]).astype(float)
            data = line[:-1]
            label = line[-1]
            # print(len(data))
            # print(label, '\n')
            features.append(data)
            labels.append(label)
    return features, labels

def read_test_data(filepath):
    features = []
    with open(filepath) as f:
        count=0
        for line in f:
            count+=1
            if(count==1): continue
            line = line.strip('\n')
            line = line.split(',')
            data = np.asarray(line).astype(float)
            features.append(data)
    return features

#
features, labels = read_train_data('train_data.csv')
test_data = read_test_data('test.csv')
# print(len(test_data[0]))

features = np.asarray(features)
features = torch.FloatTensor(features)
# print(features.shape)

labels = np.asarray(labels)
labels = torch.FloatTensor(labels)
# print(labels.unsqueeze(1).shape)

train_data = TensorDataset(features, labels)
train_loader = DataLoader(train_data, shuffle=False, batch_size=32, drop_last=False)
# print('success')



test_data = np.asarray(test_data)
test_data = torch.FloatTensor(test_data)

test_data = TensorDataset(test_data)
test_loader = DataLoader(test_data, shuffle=False, batch_size=32, drop_last=False)


def train_test(myRNN, train_loader, test_loader, print_every):
    lr = 0.002

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(myRNN.parameters(), lr=lr)

    epochs = 15
    clip = 5  # gradient clipping

    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        myRNN.cuda()

    myRNN.train()
    for e in range(epochs):
        counter = 0
        for inputs, labels in train_loader:
            counter += 1
            h = myRNN.init_hidden(inputs.shape[0])

            # print('input : ', inputs.shape)
            # print('label : ', labels.shape)

            inputs = inputs.unsqueeze(2)
            labels = labels.unsqueeze(1)

            # print('input : ', inputs.shape)
            # print('label : ', labels.shape)

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            h = tuple([each.data for each in h])

            myRNN.zero_grad()

            output, h = myRNN(inputs, h) #output:32 seq_len:39  label:21

            # output_loss = output.view(32, 21, -1)
            # labels_loss = labels.type(torch.LongTensor)
            # output_loss, labels_loss = output_loss.cuda(), labels_loss.cuda()



            loss = criterion(output[:, 0, :], labels)
            loss.backward()

            optimizer.step()

            if counter%print_every == 0:
                print(loss)

    myRNN.eval()
    outputs = []
    for test_feature in test_loader:
        test_feature = torch.cat(test_feature)
        print(test_feature.shape)
        print(test_feature.shape[0])
        h = myRNN.init_hidden(test_feature.shape[0])

        test_feature = test_feature.unsqueeze(2)

        if (train_on_gpu):
            test_feature = test_feature.cuda()

        h = tuple([each.data for each in h])

        output, h = myRNN(test_feature, h)

        outputs += output[:, 0, :]

    return outputs

outputs = train_test(rnn, train_loader, test_loader, 100)

output_arr = [e.item() for e in outputs]


user_id = pd.read_csv('預測的案件名單及提交檔案範例.csv')
user_id = user_id['alert_key'].tolist()

print_out_dataframe = pd.DataFrame({'alert_key': user_id, 'probability': output_arr})
print_out_dataframe.to_csv('submit.csv', index=False)

