import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        #print(T,b,h)
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class BidirectionalLSTM2(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM2, self).__init__()
        self.rnn1 = nn.LSTM(nIn, 256, bias=False, bidirectional=False)
        self.rnn2 = nn.LSTM(256, 256, bias=False, bidirectional=False)
        self.embedding = nn.Linear(256, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn1(input)
        # for i in range(recurrent.shape[0]):
        #     print(recurrent[i])
        #print (recurrent)
        recurrent, _ = self.rnn2(recurrent)
        T, b, h = recurrent.size()
        #print(T,b,h)
        t_rec = recurrent.view(T * b, h)
        
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output