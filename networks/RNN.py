import torch
import torch.nn.functional as F
import torch.nn as nn

class RNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        batch_size,
        embedding_dimension=300,
        hidden_size=128, 
        n_layers=1,
        device='cpu',
    ):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size
        
        self.encoder = nn.Embedding(vocab_size, embedding_dimension)
        self.rnn = nn.GRU(
            embedding_dimension, # number of features in input
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
<<<<<<< Updated upstream
        # self.decoder = nn.Linear(hidden_size, 2)
=======
        
>>>>>>> Stashed changes
        self.decoder = nn.Linear(hidden_size, 1)
        
    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size).to(self.device)
<<<<<<< Updated upstream
        # return torch.randn(self.batch_size, self.hidden_size).to(self.device)
=======
        
>>>>>>> Stashed changes
    
    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
<<<<<<< Updated upstream
        # print(self.batch_size)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        
        # print(inputs)
        # exit()

        # UNCOMMENT TO USE WITHOUT ENCODING, DON'T KNOW IS THIS MAKES SENSE TO DO THOUGH
        # Add extra dimension, batched input has to be of dimensions Batch x Seq Length x Number of Features
        # inputs = inputs[:, None, :] 
        # It says it wants Long not float but it actually wants float not Long (why tf is this a thing)
        # https://discuss.pytorch.org/t/runtimeerror-expected-scalar-type-long-but-found-float/103299
        # inputs = inputs.type(torch.float)
        
=======
        
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        
>>>>>>> Stashed changes
        encoded = self.encoder(inputs)
        output, hidden = self.rnn(encoded, self.init_hidden())
        
        output = self.decoder(output[:, -1, :]).squeeze()
        output = torch.sigmoid(output)
<<<<<<< Updated upstream
        # print()
        # print(output)
        # quit()

        # for i in range(output.size()[0]):
        #     output[i,0] = torch.argmax(output[i,:])
        # output = output[:,-0]

        return output
=======

        return output
>>>>>>> Stashed changes
