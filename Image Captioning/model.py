import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
                
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True, 
                            dropout=0,
                            bidirectional=False,
                           )
                            
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        ''' Forward pass through the network '''
        
        # remove <end> token from captions
        captions = captions[:,:-1]
        
        # embed captions
        captions = self.embed(captions)
        
        # concatenate the feature and caption embeds
        embeddings = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # the first value returned by LSTM is all of the hidden states throughout the sequence
        # the second is just the most recent hidden state
        out, _ = self.lstm(embeddings)
                                
        # put out through the fully-connected layer
        out = self.linear(out)

        return out
        
    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        captions = []
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.linear(lstm_out)
            outputs = outputs.squeeze(1)
            wordz  = outputs.argmax(dim=1) 
            captions.append(wordz.item())
            inputs = self.embed(wordz.unsqueeze(0))
        return captions