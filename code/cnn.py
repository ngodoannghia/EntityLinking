import json
import torch
import torch.nn as nn


class CharacterLevelCNN(nn.Module):
    def __init__(self, vocab_chars, vocab_words, input_dim, max_length_seq_char, max_length_word, output_dim, dropout_rate=0.2):
        super(CharacterLevelCNN, self).__init__()

        # define conv layers
        
        self.vocab_chars = vocab_chars
        self.vocab_words = vocab_words
        self.dropout_input = nn.Dropout2d(dropout_rate)
        self.embedding_char = nn.Embedding(vocab_chars, input_dim, padding_idx=0)
        self.embedding_word = nn.Embedding(vocab_words, input_dim, padding_idx=0)

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                input_dim,
                256,
                kernel_size=7,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, padding=1), nn.ReLU(), nn.MaxPool1d(3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(3, stride=1),
        )

        # compute the  output shape after forwarding an input to the conv layers

        input_shape = (
            128,
            max_length_seq_char * max_length_word + max_length_word,
            # max_length_seq_char * max_length_word,
            input_dim,
        )
        self.output_dimension = self._get_conv_output(input_shape)

        # define linear layers

        self.fc1 = nn.Sequential(
            nn.Linear(self.output_dimension, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5))

        self.fc3 = nn.Linear(1024, output_dim)

        # initialize weights

        self._create_weights()

    # utility private functions

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        x = torch.rand(shape)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        output_dimension = x.size(1)
        return output_dimension

    # forward

    def forward(self, x1, x2):
        x1 = self.embedding_char(x1).transpose(1, 2)
        x2 = self.embedding_word(x2).transpose(1, 2)
        x = torch.cat((x2, x1), dim=-1).transpose(1, 2)
        # x = self.embedding_char(x1)

        # print(x.shape)
        # x = self.embedding(x)
        x = self.dropout_input(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)
        x = self.fc3(x)

        # print(x.shape)
        return x