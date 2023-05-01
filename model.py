import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.1,
    ):
        super(Decoder, self).__init__()

        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # PReLU layers, Dropout layers and a tanh layer.
        self.fc = nn.Linear(3, 1)
        self.dropout_prob = dropout_prob
        self.th = nn.Tanh()
        # ***********************************************************************

    # input: N x 3
    def forward(self, input):

        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        x = self.fc(input)
        x = self.th(x)
        # ***********************************************************************


        return x
