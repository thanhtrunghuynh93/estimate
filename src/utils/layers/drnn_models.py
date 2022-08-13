import torch
import torch.nn as nn
from torch.autograd import Variable

from utils.layers.drnn_cells import DLSTMCell

class DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_nodes, num_layers=1, bias=True, output_size=1):
        super(DLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.num_nodes = num_nodes
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(DLSTMCell(self.input_size,
                                            self.hidden_size,
                                            self.bias,
                                            self.num_nodes))
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(DLSTMCell(self.hidden_size,
                                                self.hidden_size,
                                                self.bias,
                                                self.num_nodes))

    def forward(self, input, hx=None):

        if hx is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
             h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        (hidden[layer][0],hidden[layer][1])
                        )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        (hidden[layer][0], hidden[layer][1])
                        )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        if torch.cuda.is_available():
            out = torch.stack(outs, axis=1).cuda()
        else:
            out = torch.stack(outs, axis=1)

        return out, hidden