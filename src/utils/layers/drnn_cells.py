import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type
        self.weights = None
        self.biases = None

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias, num_nodes):
        super(DLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_nodes = num_nodes
        self.rnn_units = hidden_size // num_nodes
        self._num_nodes = num_nodes
        self._memory_dim = 16
        self._bottleneck_dim = 4


        self._rnn_params = LayerParams(self, 'rnn_params')
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, inputs, hx=None):

        if hx is None:
            hx = Variable(inputs.new_zeros(inputs.size(0), self.hidden_size))
            hx = (hx, hx)

        hx, cx = hx

        gates = self._fc_dynamic(inputs, hx, 4*self.rnn_units, bias_start=1.0, param_layer=self._rnn_params)
        gates = torch.reshape(gates, (-1, self._num_nodes, 4*self.rnn_units))
        # Get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=2)
        input_gate = torch.reshape(input_gate, (-1, self._num_nodes * self.rnn_units))
        forget_gate = torch.reshape(forget_gate, (-1, self._num_nodes * self.rnn_units))
        cell_gate = torch.reshape(cell_gate, (-1, self._num_nodes * self.rnn_units))
        output_gate = torch.reshape(output_gate, (-1, self._num_nodes * self.rnn_units))
        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)


        return (hy, cy)

    def _fc(self, inputs, state, output_size, bias_start=0.0, param_layer=None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)

        input_size = inputs_and_state.shape[-1]
        weights = param_layer.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = param_layer.get_biases(output_size, bias_start)
        value = value + biases
        return value

    def _fc_dynamic(self, inputs, state, output_size, bias_start=0.0, param_layer=None, supports=None, full_input=None):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]

        memory = param_layer.get_weights((self._num_nodes, self._memory_dim))

        w1 = param_layer.get_weights((memory.shape[1], memory.shape[1]))
        b1 = param_layer.get_biases(memory.shape[1], bias_start)

        w2 = param_layer.get_weights((memory.shape[1], self._bottleneck_dim))
        b2 = param_layer.get_biases(self._bottleneck_dim, bias_start)

        w3 = param_layer.get_weights((self._bottleneck_dim, input_size * output_size))
        b3 = param_layer.get_biases(input_size * output_size, bias_start)

        mem = torch.tanh(torch.matmul(memory, w1) + b1)
        mem = torch.tanh(torch.matmul(mem, w2) + b2)

        weights = (torch.matmul(mem, w3) + b3).reshape([self._num_nodes, input_size, output_size])
        
        weights = weights.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        weights = weights.reshape([batch_size * self._num_nodes, input_size, output_size])

        b_out = param_layer.get_biases(output_size, bias_start)
        value = torch.sigmoid(torch.matmul(inputs_and_state.unsqueeze(1), weights).squeeze())
        value = value + b_out
        return value