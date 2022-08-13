import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import HypergraphConv
from utils.layers.temporal_attention import get_subsequent_mask, TemporalAttention
from utils.layers.drnn_models import DLSTM
from utils.layers.hwnn import HWNNLayer
 
class Model(nn.Module):
    def __init__(self, snapshots, num_stock, history_window, num_feature, embedding_dim = 16, rnn_hidden_unit = 8, mlp_hidden = 16, n_head = 4, d_k = 8, d_v = 8, drop_prob = 0.2):
        super(Model, self).__init__()
        
        self.num_stock = num_stock
        self.seq_len = history_window
        self.num_feature = num_feature
        self.rnn_hidden_unit = rnn_hidden_unit 
        self.drop_prob = drop_prob
        self.embedding_dim = embedding_dim        
        self.mlp_hidden = mlp_hidden

        self.snapshots = snapshots
        self.hyper_snapshot_num = len(self.snapshots.hypergraph_snapshot)
        self.par = torch.nn.Parameter(torch.Tensor(self.hyper_snapshot_num))
        torch.nn.init.uniform_(self.par, 0, 0.99)
       
        self.embedding = nn.Linear(num_feature, embedding_dim)        
        self.lstm1 = DLSTM(embedding_dim * num_stock, self.rnn_hidden_unit * num_stock * 2, num_stock)
        self.ln_1 = nn.LayerNorm(self.rnn_hidden_unit * num_stock * 2)
        self.lstm2 = DLSTM(self.rnn_hidden_unit * num_stock * 2, self.rnn_hidden_unit * num_stock, num_stock)
        self.temp_attn = TemporalAttention(n_head, self.rnn_hidden_unit * num_stock, d_k, d_v, dropout=drop_prob)        
        self.dropout = nn.Dropout(self.drop_prob)
        
        self.hatt1 = HypergraphConv(self.rnn_hidden_unit * self.seq_len, self.rnn_hidden_unit * self.seq_len, use_attention=False, heads=4, concat=False, negative_slope=0.1, dropout=drop_prob, bias=True)
        self.hatt2 = HypergraphConv(self.rnn_hidden_unit * self.seq_len, self.rnn_hidden_unit * self.seq_len, use_attention=False, heads=1, concat=False, negative_slope=0.1, dropout=drop_prob, bias=True)
        self.convolution_1 = HWNNLayer(self.rnn_hidden_unit * self.seq_len,
                                       self.rnn_hidden_unit * self.seq_len,
                                       self.num_stock,
                                       K1=3,
                                       K2=3,
                                       approx=False,
                                       data=self.snapshots)

        self.convolution_2 = HWNNLayer(self.rnn_hidden_unit * self.seq_len,
                                       self.rnn_hidden_unit * self.seq_len,
                                       self.num_stock,
                                       K1=3,
                                       K2=3,
                                       approx=False,
                                       data=self.snapshots)

        self.mlp_1 = nn.Linear(rnn_hidden_unit * 2, mlp_hidden)
        self.act_1 = nn.ReLU()        
        self.mlp_2 = nn.Linear(mlp_hidden, 1)

    def forward(self, inputs):        
        
        inputs = self.embedding(inputs)
        
        inputs = inputs.permute(0, 2, 1, 3) # batch_size, seq_len, num_stock, num_feature
        
        inputs = torch.reshape(inputs, (-1 , self.seq_len, self.embedding_dim * self.num_stock))
        
        slf_attn_mask = get_subsequent_mask(inputs).bool()                
        
        output, _ = self.lstm1(inputs)
        output = self.ln_1(output)                
        enc_output, _ = self.lstm2(output)
        enc_output, enc_slf_attn = self.temp_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask.bool())
        
        enc_output = torch.reshape(enc_output, (-1, self.seq_len, self.num_stock, self.rnn_hidden_unit))
        enc_output = self.dropout(enc_output)  
        enc_output = enc_output.permute(0, 2, 1, 3)
        
        outputs = []
        for i in range(enc_output.shape[0]):
            x = enc_output[i].reshape(self.num_stock, self.seq_len * self.rnn_hidden_unit)
            channel_feature = []
            for snap_index in range(self.hyper_snapshot_num):
                deep_features_1 = F.leaky_relu(self.convolution_1(x,
                                                            snap_index,
                                                            self.snapshots), 0.1)
                deep_features_1 = self.dropout(deep_features_1)
                deep_features_2 = self.convolution_2(deep_features_1,
                                                    snap_index,
                                                    self.snapshots)
                deep_features_2 = F.leaky_relu(deep_features_2, 0.1)
                channel_feature.append(deep_features_2)
            deep_features_3 = torch.zeros_like(channel_feature[0])
            for ind in range(self.hyper_snapshot_num):
                deep_features_3 = deep_features_3 + self.par[ind] * channel_feature[ind]
            outputs.append(deep_features_3)
            
        hyper_output = torch.stack(outputs).reshape(-1, self.num_stock, self.seq_len, self.rnn_hidden_unit)
        
        enc_output = torch.cat((enc_output, hyper_output), dim = 3)
        
        output = self.mlp_1(enc_output)        
        output = F.relu(output)
        output = self.mlp_2(output)              
        output = torch.reshape(output, (-1, self.num_stock, self.seq_len))
            
        return output