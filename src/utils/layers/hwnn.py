import torch

class HWNNLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, num_stock, K1=2, K2=2, approx=False, data=None):
        super(HWNNLayer, self).__init__()
        self.data = data
        self.input_size = input_size
        self.output_size = output_size
        self.num_stock = num_stock
        self.K1 = K1
        self.K2 = K2
        self.approx = approx
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_size, self.output_size))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.num_stock))
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, snap_index, data):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter)
        # Theta=self.data.Theta
        Theta = data.hypergraph_snapshot[snap_index]["Theta"]
        Theta_t = torch.transpose(Theta, 0, 1)

        if self.approx:
            poly = self.par[0] * torch.eye(self.num_stock)
            Theta_mul = torch.eye(self.num_stock)
            for ind in range(1, self.K1):
                Theta_mul = Theta_mul @ Theta
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * torch.eye(self.num_stock)
            Theta_mul = torch.eye(self.num_stock)
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = Theta_mul @ Theta_t  # 这里也可以使用Theta_transpose
                poly_t = poly_t + self.par[ind] * Theta_mul

            # poly=self.par[0]*torch.eye(self.num_stock)+self.par[1]*Theta+self.par[2]*Theta@Theta
            # poly_t = self.par[3] * torch.eye(self.num_stock) + self.par[4] * Theta_t + self.par[5] * Theta_t @ Theta_t
            # poly_t = self.par[3] * torch.eye(self.num_stock) + self.par[4] * Theta + self.par[
            #     5] * Theta @ Theta
            local_fea_1 = poly @ diagonal_weight_filter @ poly_t @ features @ self.weight_matrix
        else:
            wavelets = self.data.hypergraph_snapshot[snap_index]["wavelets"]
            wavelets_inverse = self.data.hypergraph_snapshot[snap_index]["wavelets_inv"]
            local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix

        localized_features = local_fea_1
        return localized_features