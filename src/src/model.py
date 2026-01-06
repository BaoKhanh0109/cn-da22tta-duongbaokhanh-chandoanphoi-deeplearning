import torch
import torch.nn as nn
import timm
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Công thức GCN: Output = A * (X * W)
        support = torch.mm(input, self.weight) 
        output = torch.mm(adj, support)        
        if self.bias is not None:
            return output + self.bias
        return output

class GCNResnet(nn.Module):
    def __init__(self, num_classes=14, in_channel=1792, adj_matrix=None):
        super(GCNResnet, self).__init__()
        
        # Backbone EfficientNet-B4 (Pre-trained, No Classifier)
        self.backbone = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=0, drop_path_rate=0.2, global_pool='')
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # GCN Layers
        self.gcn1 = GraphConvolution(in_channel, in_channel)
        self.gcn2 = GraphConvolution(in_channel, in_channel)
        self.relu = nn.LeakyReLU(0.2)
        
        if adj_matrix is None:
            adj_matrix = torch.eye(num_classes)
        self.register_buffer('adj', adj_matrix)
        
        # Label Embeddings: Vector đặc trưng khởi tạo đại diện cho 14 bệnh
        self.label_embeddings = nn.Parameter(torch.Tensor(num_classes, in_channel))
        nn.init.xavier_uniform_(self.label_embeddings)

    def forward(self, x):
        # 1. Image Features
        features = self.backbone(x) 
        features = self.pooling(features).view(features.size(0), -1) 
        
        # 2. Graph Reasoning: Embeddings -> Classifier Weights
        w = self.relu(self.gcn1(self.label_embeddings, self.adj))
        w = self.gcn2(w, self.adj) 
        
        # 3. Classification (Dot Product)
        logits = torch.matmul(features, w.t())
        return logits