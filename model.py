import torch.nn as nn
import torch.nn.functional as F
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score


class DiffPoolingBlock(torch.nn.Module):
    def __init__(self, in_dim, n_clusters, tau=2, dim=-1, n_layer=1):
        super().__init__()
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_dim, n_clusters))
        for i in range(n_layer-1):
            self.linear.append(nn.Linear(n_clusters, n_clusters))
        self.softmax = nn.Softmax(dim=dim)
        self.activation = nn.ReLU()
        self.tau = tau
        self.dim = dim

    def reset_parameters(self):
        for layer in self.linear:
            layer.reset_parameters()

    def forward(self, h, adj):
        out = h
        for i in range(len(self.linear)):
            out = self.activation(torch.mm(adj, self.linear[i](out)) + 1e-15)
        return self.softmax(out/self.tau)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, readout, config):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn1 = GCN(n_in, 2 * n_h, activation)
        self.gcn2 = GCN(2 * n_h, n_h, activation)
        self.act = nn.PReLU()
        self.fc1 = nn.Linear(n_h, 2 * n_h, bias=False)
        self.diffpool = DiffPoolingBlock(n_in, n_clusters=config['clusters'], dim=-1)
        self.ReLU = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()
        self.tau = 1
        self.lamb = config['lamb']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.norm = config['norm']

    def normalize_adj_tensor(self, adj):
        row_sum = torch.sum(adj, 0)
        r_inv = torch.pow(row_sum, -0.5).flatten()
        r_inv = torch.nan_to_num(r_inv, posinf=0, neginf=0)
        adj = torch.mm(adj, torch.diag_embed(r_inv))
        adj = torch.mm(torch.diag_embed(r_inv), adj)
        return adj

    def forward(self, seq, adj, raw_adj, adj2=None,  raw_adj2=None, sparse=False):
        if adj2 is not None:
            feat1 = self.gcn2(self.gcn1(seq, adj[0]), adj[0])
            feat2 = self.gcn2(self.gcn1(seq, adj2[0]), adj2[0])
            feat = (feat1 + feat2)/2
            s_adj = (adj[0] + adj2[0])/2
        else:
            feat = self.gcn2(self.gcn1(seq, adj[0]), adj[0])
            s_adj = adj[0]
            raw_adj2 = raw_adj
        assign = self.diffpool(seq, s_adj)
        self.G = assign
        raw_adj = (raw_adj[0] + raw_adj2[0])/2
        con_loss = self.graph_contrastive_loss(feat, feat, assign, raw_adj)
        cluster_sim = self.alpha * torch.mm(assign, assign.T) + raw_adj * (1 - self.alpha)
        if self.norm:
            cluster_sim = self.normalize_adj_tensor(cluster_sim)
        if adj2 is None:
            view_consistency = 0
        else:
            view_consistency = torch.norm(feat1 - feat2, dim=1, p=2).mean()
        loss, message_sum1 = self.max_message(feat, cluster_sim)
        fc1 = self.fc1(feat)
        if self.beta != 0:
            loss += self.beta * self.reg_edge(fc1, adj[0])
        return feat, cluster_sim, loss + self.lamb * (con_loss + view_consistency)

    def graph_contrastive_loss(self, v1, v2, assignment, adj, test_phase=False):
        if v1.shape[0] > 10000 and not test_phase:
            idx = torch.randperm(v1.shape[0])[:8000]
        else:
            idx = torch.arange(v1.shape[0])
        v1, v2, assignment = v1[idx], v2[idx], assignment[idx]
        adj = adj[idx, :][:, idx]
        v = (v1 + v2)/2
        sim = self.cosine_sim(v, v, self.tau)
        O = self.alpha * torch.matmul(assignment, assignment.T) + (1 - self.alpha) * adj
        n = assignment.shape[0]
        loss = torch.norm(O / O.sum(dim=1).clamp_min(0.01) - sim, 2, dim=1).mean()
        return loss

    def view_consistency(self, seq, adj, adj2, sparse=False):
        if adj2 is None:
            return 0
        feat1 = self.gcn2(self.gcn1(seq, adj[0]), adj[0])
        feat2 = self.gcn2(self.gcn1(seq, adj2[0]), adj2[0])
        view_consistency = torch.norm(feat1 - feat2, dim=1, p=2)
        return view_consistency * self.lamb

    def cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)

    def reg_edge(self, emb, adj):
        emb = emb / torch.norm(emb, dim=-1, keepdim=True)
        sim_u_u = torch.mm(emb, emb.T)
        adj_inverse = (1 - adj)
        sim_u_u = sim_u_u * adj_inverse
        sim_u_u_no_diag = torch.sum(sim_u_u, 1)
        row_sum = torch.sum(adj_inverse, 1)
        r_inv = torch.pow(row_sum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        sim_u_u_no_diag = sim_u_u_no_diag * r_inv
        loss_reg = torch.sum(sim_u_u_no_diag)
        return loss_reg

    def max_message(self, feature, adj_matrix):
        feature = feature / torch.norm(feature, dim=-1, keepdim=True)
        sim_matrix = torch.mm(feature, feature.T)
        sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
        sim_matrix[torch.isinf(sim_matrix)] = 0
        sim_matrix[torch.isnan(sim_matrix)] = 0
        row_sum = torch.sum(adj_matrix, 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        message = torch.sum(sim_matrix, 1)
        message = message * r_inv
        return - torch.sum(message), message

    def inference(self, feature, adj_matrix):
        feature = feature / torch.norm(feature, dim=-1, keepdim=True)
        sim_matrix = torch.mm(feature, feature.T)
        sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
        row_sum = torch.sum(adj_matrix, 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        message = torch.sum(sim_matrix, 1)
        message = message * r_inv
        return message

    def evaluation(self, score, ano_label):
        auc = roc_auc_score(ano_label, score)
        AP = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)
        print('AP::{:.4f}, AUC:{:.4f}'.format(AP, auc))
