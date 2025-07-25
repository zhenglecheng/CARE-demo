import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import copy
import random
import os
import pickle as pkl


def load_dataset(args):
    config = dict()
    config['beta'] = 0
    config['norm'] = True
    config['clusters'] = args.clusters
    if args.dataset == 'Amazon':
        adj1, adj2, features, ano_label = load_mat(args.dataset)
        features, _ = preprocess_features(features)
        raw_features = features
        config['cutting'] = 25
        config['lamb'] = 1
        config['alpha'] = 0.8
        config['norm'] = False
    elif args.dataset == 'YelpChi':
        adj1, adj2, features, ano_label = load_mat(args.dataset)
        features, _ = preprocess_features(features)
        raw_features = features
        config['cutting'] = 3
        config['lamb'] = 1
        config['alpha'] = 0.8
        config['norm'] = False
    elif args.dataset == 'dblp':
        adj1, adj2, features, ano_label = load_dblp_graph()
        raw_features = features
        config['cutting'] = 20
        config['lamb'] = 0.01
        config['alpha'] = 1
    elif args.dataset == 'imdb':
        adj1, adj2, features, ano_label = load_imdb_graph()
        raw_features = features
        config['cutting'] = 15
        config['lamb'] = 0.01
        config['alpha'] = 1
    elif args.dataset == 'cert':
        adj1, adj2, features, ano_label = load_cert_graph()
        raw_features = features
        config['cutting'] = 7
        config['lamb'] = 0.1
        config['alpha'] = 0.8
    else:
        adj1, adj2, features, ano_label = load_mat(args.dataset)
        raw_features = features.todense()
        features = raw_features
        config['cutting'] = 7
        config['beta'] = 1
        if args.dataset == 'BlogCatalog':
            config['clusters'] = 5
            config['lamb'] = 0.01
            config['alpha'] = 0
    config['ft_size'] = features.shape[1]
    raw_adj1 = (adj1 + sp.eye(adj1.shape[0])).todense()
    raw_adj1 = torch.FloatTensor(raw_adj1[np.newaxis])
    if adj2 is None:
        raw_adj2 = None
    else:
        raw_adj2 = (adj2 + sp.eye(adj2.shape[0])).todense()
        raw_adj2 = torch.FloatTensor(raw_adj2[np.newaxis])
    raw_features = torch.FloatTensor(raw_features[np.newaxis])
    features = torch.FloatTensor(features[np.newaxis])
    return raw_features, features, adj1, adj2, ano_label, raw_adj1, raw_adj2, config

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def RandomDropEdge(x, adj_t, drop_percent=0.05):
    percent = drop_percent / 2
    row_idx, col_idx = x.nonzero().T

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    edge_num = int(len(row_idx) / 2)  # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = copy.deepcopy(adj_t)

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[index_list[i][0]][index_list[i][1]] = 0
        aug_adj[index_list[i][1]][index_list[i][0]] = 0

    node_num = x.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    adj = aug_adj.to_sparse()
    return adj


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj_tensor(raw_adj, dataset='BlogCatalog'):
    # if dataset =='BlogCatalog':
    #     return raw_adj
    adj = raw_adj[0, :, :]
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    adj = adj.unsqueeze(0)
    return adj


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score


def process_dis(init_value, cutting_dis_array):
    r_inv = np.power(init_value, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    cutting_dis_array = cutting_dis_array.dot(sp.diags(r_inv))
    cutting_dis_array = sp.diags(r_inv).dot(cutting_dis_array)
    return cutting_dis_array


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset):
    """Load .mat dataset."""

    data = sio.loadmat("./data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    adj1 = adj
    adj2 = None
    return adj1, adj2, feat, ano_labels


from scipy import sparse
def load_dblp_graph():
    if not os.path.exists('./data/DBLP_anomaly.mat'):
        data = sio.loadmat('./data/DBLP')
        features = data['feature'].astype(np.float64)
        label = np.zeros((features.shape[0], 1))
        p = 0.07
        index = np.random.choice(features.shape[0], int(p * features.shape[0]), replace=False)
        label[index] = 1
        feature_anomaly_idx = index[:index.shape[0]//2]
        edge_anomaly_idx = index[index.shape[0]//2:]
        features[feature_anomaly_idx] += np.random.normal(2, 10, size=(feature_anomaly_idx.shape[0], features.shape[1]))
        adj1 = data['net_APA'] + np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], data['net_APA'].shape[1]))
        adj2 = data['net_APCPA'] + np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], data['net_APCPA'].shape[1]))
        data = {'feature': features, 'label': label, 'adj1': adj1, 'adj2': adj2}
        sio.savemat('./data/dblp_anomaly.mat', data)
    else:
        data = sio.loadmat('./data/dblp_anomaly.mat')
        features = data['feature']
        adj1 = data['adj1']
        adj2 = data['adj2']
        label = data['label'].reshape(-1, )
    # print('size of anomalies:', sum(label))
    # features = sparse.csr_matrix(features)
    adj1 = sparse.csr_matrix(adj1)
    adj2 = sparse.csr_matrix(adj2)
    return adj1, adj2, features, label


def load_imdb_graph():
    if not os.path.exists('./data/imdb5k_anomaly.mat'):
        data = sio.loadmat('./data/imdb5k.mat')
        features = data['feature'].astype(np.float64)
        label = np.zeros((features.shape[0], 1))
        p = 0.07
        index = np.random.choice(features.shape[0], int(p * features.shape[0]), replace=False)
        label[index] = 1
        feature_anomaly_idx = index[:index.shape[0]//2]
        edge_anomaly_idx = index[index.shape[0]//2:]
        features[feature_anomaly_idx] += np.random.normal(2, 10, size=(feature_anomaly_idx.shape[0], features.shape[1]))
        adj1 = data['MAM'] + np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], data['MAM'].shape[1]))
        adj2 = data['MDM'] + np.random.binomial(1, 0.5, size=(edge_anomaly_idx.shape[0], data['MDM'].shape[1]))
        data = {'feature': features, 'label': label, 'adj1': adj1, 'adj2': adj2}
        sio.savemat('./data/imdb5k_anomaly.mat', data)
    else:
        data = sio.loadmat('./data/imdb5k_anomaly.mat')
        features = data['feature']
        adj1 = data['adj1']
        adj2 = data['adj2']
        label = data['label'].reshape(-1, )
    # print('size of anomalies:', sum(label))
    # features = sparse.csr_matrix(features)
    adj1 = sparse.csr_matrix(adj1)
    adj2 = sparse.csr_matrix(adj2)
    return adj1, adj2, features, label


def load_cert_graph(d=100):
    v2 = pkl.load(open('./CERT/logon.pkl', 'rb'))
    v1 = pkl.load(open('./CERT/email.pkl', 'rb'))
    malicious_user = pkl.load(open('./CERT/label.pkl', 'rb'))['label']
    label = []
    email_pc_dict = v2['pc_dict']
    email_user_dict = v2['user_dict']
    overlapped_idx = []
    for item, key in email_pc_dict.items():
        if item in v1['pc_dict']:
            overlapped_idx.append(v1['pc_dict'][item])
    v2['graph'] = v2['graph'][overlapped_idx, :]
    v2['weight'] = v2['weight'][overlapped_idx, :]
    overlapped_idx = []
    for item, key in email_user_dict.items():
        if item in v1['user_dict']:
            overlapped_idx.append(v1['user_dict'][item])
        if item in malicious_user:
            label.append(1)
        else:
            label.append(0)
    v2['graph'] = v2['graph'][:, overlapped_idx]
    v2['weight'] = v2['weight'][:, overlapped_idx]
    if not os.path.exists('./CERT/logon_edge_list.txt'):
        n_nodes = v2['weight'].shape[0]
        with open('./CERT/logon_edge_list.txt', 'w') as f:
            for i in range(len(v2['graph'])):
                idx = np.nonzero(v2['graph'][i, :])[0]
                for j in idx:
                    f.write('{} {}\n'.format(i, j + n_nodes))
    if not os.path.exists('./CERT/email_edge_list.txt'):
        n_nodes = v1['weight'].shape[0]
        with open('./CERT/email_edge_list.txt', 'w') as f:
            for i in range(len(v1['graph'])):
                idx = np.nonzero(v1['graph'][i, :])[0]
                for j in idx:
                    f.write('{} {}\n'.format(i, j + n_nodes))
    if not os.path.exists('./CERT/email_edge_list_emb_{}'.format(d)):
        os.system("python ./deepwalk/main.py --representation-size {} --input ./CERT/email_edge_list.txt"
                  " --output ./CERT/email_edge_list_emb_{}".format(d, d))
    if not os.path.exists('./CERT/logon_edge_list_emb_{}'.format(d)):
        os.system("python ./deepwalk/main.py --representation-size {} --input ./CERT/logon_edge_list.txt"
                  " --output ./CERT/logon_edge_list_emb_{}".format(d, d))
    v1_feature = np.zeros((v1['weight'].shape[0], d))
    v2_feature = np.zeros((v2['weight'].shape[0], d))
    with open('./CERT/logon_edge_list_emb_{}'.format(d), 'r') as f:
        next(f)
        for line in f.readlines():
            line = list(map(float, line.split()))
            if line[0] >= 1000:
                v1_feature[int(line[0]) - 1000] = line[1:]
    with open('./CERT/email_edge_list_emb_{}'.format(d), 'r') as f:
        next(f)
        for line in f.readlines():
            line = list(map(float, line.split()))
            if line[0] >= 1000:
                v2_feature[int(line[0]) - 1000] = line[1:]
    v1_feats = v1_feature
    v2_feats = v2_feature
    # v1_adj = np.dot(v1['graph'], v1['graph'].transpose())
    # v2_adj = np.dot(v2['graph'], v2['graph'].transpose())
    v1_adj = v1['graph']
    v2_adj = v2['graph']
    label = np.array(label).reshape(-1, 1)
    adj1 = sparse.csr_matrix(v1_adj)
    adj2 = sparse.csr_matrix(v2_adj)
    return adj1, adj2, v2_feats, label


# compute the distance between each node
def calc_distance(adj, seq):
    dis_array = torch.zeros((adj.shape[0], adj.shape[1]))
    row = adj.shape[0]
    for i in range(row):
        # print(i)
        node_index = torch.argwhere(adj[i, :] > 0)
        for j in node_index:
            dis = torch.sqrt(torch.sum((seq[i] - seq[j]) * (seq[i] - seq[j])))
            dis_array[i][j] = dis
    return dis_array


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def calc_sim(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = np.zeros((row, col))
    for i in range(row):
        # print(i)
        node_index = np.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            dis = get_cos_similar(attr_matrix[i].tolist(), attr_matrix[j].tolist())
            dis_array[i][j] = dis

    return dis_array


def graph_nsgt(dis_array, adj):
    dis_array = dis_array.cuda()
    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0)
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere(dis_array[i, node_index[:, 0]] > random_value)
                if cutting_edge.shape[0] != 0:
                    adj[i, node_index[cutting_edge[:, 0]]] = 0
    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (8.5, 7.5)
from matplotlib.backends.backend_pdf import PdfPages

def draw_pdf(message, ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        abnormal_message_all = message[ano_label == 1]
        message_all = [normal_message_all, abnormal_message_all]
        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of abnormal {}'.format(mu_1))
        print('The std of abnormal {}'.format(sigma_1))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)
        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.legend(loc='upper left', fontsize=30)
        plt.title(''.format(dataset), fontsize=25)
        plt.show()


def draw_pdf_str_attr(message, ano_label, str_ano_label, attr_ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        str_abnormal_message_all = message[str_ano_label == 1]
        attr_abnormal_message_all = message[attr_ano_label == 1]
        message_all = [normal_message_all, str_abnormal_message_all, attr_abnormal_message_all]

        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of str_abnormal {}'.format(mu_1))
        print('The std of str_abnormal {}'.format(sigma_1))
        mu_2 = np.mean(message_all[2])
        sigma_2 = np.std(message_all[2])
        print('The mean of attt_abnormal {}'.format(mu_2))
        print('The std of attt_abnormal {}'.format(sigma_2))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Structural Abnormal', 'Contextual Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)
        y_2= mlab.normpdf(bins, mu_2, sigma_2)  #

        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_2, color='green', linestyle='--', linewidth=3.5)

        plt.xlabel('TAM-based Affinity', fontsize=25)
        plt.ylabel('Number of Samples', size=25)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(loc='upper left', fontsize=18)
        plt.title('{}'.format(dataset), fontsize=25)
        plt.show()



