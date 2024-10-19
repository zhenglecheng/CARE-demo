import os.path
from utils import *
from model import Model
import argparse
import time


def main(args):
    seed =args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Load and preprocess data
    raw_features, features, adj1, adj2, ano_label, raw_adj1, raw_adj2, config = load_dataset(args)
    # Initialize model and optimizer
    optimiser_list = []
    model_list = []
    for i in range(args.cutting):
        model = Model(config['ft_size'], args.embedding_dim, 'prelu', args.readout, config)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if torch.cuda.is_available():
            model = model.cuda()
            optimiser_list.append(optimiser)
            model_list.append(model)

    if torch.cuda.is_available():
        print('Using CUDA')
        features = features.cuda()
        raw_features = raw_features.cuda()
        raw_adj1 = raw_adj1.cuda()
        if raw_adj2 is not None:
            raw_adj2 = raw_adj2.cuda()

    start = time.time()
    # Train model
    total_epoch = args.num_epoch * config['cutting']
    new_adj_list1 = []
    new_adj_list2 = []
    new_adj_list1.append(raw_adj1)
    all_cut_adj1 = torch.cat(new_adj_list1)
    if os.path.exists('./data/{}_distance1.npy'.format(args.dataset)):
        dis_array1 = torch.FloatTensor(np.load('./data/{}_distance1.npy'.format(args.dataset)))
    else:
        dis_array1 = calc_distance(raw_adj1[0, :, :], raw_features[0, :, :])
        np.save('./data/{}_distance1.npy'.format(args.dataset), dis_array1.cpu().numpy())
    if raw_adj2 is not None:
        new_adj_list2.append(raw_adj2)
        all_cut_adj2 = torch.cat(new_adj_list2)
        if os.path.exists('./data/{}_distance2.npy'.format(args.dataset)):
            dis_array2 = torch.FloatTensor(np.load('./data/{}_distance2.npy'.format(args.dataset)))
        else:
            dis_array2 = calc_distance(raw_adj1[0, :, :], raw_features[0, :, :])
            np.save('./data/{}_distance2.npy'.format(args.dataset), dis_array2.cpu().numpy())
    index = 0
    message_mean_list = []
    for n_cut in range(config['cutting']):
        message_list = []
        optimiser_list[index].zero_grad()
        model_list[index].train()
        if torch.cuda.is_available():
            dis_array1 = dis_array1.cuda()
        cut_adj1 = graph_nsgt(dis_array1, all_cut_adj1[0, :, :])
        dis_array1 = dis_array1.cpu()
        cut_adj1 = cut_adj1.unsqueeze(0)
        adj_norm1 = normalize_adj_tensor(cut_adj1, args.dataset)
        if raw_adj2 is not None:
            if torch.cuda.is_available():
                dis_array2 = dis_array2.cuda()
            cut_adj2 = graph_nsgt(dis_array2, all_cut_adj2[0, :, :])
            dis_array2 = dis_array2.cpu()
            cut_adj2 = cut_adj2.unsqueeze(0)
            adj_norm2 = normalize_adj_tensor(cut_adj2)
        else:
            adj_norm2 = None
        for epoch in range(args.num_epoch):
            node_emb, cluster_sim, loss = model_list[index].forward(features[0], adj_norm1, raw_adj1, adj_norm2, raw_adj2)
            loss.backward()
            optimiser_list[index].step()
            loss = loss.detach().cpu().numpy()
            if (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], time: {:.4f}, Loss: {:.4f}'.format(n_cut * args.num_epoch + epoch + 1,
                                                                         total_epoch, time.time() - start, loss))
        message_sum = model_list[index].inference(node_emb, cluster_sim) + model_list[index].view_consistency(features[0], adj_norm1, adj_norm2)
        message_list.append(torch.unsqueeze(message_sum.cpu().detach(), 0))
        all_cut_adj1[0, :, :] = torch.squeeze(cut_adj1)
        if raw_adj2 is not None:
            all_cut_adj2[0, :, :] = torch.squeeze(cut_adj2)
        index += 1
        message_list = torch.mean(torch.cat(message_list), 0)
        message_mean_list.append(torch.unsqueeze(message_list, 0))
        message_mean_cut = torch.mean(torch.cat(message_mean_list), 0)
        message_mean = np.array(message_mean_cut.cpu().detach())
        score = 1 - normalize_score(message_mean)
        model_list[index-1].evaluation(score, ano_label)
    end = time.time()
    print(end - start)


if __name__ == '__main__':
    # Set argument
    parser = argparse.ArgumentParser(description='Cluster-Aware Graph Anomaly Detection (CARE-demo)')
    parser.add_argument('--dataset', type=str, default='Amazon', help='Amazon | BlogCatalog | imdb | dblp')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
    parser.add_argument('--cutting', type=int, default=25)  # 7, 15, 25
    parser.add_argument('--lamb', type=float, default=0.1)  # 0  0.5  1
    parser.add_argument('--alpha', type=float, default=0.8)  # 0  0.5  1
    parser.add_argument('--clusters', type=int, default=10)  # 0  0.5  1
    args = parser.parse_args()
    print('Dataset: ', args.dataset)
    main(args)
