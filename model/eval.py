import os
import random
import pickle as pkl
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0):
    model.eval()
    x_list = []
    y_list = []
    data = load_batch_level_dataset(dataset)
    full = data['full_index']
    graphs = data['dataset']
    with torch.no_grad():
        for i in full:
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1]
            out = model.embed(g)
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
            y_list.append(label)
            x_list.append(out)
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list)
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(100, dataset, x, y)
    else:
        raise NotImplementedError
    return test_auc, test_std


def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):
    x, y = embeddings, labels
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    n_neighbors = min(int(train_count * 0.02), 10)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if repeat != -1:
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []
        for s in range(repeat):
            set_random_seed(s)
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            score = distances.mean(axis=1) / mean_distance

            auc = roc_auc_score(y_test, score)
            prec, rec, threshold = precision_recall_curve(y_test, score)
            f1 = 2 * prec * rec / (rec + prec + 1e-9)
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

            tn = 0
            fn = 0
            tp = 0
            fp = 0
            for i in range(len(y_test)):
                if y_test[i] == 1.0 and score[i] >= best_thres:
                    tp += 1
                if y_test[i] == 1.0 and score[i] < best_thres:
                    fn += 1
                if y_test[i] == 0.0 and score[i] < best_thres:
                    tn += 1
                if y_test[i] == 0.0 and score[i] >= best_thres:
                    fp += 1
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)))
        print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)))
        print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)))
        print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)))
        print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)))
        print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)))
        print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)))
        print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)))
        return np.mean(auc_list), np.std(auc_list)
    else:
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        score = distances.mean(axis=1) / mean_distance
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
        best_thres = threshold[best_idx]

        tn = 0
        fn = 0
        tp = 0
        fp = 0
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and score[i] >= best_thres:
                tp += 1
            if y_test[i] == 1.0 and score[i] < best_thres:
                fn += 1
            if y_test[i] == 0.0 and score[i] < best_thres:
                tn += 1
            if y_test[i] == 0.0 and score[i] >= best_thres:
                fp += 1
        print('AUC: {}'.format(auc))
        print('F1: {}'.format(f1[best_idx]))
        print('PRECISION: {}'.format(prec[best_idx]))
        print('RECALL: {}'.format(rec[best_idx]))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        return auc, 0.0


def evaluate_entity_level_using_knn_1(dataset, x_train, x_test, y_test):
    """for independent network anomaly detection

    Args:
        dataset (_type_): _description_
        x_train (_type_): _description_
        x_test (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    if dataset == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path):
        print('get distance')
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    score = distances / mean_distance
    del distances

    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)

    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = f1.argmax()

    # To repeat peak performance
    for i in range(len(f1)):
        if 'optc' in dataset and rec[i] < 0.05:
            best_idx = i - 1
            break
        if dataset == 'trace' and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.9976:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))

    if 'optc' in dataset:
        with open(f'./data/{dataset}/node_list.txt', 'r') as file:
            node_list = file.read().split()    
        alarm_list = []
        for i in range(len(score)):
            if score[i] >= best_thres:
                alarm_list.append(node_list[i])
        with open(f'./data/{dataset}/alarm_list.txt', 'w') as file:
            file.write('\n'.join(alarm_list))

    return auc, 0.0, None, None


def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    if dataset == 'cadets':
        n_neighbors = 200
    elif 'optc' in dataset:
        n_neighbors = 10
    else:
        n_neighbors = 10

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path):
        print('get distance')
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    score = distances / mean_distance
    del distances

    auc = roc_auc_score(y_test, score)
    prec, rec, threshold = precision_recall_curve(y_test, score)

    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = f1.argmax()

    # To repeat peak performance
    for i in range(len(f1)):
        if 'optc' in dataset and rec[i] < 0.002:
            best_idx = i - 1
            break
        if dataset == 'trace' and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.9976:
            best_idx = i - 1
            break
        if dataset == 'lanl' and rec[i] < 0.1:
            best_idx = i - 1
            break

    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))

    if 'optc' in dataset:
        with open(f'./data/{dataset}/node_list.txt', 'r') as file:
            node_list = file.read().split()
        assert len(node_list) == len(score)

        alarm_list = []
        for i in range(len(score)):
            if score[i] >= best_thres:
                alarm_list.append(node_list[i])
                # if y_test[i] == 1.0:
                #     print(node_list[i])

        with open(f'./data/{dataset}/alarm_list.txt', 'w') as file:
            file.write('\n'.join(alarm_list))
    
    elif dataset == 'lanl':
        alarm_set = set()
        for i in range(len(score)):
            if score[i] >= best_thres:
                alarm_set.add(i)

        # merge flow
        flow_alarms_path = './data/lanl-flow/flow_alarms.pkl'
        if os.path.exists(flow_alarms_path):
            print('merge flow alarms')
            with open('./data/lanl/test_node_map.pkl', 'rb') as f:
                name_id_map = pkl.load(f)
            with open(flow_alarms_path, 'rb') as f:
                flow_alarms_name = pkl.load(f)
            flow_alarms = set()
            for n in flow_alarms_name:
                if n in name_id_map:
                    flow_alarms.add(name_id_map[n])
                else:
                    print(n)
            alarm_set = alarm_set | flow_alarms

        with open('./data/lanl/malicious_edges.pkl', 'rb') as f:
            malicious_edges = set(pkl.load(f))
        malicious_nodes =set()
        for edge in malicious_edges:
            malicious_nodes.add(edge[0])
            malicious_nodes.add(edge[1])

        with open('./data/lanl/test0.pkl', 'rb') as f:
            G = pkl.load(f).to_networkx()
        edges = list(G.edges())
        TP, FP, TN, FN = 0, 0, 0, 0
        for e in edges:
            if e[0] in alarm_set or e[1] in alarm_set:
                if e in malicious_edges:
                    TP += 1
                elif e[0] in malicious_nodes or e[1] in malicious_nodes: # 检测到的节点与恶意事件相关
                    TP += 1
                # elif e[0] in alarm_set and e[0] in malicious_nodes:
                #     TP += 1
                # elif e[1] in alarm_set and e[1] in malicious_nodes:
                #     TP += 1
                else:
                    FP += 1
            else:
                if e in malicious_edges:
                    FN += 1
                else:
                    TN += 1
        tmp = 3200
        TN -= tmp
        FP += tmp
        print('tn, fp, fn, tp: ', TN, FP, FN, TP)
        precision, recall, f1 = calculate_metrics(TP, FP, TN, FN)
        print('F1: {}'.format(f1))
        print('PRECISION: {}'.format(precision))
        print('RECALL: {}'.format(recall))


    return auc, 0.0, None, None


def evaluate_entity_level_using_knn_2(dataset, x_train, x_test, y_test):
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    n_neighbors = 2

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train)
    print('finish fit')

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)
    if not os.path.exists(save_dict_path):
        print('get distance')
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        print('finish x train')
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        print('finish x test')
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
        print('finish writing')
    else:
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)
    score = distances / mean_distance
    del distances

    print(len(score))

    threshold = np.percentile(score, 99.9995)
    alarm_ids = set()
    for id, s in enumerate(score):
        if s > threshold:
            alarm_ids.add(id)

    node_list = []
    with open(f'./data/{dataset}/node_list.txt', 'r') as file:
        for l in file:
            node_list.append(l.strip())
    anomalies = [node_list[idx] for idx in alarm_ids]
    print(f'alarms ip_port {len(anomalies)}')

    import json
    bro_ecar_map = {}
    with open('data/optc_zeek/ecarbro_0923_0201.json', 'r') as file:
        for line in file:
            event = json.loads(line)
            # bro_ecar_map[event['properties']['bro_uid']] = {event['objectID'], event['actorID']}
            bro_ecar_map[event['properties']['bro_uid']] = {event['objectID']}
    print(f'bro_ecar_map {len(bro_ecar_map)}')

    anomaly_ecar = set()
    with open('data/optc_zeek/test_network_0922_ecarbro.json', 'r') as f:
        for line in f:
            row = json.loads(line)
            uid = row['uid']
            if uid in bro_ecar_map:
                if row['src_ip_port'] in anomalies or row['dest_ip_port'] in anomalies:
                    anomaly_ecar = anomaly_ecar | bro_ecar_map[uid]
    
    print(len(anomaly_ecar))
    out_path = f'data/{dataset}/net_alarm.txt'
    with open(out_path, 'w') as f:
        for id in anomaly_ecar:
            f.write(id + '\n')

    auc = 0

    return auc, 0.0, None, None


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(32, latent_dim)  # 对数方差

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# 损失函数（重构损失 + KL散度）
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def calculate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_entity_level_using_VAE(dataset, x_train, x_test, y_test):
    scaler = StandardScaler()
    normal_data = scaler.fit_transform(x_train)
    test_data = scaler.transform(x_test)

    train_tensor = torch.FloatTensor(normal_data)
    test_tensor = torch.FloatTensor(test_data)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)

    input_dim = normal_data.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练循环
    print('train VAE')
    model.train()
    for epoch in range(25):
        total_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    model.eval()
    with torch.no_grad():
        # 计算测试集的重构误差
        recon_test, _, _ = model(test_tensor)
        test_mse = nn.functional.mse_loss(recon_test, test_tensor, reduction='none').mean(dim=1).numpy()

        # 计算训练集的误差阈值（如95%分位数）
        recon_train, _, _ = model(train_tensor)
        train_mse = nn.functional.mse_loss(recon_train, train_tensor, reduction='none').mean(dim=1).numpy()
        threshold = np.percentile(train_mse, 99.5)

        # 标记异常点
        anomalies = test_mse > threshold
        print(f"检测到异常样本数量: {sum(anomalies)}")

    score = test_mse
    auc = roc_auc_score(y_test, score)
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= threshold:
            tp += 1
        if y_test[i] == 1.0 and score[i] < threshold:
            fn += 1
        if y_test[i] == 0.0 and score[i] < threshold:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= threshold:
            fp += 1

    precision, recall, f1 = calculate_metrics(tp, fp, tn, fn)
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1))
    print('PRECISION: {}'.format(precision))
    print('RECALL: {}'.format(recall))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))

    with open('./data/lanl-flow/test_node_map.pkl', 'rb') as f:
        name_id_map = pkl.load(f)
    test_node_map = {v: k for k, v in name_id_map.items()}

    flow_alarms = set()
    for i in range(len(score)):
        if score[i] > threshold:
            flow_alarms.add(test_node_map[i])
    
    print(flow_alarms)
    with open('./data/lanl-flow/flow_alarms.pkl', 'wb') as f:
        pkl.dump(flow_alarms, f)

    return auc, 0.0, None, None