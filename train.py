import os
import random
import torch
import warnings
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
warnings.filterwarnings('ignore')


def extract_dataloaders(entries, batch_size):
    random.shuffle(entries)
    train_idx = torch.arange(len(entries))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    return train_loader


def draw_loss(loss_list, dataset_name):
    import matplotlib.pyplot as plt
    path = f'./figs/train_loss_{dataset_name}.png'
    # draw training loss
    plt.plot(loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    # plt.show()


def main(main_args):
    # default CPU
    device = main_args.device if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset
    if dataset_name == 'streamspot':
        main_args.num_hidden = 256
        main_args.max_epoch = 5
        main_args.num_layers = 4
    elif dataset_name == 'wget':
        main_args.num_hidden = 256
        main_args.max_epoch = 2
        main_args.num_layers = 4
    elif dataset_name == 'optc_zeek':
        main_args.num_hidden = 12
        main_args.max_epoch = 50
        main_args.num_layers = 3
    elif dataset_name == 'optc_day23' or dataset_name == 'optc_day24' or dataset_name == 'optc_day25':
        main_args.num_hidden = 64
        main_args.max_epoch = 75
        main_args.num_layers = 3
    elif dataset_name == 'lanl':
        main_args.num_hidden = 64
        main_args.max_epoch = 50
        main_args.num_layers = 4
        main_args.lr = 5e-4
    elif dataset_name == 'lanl-flow':
        main_args.num_hidden = 64
        main_args.max_epoch = 75
        main_args.num_layers = 4
        main_args.lr = 5e-4
    else:
        main_args.num_hidden = 64
        main_args.max_epoch = 125
        main_args.num_layers = 3
    set_random_seed(0)

    if dataset_name == 'streamspot' or dataset_name == 'wget':
        if dataset_name == 'streamspot':
            batch_size = 12
        else:
            batch_size = 1
        dataset = load_batch_level_dataset(dataset_name)
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        graphs = dataset['dataset']
        train_index = dataset['train_index']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat
        model = build_model(main_args)
        model = model.to(device)
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        model = batch_level_train(model, graphs, (extract_dataloaders(train_index, batch_size)),
                                  optimizer, main_args.max_epoch, device, main_args.n_dim, main_args.e_dim)
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))

    else:
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']
        model = build_model(main_args)
        model = model.to(device)
        model.train()
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        epoch_iter = tqdm(range(main_args.max_epoch))
        n_train = metadata['n_train']
        loss_list = []
        best_loss = float('inf')
        for epoch in epoch_iter:
            epoch_loss = 0.0
            for i in range(n_train):
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                loss = model(g)
                loss /= n_train
                optimizer.zero_grad()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                del g
            loss_list.append(epoch_loss)
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")
            if epoch > 40 and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
        # torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
        save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset_name)
        draw_loss(loss_list, dataset_name)
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)
    return


if __name__ == '__main__':
    args = build_args()
    main(args)
