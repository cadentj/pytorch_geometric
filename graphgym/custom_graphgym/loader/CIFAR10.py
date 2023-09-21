import torch
import datasets
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data

def cast(graph):
    return {
        'x': torch.FloatTensor(graph['node_feat']),
        'edge_index': torch.LongTensor(graph['edge_index']),
        'edge_attr': torch.FloatTensor(graph['edge_attr']),
        'y': torch.LongTensor(graph['y']),
        'pos': torch.FloatTensor(graph['pos'])
    }


class Dataset(InMemoryDataset):
    def __init__(self, root, data_list=None, transform=None):
        if data_list is not None:
            self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.pt'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


def CIFAR10(dataset_dir):
    # Download CIFAR10 graphs-dataset
    # https://huggingface.co/datasets/graphs-datasets/CIFAR10
    data = datasets.load_dataset("graphs-datasets/CIFAR10")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    # Load all hf datasets into python lists
    # dataset_train_list = [Data.from_dict(cast(graph)).to(device) for graph in tqdm(data["train"])]
    # dataset_test_list = [Data.from_dict(cast(graph)).to(device) for graph in tqdm(data["test"])]
    dataset_val_list = [Data.from_dict(cast(graph)).to(
        device) for graph in tqdm(data["val"])]

    # cifarData = Dataset('CIFAR10', dataset_train_list + dataset_test_list + dataset_val_list)
    
    cifarData = Dataset(dataset_dir, dataset_val_list)

    train_indices, temp_indices = train_test_split(
        range(len(cifarData)), test_size=0.2, random_state=42)
    temp_indices, val_indices = train_test_split(
        cifarData, test_size=0.25, random_state=42)

    cifarData.data['train_graph_index'] = torch.tensor(train_indices)
    cifarData.data['test_graph_index'] = torch.tensor(temp_indices)
    cifarData.data['val_graph_index'] = torch.tensor(val_indices)

    return cifarData
