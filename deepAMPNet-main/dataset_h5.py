import os
import torch
import numpy as np
import pandas as pd
from Bio import PDB
from torch_geometric.data import Data, InMemoryDataset
import argparse
import warnings
warnings.filterwarnings('ignore')
import h5py
from Bio.SeqUtils import seq1  # <-- 在这里添加这一行

def load_data(pdb_dir, h5_file, label_file=False, threshold=20):

    if label_file:
        AMPs_name = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                AMPs_name.append(line)
    graph_labels = torch.tensor([0, 1], dtype=torch.long)

    pdbfile = os.listdir(pdb_dir)
    h5file = h5py.File(h5_file, 'r')

    data_list = []
    for name in pdbfile:

        seq_name = name.replace(".pdb", "")
        if label_file:
            y = graph_labels[1] if seq_name in AMPs_name else graph_labels[0]
        else:
            y = graph_labels[1] if 'Non' not in seq_name else graph_labels[0]

        pdb_path = os.path.join(pdb_dir, name)
        parser = PDB.PDBParser()
        structure = parser.get_structure('protein', pdb_path)

        amino_acid = ''
        ca_coordinates = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue):
                        amino_acid += seq1(residue.get_resname())
                        ca_atom = residue['CA']
                        ca_coordinates.append(ca_atom.get_coord())
        ca_coordinates = np.array(ca_coordinates)

        ca = torch.from_numpy(ca_coordinates)
        edge_index = []
        for i in range(len(ca_coordinates)):
            for j in range(i + 1, len(ca_coordinates)):
                distance = torch.norm(ca[i] - ca[j])
                if distance < threshold:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        node_features = h5file[seq_name][:]
        x = torch.from_numpy(node_features)

        data = Data(x=x, edge_index=edge_index, y=y, id=seq_name)
        data_list.append(data)
    # loader = DataLoader(data_list, batch_size=32)
    return data_list

# // file: dataset_h5.py

class StructureDataset(InMemoryDataset):
    # 1. 在 __init__ 中添加 pdb_dir, h5_file 等参数
    def __init__(self, root, pdb_dir, h5_file, label_file=None, threshold=20, transform=None, pre_transform=None):
        # 2. 将这些参数保存为实例属性
        self.pdb_dir = pdb_dir
        self.h5_file = h5_file
        self.label_file = label_file
        self.threshold = threshold
        
        # 必须在super().__init__之前定义好所有后面process会用到的属性
        super(StructureDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # 虽然我们不直接用这个，但定义好原始数据目录的逻辑在这里更清晰
        return os.listdir(self.pdb_dir)

    @property
    def processed_file_names(self):
        return ['AMP_NonAMP.dataset']

    def download(self):
        pass

    def process(self):
        # 3. 在 process 方法中使用 self.xxx 来调用这些路径
        data_list = load_data(self.pdb_dir, self.h5_file, self.label_file, self.threshold)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, help='path of pdb dir')
    parser.add_argument('--h5', type=str, help='path of h5 dir')
    parser.add_argument('--label', type=str, default=False, help='path of label file')
    parser.add_argument('--threshold', type=int, default=20, help='path of data')
    parser.add_argument('--root', type=str, help='path of store dataset')
    args = parser.parse_args()

    pdb_dir = args.pdb
    h5_file = args.h5
    label_file = args.label
    threshold = args.threshold
    root = args.root

    # 这里的调用需要更新，传入所有必需的参数
    dataset = StructureDataset(root=root, 
                               pdb_dir=pdb_dir, 
                               h5_file=h5_file, 
                               label_file=label_file, 
                               threshold=threshold)
