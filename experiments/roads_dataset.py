import os.path as osp
from typing import Callable, List, Optional

import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce
from torch_geometric.transforms import NormalizeFeatures

class Roads(InMemoryDataset):
    r"""The Roads dataset from SSTD 2005 "On Trip Planning Queries in Spatial Databases"
    https://users.cs.utah.edu/~lifeifei/SpatialDataset.htm
    
    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"cal"`, :obj:`"SF"`, 
            :obj:`"NA"`, :obj:`"TG"`, :obj:`"OL"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name
        assert self.name in ['cal','SF','NA','TG','OL']
        self.edge_url = ('https://users.cs.utah.edu/~lifeifei/research/tpq/{}.cedge')
        self.node_url = ('https://users.cs.utah.edu/~lifeifei/research/tpq/{}.cnode')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.name}.cnode', f'{self.name}.cedge']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        download_url(self.edge_url.format(self.name), self.raw_dir)
        download_url(self.node_url.format(self.name), self.raw_dir)

    def process(self) -> None:
        index_map, xs = {}, []
        with open(self.raw_paths[0], 'r') as f:
            rows = f.read().split('\n')[:-1] #see
            for i, row in enumerate(rows):
                idx, long, lat = row.split()
                index_map[int(idx)] = i
                xs.append([float(long), float(lat)])
        x = torch.tensor(xs)
        
        vert_coord = x.numpy()
        bin_edges = np.linspace(min(vert_coord[:,1]), max(vert_coord[:,1]), num=11)
        y = np.digitize(vert_coord[:,1], bins=bin_edges, right=True)
#         y = torch.tensor(ys, dtype=torch.long)

        edge_indices, edge_weight = [], []
        with open(self.raw_paths[1], 'r') as f:
            rows = f.read().split('\n')[:-1] #see
            for row in rows:
                _, src, dst, w = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
                edge_weight.append(float(w))
                
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_index = coalesce(edge_index, num_nodes=x.size(0))
        
        edge_weight = torch.tensor(edge_weight)
        edge_weight = 1.01-torch.exp((edge_weight - max(edge_weight))/max(edge_weight))

        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name.capitalize()}Roads()'