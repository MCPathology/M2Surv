import os, sys
import argparse
from os.path import join
import h5py
import math
from math import floor
import pdb
from time import time
from tqdm import tqdm

### Numerical Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import percentileofscore

### Graph Network Packages
import nmslib
import networkx as nx

### PyTorch / PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import convert


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def pt2graph(wsi_h5, radius=5):
    from torch_geometric.data import Data as geomData
    from itertools import chain
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]
    
    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_spatial = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)
    
    model = Hnsw(space='l2')
    model.fit(features)
    a = np.repeat(range(num_patches), radius-1)
    b = np.fromiter(chain(*[model.query(features[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),dtype=int)
    edge_latent = torch.Tensor(np.stack([a,b])).type(torch.LongTensor)

    G = geomData(x = torch.Tensor(features),
                 edge_index = edge_spatial,
                 edge_latent = edge_latent,
                 centroid = torch.Tensor(coords))
    return G

def pt2wholegraph(ff_slide, ffpe_slide, radius=9):
    from torch_geometric.data import Data as geomData
    from itertools import chain

    def process_slide(slide, current_node_num, radius):
        try:
            wsi_h5 = h5py.File(slide, "r")
            coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
            assert coords.shape[0] == features.shape[0]
            
            num_patches = coords.shape[0]

            model = Hnsw(space='l2')
            model.fit(coords)
            
            a = np.repeat(range(current_node_num, current_node_num + num_patches), radius - 1)
            b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int) + current_node_num
            
            edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
            
            wsi_h5.close()
            
            return features, edge_spatial
        
        except OSError:
            print(f'{slide} - Broken H5')
            return None, None

    # Process the single FF slide
    ff_features, ff_edge_spatial = process_slide(ff_slide, 0, radius)

    # Process the single FFPE slide with updated node count starting point
    # ffpe_features_start_index = ff_features.shape[0] if ff_features is not None else 0
    ffpe_features, ffpe_edge_spatial = process_slide(ffpe_slide, 0, radius)

    # Combine all features for shared edge computation based on feature similarity
    if True:
        whole_features_combined = np.concatenate([ff_features, ffpe_features], axis=0)

        model_shared_edges = Hnsw(space='l2')
        model_shared_edges.fit(whole_features_combined)

        a_shared = np.repeat(range(whole_features_combined.shape[0]), radius - 1)
        
        # Create shared edges based on feature similarity across both datasets (FF + FFPE combined).
        b_shared = np.fromiter(chain(*[model_shared_edges.query(whole_features_combined[v_idx], topn=radius)[1:] 
                                       for v_idx in range(whole_features_combined.shape[0])]), dtype=int)

        edge_latent_shared = torch.Tensor(np.stack([a_shared,b_shared])).type(torch.LongTensor)
        # print(b_shared)
        G = geomData(ff=torch.Tensor(ff_features),
                     ffpe=torch.Tensor(ffpe_features),
                     ff_edge_index=ff_edge_spatial,
                     ffpe_edge_index=ffpe_edge_spatial,
                     share_edge=edge_latent_shared) 

        return G
    
    else:
        print("Error processing slides.")
        return None

    
def createDir_h5toWholeG(ff_path, ffpe_path, save_path):
    import os
    import torch
    from tqdm import tqdm

    # Get list of all files in the FFPE directory
    ffpe_list = os.listdir(ffpe_path)
    
    # Extract unique prefixes from filenames (assuming first 12 characters are used as a prefix)
    unique_prefixes = set(filename[:12] for filename in ffpe_list)
    
    # Convert set to list for iteration
    ffpe_name = list(unique_prefixes)
    
    pbar = tqdm(ffpe_name)

    # List to store cases where matching files are not found
    missing_cases = []
    
    for case in pbar:
        pbar.set_description('%s - Creating Graph' % (case))
        
        ff_h5 = None
        ffpe_h5 = None
    
        # Find one matching FFPE file for the current case
        for filename in os.listdir(ffpe_path):
            if case in filename:
                full_path_ffpe = os.path.join(ffpe_path, filename)
                ffpe_h5 = full_path_ffpe
                break  # Stop after finding one match
        
        # Find one matching FF file for the current case
        for filename in os.listdir(ff_path):
            if case in filename:
                full_path_ff = os.path.join(ff_path, filename)
                ff_h5 = full_path_ff
                break  # Stop after finding one match
        
        if ff_h5 and ffpe_h5:
            G = pt2wholegraph(ff_h5, ffpe_h5, radius=9)  # Call your graph creation function with single files
            torch.save(G, os.path.join(save_path, case + '.pt'))
        else:
            print(f"Could not find both FF and FFPE files for case {case}")
            missing_cases.append(case)

    # Append missing cases to a text file
    with open(os.path.join(save_path, 'missing_cases.txt'), 'a') as f:  # Open in append mode ('a')
        for case in missing_cases:
            f.write(f"{case}\n")

def createDir_h5toPyG(h5_path, save_path):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))

        try:
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G = pt2graph(wsi_h5,radius=9)
            torch.save(G, os.path.join(save_path, h5_fname[:-3]+'.pt'))
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')

def main(args):
    ffpe_path = args.ffpe_path
    save_path = args.graph_save_path
    os.makedirs(save_path, exist_ok=True)
    ff_path = args.ff_path
    createDir_h5toWholeG(ff_path, ffpe_path, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ffpe_path', type = str,default='',
                        help='path to FFPE h5 files proceeded by CLAM')
    parser.add_argument('--ff_path', type = str,default='',
                        help='path to FF h5 files proceeded by CLAM')
    parser.add_argument('--graph_save_path', type = str,default='',
                        help='path to store the generated graph')
    args = parser.parse_args()
    results = main(args)
    print("finished!")
