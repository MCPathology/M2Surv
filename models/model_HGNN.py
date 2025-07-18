import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from models.layers.cross_attention import FeedForward, MMAttentionLayer
from models.layers.fusion import GraphFusion, AlignFusion
from models.layers.layers import *
from models.layers.sheaf_builder import *
from torch_scatter import scatter_mean
from .util import initialize_weights
from .util import NystromAttention
from .util import SNN_Block
from .util import MultiheadAttention
import dhg
from dhg.nn import HGNNPConv
from collections import defaultdict
    
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.cross_attention import FeedForward
from models.layers.fusion import AlignFusion
from torch_scatter import scatter_mean
from .util import initialize_weights
from .util import SNN_Block
from .layers.layers import *
import dhg
from dhg.nn import HGNNPConv
from collections import defaultdict

def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class Prototype(nn.Module):
    def __init__(self,
                 x_dim = 256,
                 z_dim = 256,
                 beta = 1e-2,
                 sample_num = 50,
                 topk = 256,
                 num_classes = 4,
                 seed = 1):
        super(PIB, self).__init__()

        self.beta = beta
        self.sample_num = sample_num
        self.topk = topk
        self.num_classes = num_classes
        self.seed = seed
        self.z_dim = z_dim

        # decoder a simple logistic regression as in the paper
        self.decoder_logits = nn.Linear(z_dim, num_classes)

        # design proxies for histology images
        self.proxies = nn.Parameter(torch.empty([num_classes*2, z_dim*2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)

    def gaussian_noise(self, samples, K, seed):
        # works with integers as well as tuples
        if self.training:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K),
                                generator=torch.manual_seed(seed)).cuda() # must be the same seed as the training seed


    def encoder_proxies(self):
        mu_proxy = self.proxies[:, :self.z_dim]
        sigma_proxy = torch.nn.functional.softplus(self.proxies[:, self.z_dim:]) # Make sigma always positive

        return mu_proxy, sigma_proxy

    def forward(self,x, y=None, c=None):
        # 获取固定的代理
        mu_proxy, sigma_proxy = self.encoder_proxies()

        # 计算z
        z = self.encoder_result(x)

        # 计算注意力权重
        z_norm = F.normalize(z, dim=2)
        mu_proxy_norm = F.normalize(mu_proxy, dim=1)
    
        # 计算注意力得分
        att_scores = torch.matmul(z_norm, mu_proxy_norm.T)  # 假设z和mu_proxy都是(batch_size, feature_dim)

        # 通过softmax获取注意力权重
        att_weights = F.softmax(att_scores, dim=-1)

        # 根据注意力权重计算动态proxy
        dynamic_mu = torch.matmul(att_weights, mu_proxy)
    
        return dynamic_mu

class M2Surv(nn.Module):
    def __init__(self, genomic_sizes=[], n_classes=4, fusion="concat", model_size="small",graph_type="HGNN"):
        super(M2Surv, self).__init__()
        self.graph_type = graph_type
        self.genomic_sizes = genomic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024,256,  256], "large": [1024, 1024, 1024, 256]},
        }
        
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        
        fc_ff = []
        for idx in range(len(hidden) - 1):
            fc_ff.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            
            fc_ff.append(nn.ReLU6())
            fc_ff.append(nn.Dropout(0.25))
        self.ff_fc = nn.Sequential(*fc_ff)

        fc_ffpe = []
        for idx in range(len(hidden) - 1):
            fc_ffpe.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            
            fc_ffpe.append(nn.ReLU6())
            fc_ffpe.append(nn.Dropout(0.25))
        self.ffpe_fc = nn.Sequential(*fc_ffpe)
        
        # Genomic Embedding Network
        self.init_per_gene_model(genomic_sizes)
        
        self.graph_p = [HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, use_bn = True, drop_rate=0.25).to("cuda")]
        self.graph_g = [HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, drop_rate=0.25).to("cuda"),
                                        HGNNPConv(256, 256, use_bn = True, drop_rate=0.25).to("cuda")]
    
        # multi-modal fusion
        self.attention_fusion = AlignFusion(
            embedding_dim=256,
            num_heads=8
        )
        
        # Classification Layer
        self.feed_forward = FeedForward(256, dropout=0.25)
        self.layer_norm = nn.LayerNorm(256)
        self.p_norm = nn.LayerNorm(256)
        
        self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1]*2,hidden[-1]//2), nn.ReLU6()]
            )
        
        self.classifier = nn.Linear(hidden[-1]//2, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        x_ff = kwargs["x_ff"]
        x_ffpe = kwargs["x_ffpe"]
        graph = kwargs["graph"]

        
        if x_ff is not None and kwargs["x_omic1"] is not None:
            # all modality 
            ff_features = self.ff_fc(x_ff)[0]
            ffpe_features = self.ffpe_fc(x_ffpe)[0]
            pathology_features = torch.cat((ff_features,ffpe_features), dim=-2)
        
            ff_index = self.get_hyperedge(graph.ff_edge_index)
            ffpe_index = self.get_hyperedge(graph.ffpe_edge_index)
            hyper_latent = self.get_hyperedge(graph.share_edge)
            p_hg = dhg.Hypergraph(num_v=pathology_features.shape[0], e_list=ff_index+ffpe_index+hyper_latent)
            for layer in self.graph_p:
                pathology_features = layer(pathology_features, p_hg)
            x_genomic = [kwargs["x_omic%d" % i].to("cuda") for i in range(1, 7)]
            genomic = [self.gene_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_genomic)] 
            genomics_features = torch.stack(genomic) # [6+331,256]
            memory_dict = {'pathology_feature':pathology_features, 'genomics_feature':genomics_features}
        elif x_ff is not None:
            memory = kwargs["memory"]
            ff_features = self.ff_fc(x_ff)[0]
            ffpe_features = self.ffpe_fc(x_ffpe)[0]
            pathology_features = torch.cat((ff_features,ffpe_features), dim=-2)
        
            ff_index = self.get_hyperedge(graph.ff_edge_index)
            ffpe_index = self.get_hyperedge(graph.ffpe_edge_index)
            hyper_latent = self.get_hyperedge(graph.share_edge)
            p_hg = dhg.Hypergraph(num_v=pathology_features.shape[0], e_list=ff_index+ffpe_index+hyper_latent)
            for layer in self.graph_p:
                pathology_features = layer(pathology_features, p_hg)
            genomics_features = memory.retrieveGene(pathology_features)
            memory_dict = None
        else:
            memory = kwargs["memory"]
            x_genomic = [kwargs["x_omic%d" % i].to("cuda") for i in range(1, 7)]
            genomic = [self.gene_sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_genomic)] 
            genomics_features = torch.stack(genomic) # [6+331,256]
            pathology_features = memory.retrievePathology(genomics_features)
            memory_dict = None
        attn_scores = torch.matmul(genomics_features, pathology_features.T)
        
        k=16
        _, top_pathology_indices = torch.topk(attn_scores, k, dim=1)

        g_total = genomics_features.shape[0]
        token = torch.cat((genomics_features, pathology_features), dim=-2)
        
        source_nodes = torch.arange(g_total).repeat_interleave(k)
        target_nodes = top_pathology_indices.flatten() + g_total

        corss_edge_index = self.get_hyperedge(torch.stack([source_nodes.to("cuda"), target_nodes.to("cuda")]))
        g_hg = dhg.Hypergraph(num_v=token.shape[0], e_list=corss_edge_index)
        
        for layer in self.graph_g:
            token = layer(token, g_hg)
        token_cross = token.unsqueeze(0)
        # print(token.shape)
        # token_cross = self.attention_fusion(token_cross)
        
        token_cross = self.feed_forward(token_cross)
        token_cross = self.layer_norm(token_cross)
       
        gene_embed = token_cross[:, :g_total, :]
        gene_embed = torch.mean(gene_embed, dim=1)
        
        path_embed = token_cross[:, g_total:, :]
        path_embed = torch.mean(path_embed, dim=1)
        
        fusion = self.mm(
                torch.cat([gene_embed, path_embed], dim=1)
            )  
        
        logits = self.classifier(fusion)  # [1, n_classes]
        
        return logits, memory_dict

    def init_per_gene_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.gene_sig_networks = nn.ModuleList(sig_networks)
        
    def init_per_trans_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.trans_sig_networks = nn.ModuleList(sig_networks)

    def get_hyperedge(self, edge):
        
        adj_matrix = edge.cpu().numpy()

        hyperedges = defaultdict(set)
        
        for start, end in adj_matrix.T:
            hyperedges[start].add(end)

        hypergraph_edges = [list({start}.union(ends)) for start, ends in hyperedges.items()]

        return hypergraph_edges
