import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F

class MemoryBank:
    def __init__(self, path='memory', theta = 0.0):
        """
        初始化MemoryBank，指定存储路径
        :param path: HDF5文件存储目录路径
        """
        self.path = path
        self.theta = theta
    
    def clear(self):
        """
        清除存储目录中的所有内容
        """
        
        file_path = self.path
        if os.path.isfile(file_path):
            os.unlink(file_path)
    
    def save(self, memory_dict_list):
        """
        保存数据。对于 pathology_feature:
        - 如果长度超过4096，则取末尾4096行。
        - 如果不足4096，则通过复制的方式填充至4096行。
        """
        with h5py.File(self.path, 'w') as hf:
            for i, patient_data in enumerate(memory_dict_list):
                group = hf.create_group(f"patient_{i}")
            
                for key in ['pathology_feature', 'genomics_feature']:
                    data = patient_data[key]
                
                    # 仅对 pathology_feature 进行处理
                    if key == 'pathology_feature' and isinstance(data, torch.Tensor):
                        target_len = 4096
                        N, C = data.shape
                    
                        if N >= target_len:
                            # 1. 如果足够长，取末尾 target_len 行
                            data = data[-target_len:, :]
                        elif N > 0:
                            # 2. 如果不足，通过复制来填充
                            indices = torch.arange(target_len, device=data.device) % N
                            data = data[indices]
                        else:
                            # 3. 处理空张量的情况
                            data = torch.zeros(target_len, C, device=data.device, dtype=data.dtype)

                    # 转换并存储
                    if isinstance(data, torch.Tensor):
                        data = data.detach().cpu().numpy()
                    group.create_dataset(key, data=data)
    
    def retrievePathology(self, g_feat):
        """
        根据基因特征(g_feat)检索最相似的病理特征
        :param g_feat: 基因特征向量(tensor)
        :return: 最相似患者的病理特征(tensor)
        """
        best_similarity = -1
        best_pathology = None
        g_feat = g_feat.detach().cpu()

        if not isinstance(g_feat, torch.Tensor):
            g_feat = torch.tensor(g_feat)
        
        with h5py.File(self.path, 'r') as hf:
            for patient in hf:
                patient_data = hf[patient]
                # 修改点: 读取存储的基因特征，而不是原型
                stored_g_feat = torch.tensor(patient_data['genomics_feature'][:])
                        
                # 修改点: 计算输入特征和存储特征之间的相似度
                similarity = F.cosine_similarity(
                    g_feat.flatten().unsqueeze(0),
                    stored_g_feat.flatten().unsqueeze(0)
                ).item()
                        
                # 更新最佳匹配
                if similarity > best_similarity:
                    best_similarity = similarity
                    # 返回的是病理特征，这部分逻辑不变
                    best_pathology = torch.tensor(patient_data['pathology_feature'][:])
        
        return best_pathology.to('cuda')
    
    def retrieveGene(self, p_feat):
        """
        根据病理特征(p_feat)检索最相似的基因特征
        :param p_feat: 病理特征向量(tensor)
        :return: 最相似患者的基因特征(tensor)
        """
        best_similarity = -1
        best_gene = None
        p_feat = p_feat.detach().cpu()
        target_len = 4096
        N, C = p_feat.shape
                    
        if N >= 4096:
            p_feat = p_feat[-target_len:, :]
        elif N > 0:
            indices = torch.arange(target_len, device=p_feat.device) % N
            p_feat = p_feat[indices]
        
        with h5py.File(self.path, 'r') as hf:
            for patient in hf:
                patient_data = hf[patient]
                # 修改点: 读取存储的病理特征，而不是原型
                stored_p_feat = torch.tensor(patient_data['pathology_feature'][:])
                        
                # 修改点: 计算输入特征和存储特征之间的相似度
                similarity = F.cosine_similarity(
                    p_feat.flatten().unsqueeze(0),
                    stored_p_feat.flatten().unsqueeze(0)
                ).item()
                        
                # 更新最佳匹配
                if similarity > best_similarity:
                    best_similarity = similarity
                    # 返回的是基因特征，这部分逻辑不变
                    best_gene = torch.tensor(patient_data['genomics_feature'][:])
        
        return best_gene.to('cuda')