'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-03-28 10:46:38
Email: haimingzhang@link.cuhk.edu.cn
Description: Validate the flow transformation.
'''
import numpy as np
import pickle
import os
import os.path as osp
import open3d as o3d
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import shutil
from pyquaternion import Quaternion


def occupancy2pointcloud(occupancy,
                         pc_range=[-40.0, -40.0,  -1.0, 40.0, 40.0, 5.4],
                         voxel_size=0.4):
    occupancy[occupancy == 17] = 0
    fov_voxels = np.stack(occupancy.nonzero())  # (3, N)
    fov_voxels = fov_voxels.transpose((1, 0))  # to (N, 3)

    fov_voxels = fov_voxels.astype(np.float32)

    fov_voxels[:, :3] = (fov_voxels[:, :3].astype(np.float32) + 0.5) * voxel_size
    fov_voxels[:, 0] += pc_range[0]
    fov_voxels[:, 1] += pc_range[1]
    fov_voxels[:, 2] += pc_range[2]
    return fov_voxels


def load_occupancy(data_dir, scene_name, token):
    label_file = os.path.join(data_dir, f'{scene_name}/{token}/labels.npz')
    save_dir = f"temp/{scene_name}/{token}"
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(label_file, osp.join(save_dir, f'labels.npz'))

    label = np.load(label_file)
    occ = label['semantics']
    return occ


def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat


def get_ego2global_trans(nusc_infos, scene_name, frame_idx):
    # get the global to ego transformation matrix
    _info = nusc_infos[scene_name][frame_idx]
    ego_to_global = rt2mat(_info['ego2global_translation'],
                           _info['ego2global_rotation'])
    global_to_ego = np.linalg.inv(ego_to_global)
    
    return ego_to_global, global_to_ego


def save_pointcloud(save_path, vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_point_cloud(save_path, pcd)


def main(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    data_infos = data['infos']
    
    scene_name = 'scene-0269'
    scene = data_infos[scene_name]

    frame_idx_1 = 10
    frame_idx_2 = 11
    occ_data_dir = "occupancy"

    occ1 = load_occupancy(occ_data_dir, scene_name, scene[frame_idx_1]['token'])
    occ2 = load_occupancy(occ_data_dir, scene_name, scene[frame_idx_2]['token'])

    occ_pc1 = occupancy2pointcloud(occ1)
    occ_pc2 = occupancy2pointcloud(occ2)

    # to homogeneous coordinates
    occ_pc1 = np.concatenate([occ_pc1, np.ones((occ_pc1.shape[0], 1))], axis=1)
    occ_pc2 = np.concatenate([occ_pc2, np.ones((occ_pc2.shape[0], 1))], axis=1)

    ego2global_1, global2ego_1 = get_ego2global_trans(data_infos, scene_name, frame_idx_1)
    ego2global_2, global2ego_2 = get_ego2global_trans(data_infos, scene_name, frame_idx_2)
    
    warped_occ = warp_bev_features(occ1, 
                      voxel_flow=None, 
                      voxel_size=torch.Tensor([0.4, 0.4]), 
                      occ_size=torch.Tensor([200.0, 200.0]),
                      extrinsic_matrix=global2ego_1)
    warped_occ_pc = occupancy2pointcloud(warped_occ)
    save_pointcloud("warped_occ_pc1.ply", warped_occ_pc[:, :3])
    
    ## tranform the occupancy 2 to occupancy 1
    # 1) transform the occupancy 2 to global
    occ_pc2_global = np.matmul(occ_pc2, ego2global_2.T)
    # 2) transform the global occupancy 2 to the occupancy 1
    occ_pc2_1 = np.matmul(occ_pc2_global, global2ego_1.T)

    save_pointcloud("occ_pc1.ply", occ_pc1[:, :3])
    save_pointcloud("occ_pc2.ply", occ_pc2[:, :3])
    save_pointcloud("occ_pc2_1.ply", occ_pc2_1[:, :3])
    

def warp_bev_features(voxel_feats, 
                      voxel_flow,
                      voxel_size, 
                      occ_size,
                      extrinsic_matrix=None):
    """Warp the given voxel features using the predicted voxel flow.

    Args:
        voxel_feats (Tensor): _description_
        voxel_flow (Tensor): (bs, f, H, W, 2)
        voxel_size (Tensor): the voxel size for each voxel, for example torch.Tensor([0.4, 0.4])
        occ_size (Tensor): the size of the occupancy map, for example torch.Tensor([200, 200])
        extrinsic_matrix (_type_, optional): global to ego transformation matrix. Defaults to None.

    Returns:
        _type_: _description_
    """
    voxel_feats = torch.from_numpy(voxel_feats).permute(2, 0, 1)[None]
    voxel_flow = torch.zeros((1, 1, 200, 200, 2)).to(voxel_feats.device)

    extrinsic_matrix = torch.from_numpy(extrinsic_matrix)[None].to(voxel_feats.device)
    extrinsic_matrix = extrinsic_matrix.to(torch.float32)

    device = voxel_feats.device
    bs, num_pred, x_size, y_size, c = voxel_flow.shape

    if extrinsic_matrix is not None:
        for i in range(bs):
            _extrinsic_matrix = extrinsic_matrix[i]  # to (1, 4, 4)
            _voxel_flow = voxel_flow[i].reshape(num_pred, -1, 2)
            ## padding the zero flow for z axis
            _voxel_flow = torch.cat([_voxel_flow, torch.zeros(num_pred, _voxel_flow.shape[1], 1).to(device)], dim=-1)
            trans_flow = torch.matmul(_extrinsic_matrix[:3, :3][None], _voxel_flow.permute(0, 2, 1))
            # trans_flow = trans_flow + _extrinsic_matrix[:3, 3][None, :, None]
            trans_flow = trans_flow.permute(0, 2, 1)[..., :2]
            voxel_flow[i] = trans_flow.reshape(num_pred, *voxel_flow.shape[2:])

    voxel_flow = rearrange(voxel_flow, 'b f h w dim2 -> (b f) h w dim2')
    new_bs = voxel_flow.shape[0]

    # normalize the flow in m/s unit to voxel unit and then to [-1, 1]
    voxel_size = voxel_size.to(device)
    occ_size = occ_size.to(device)

    voxel_flow = 2 * (voxel_flow / voxel_size / occ_size)

    # generate normalized grid
    x = torch.linspace(-1.0, 1.0, x_size).view(-1, 1).repeat(1, y_size).to(device)
    y = torch.linspace(-1.0, 1.0, y_size).view(1, -1).repeat(x_size, 1).to(device)
    grid = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)  # (h, w, 2)
    
    # add flow to grid
    grid = grid.unsqueeze(0).expand(new_bs, -1, -1, -1).flip(-1) + voxel_flow

    # perform the voxel feature warping
    voxel_feats = torch.repeat_interleave(voxel_feats, num_pred, dim=0)
    warped_voxel_feats = F.grid_sample(voxel_feats.float(), 
                                       grid.float(), 
                                       mode='nearest', 
                                       padding_mode='border')
    warped_voxel_feats = rearrange(warped_voxel_feats, '(b f) c h w -> b f c h w', b=bs)
    warp_bev_features = warped_voxel_feats.squeeze().permute(1, 2, 0).numpy()
    return warp_bev_features


if __name__ == "__main__":

    pkl_path = "data/nuscenes_infos_val_temporal_v3_scene.pkl"
    main(pkl_path)
