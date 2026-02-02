import json
import os
from tabnanny import check

import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
import torch
import torch.nn
import transforms3d
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
from utils.rot6d import *
import trimesh.sample
# from kaolin.metrics.trianglemesh import point_to_mesh_distance
# from kaolin.ops.mesh import check_sign, index_vertices_by_faces, face_normals
import pytorch3d
from pytorch3d.ops import knn_points

import numpy as np
from scipy.spatial import cKDTree

from mpl_toolkits.mplot3d import Axes3D

def rotate_around_local_x(rotation_matrix, angle):
    c = torch.cos(angle)
    s = torch.sin(angle)

    rotate_x_matrix = torch.tensor([[1, 0, 0],
                                     [0, c, -s],
                                     [0, s, c]]).cuda()

    new_rotation_matrix = rotate_x_matrix @ rotation_matrix
    return new_rotation_matrix

def rotate_around_local_y(rotation_matrix, angle):
    c = torch.cos(angle)
    s = torch.sin(angle)

    rotate_y_matrix = torch.tensor([[c, 0, s],
                                     [0, 1, 0],
                                     [-s, 0, c]]).cuda()

    new_rotation_matrix = rotate_y_matrix @ rotation_matrix
    return new_rotation_matrix

def rotate_around_local_z(rotation_matrix, angle):
    c = torch.cos(angle)
    s = torch.sin(angle)

    rotate_z_matrix = torch.tensor([[c, -s, 0],
                                     [s, c, 0],
                                     [0, 0, 1]]).cuda()

    new_rotation_matrix = rotate_z_matrix @ rotation_matrix
    return new_rotation_matrix

def compute_rotation_matrix_from_hand_normal(hand_normal):
    z_axis = hand_normal / (hand_normal.norm() + 1e-10)

    x_axis = torch.tensor([1.0, 0.0, 0.0]).cuda()
    if z_axis[0] == 1.0:
        y_axis = torch.tensor([0.0, 1.0, 0.0]).cuda()
    else:
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / (y_axis.norm() + 1e-10)
        x_axis = torch.cross(y_axis, z_axis)

    rotation_matrix = torch.stack([x_axis, y_axis, z_axis])
    return rotation_matrix

def visualize_point_clouds(hand_pcd, object_point_cloud):
    hand_pcd_np = hand_pcd.numpy()
    object_point_cloud_np = object_point_cloud.numpy()

    all_points = np.vstack([hand_pcd_np, object_point_cloud_np])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(hand_pcd_np[:, 0], hand_pcd_np[:, 1], hand_pcd_np[:, 2], 
               color='blue', label='Hand PCD', alpha=0.6, s=20)
    
    ax.scatter(object_point_cloud_np[:, 0], object_point_cloud_np[:, 1], object_point_cloud_np[:, 2], 
               color='red', label='Object Point Cloud', alpha=0.6, s=20)

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    max_range = max(x_range, y_range, z_range)
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
    ax.set_box_aspect([1,1,1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('hand and object PCD viz')

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

def mass_centroid(points, weights):  
    if len(points) == 0:  
        return np.zeros(3), 0  
    total_mass = np.sum(weights)  
    centroid = np.sum(points * weights[:, np.newaxis], axis=0) / total_mass  
    return centroid, total_mass  

def remap_single_channel(contact_value, object_point_cloud, channel_a, channel_b, union_only=False):
    """
    Remap two specific channels of the contact map
    
    Args:
        contact_value (np.ndarray): Original contact value, shape (2048, 6)
        object_point_cloud (np.ndarray): Original object point cloud, shape (2048, 3)
        channel_a (int): First channel to remap
        channel_b (int): Second channel to remap
    
    Returns:
        np.ndarray: Remapped contact value for the two channels, shape (2048)
    """
    # 1. Calculate the centroid of the original point cloud
    Mo = np.mean(object_point_cloud, axis=0)

    # 2. Find points with non-zero contact values for both channels
    Pa = np.where(contact_value[:, channel_a] > 0)[0]
    Pb = np.where(contact_value[:, channel_b] > 0)[0]

    if len(Pa) == 0 and len(Pb) == 0:
        # Both clusters are empty
        if union_only:
            return np.array([])
        return np.zeros(object_point_cloud.shape[0])
    elif len(Pa) == 0:
        # Only channel_a is empty, return channel_b values
        if union_only:
            return Pb
        remapped_contact_value = np.zeros(object_point_cloud.shape[0])
        remapped_contact_value[Pb] = contact_value[Pb, channel_b]
        return remapped_contact_value
    elif len(Pb) == 0:
        # Only channel_b is empty, return channel_a values
        if union_only:
            return Pa
        remapped_contact_value = np.zeros(object_point_cloud.shape[0])
        remapped_contact_value[Pa] = contact_value[Pa, channel_a]
        return remapped_contact_value

    # Calculate centroids and total masses
    Ma, mass_a = mass_centroid(object_point_cloud[Pa], contact_value[Pa, channel_a])
    Mb, mass_b = mass_centroid(object_point_cloud[Pb], contact_value[Pb, channel_b])

    # Mapping Pb to a new point set Pb'
    remapped_Pb = np.zeros_like(Pb)
    for i in range(len(Pb)):
        pb = object_point_cloud[Pb[i]]

        vector_Mo_pb = pb - Mo
        vector_Mo_Ma = Ma - Mo
        len_Mo_pb = np.linalg.norm(vector_Mo_pb)
        len_Mo_Ma = np.linalg.norm(vector_Mo_Ma)
        unit_vector_Mo_pb = vector_Mo_pb / len_Mo_pb
        unit_vector_Mo_Ma = vector_Mo_Ma / len_Mo_Ma
        bisector_vector = unit_vector_Mo_pb * mass_b + unit_vector_Mo_Ma * mass_a
        bisector_vector /= np.linalg.norm(bisector_vector)
        tmp = Mo + bisector_vector * (len_Mo_pb + len_Mo_Ma)

        # Find the closest point in the original point cloud
        kdtree = cKDTree(object_point_cloud)
        nearest_idx = kdtree.query(tmp)[1]
        remapped_Pb[i] = nearest_idx

    # Now do the reverse for Pa
    remapped_Pa = np.zeros_like(Pa)
    for i in range(len(Pa)):
        pa = object_point_cloud[Pa[i]]
        vector_Mo_pa = pa - Mo
        vector_Mo_Mb = Mb - Mo
        len_Mo_pa = np.linalg.norm(vector_Mo_pa)
        len_Mo_Mb = np.linalg.norm(vector_Mo_Mb)
        unit_vector_Mo_pa = vector_Mo_pa / len_Mo_pa
        unit_vector_Mo_Mb = vector_Mo_Mb / len_Mo_Mb
        bisector_vector = unit_vector_Mo_pa * mass_a + unit_vector_Mo_Mb * mass_b
        bisector_vector /= np.linalg.norm(bisector_vector)
        tmp = Mo + bisector_vector * (len_Mo_pa + len_Mo_Mb)
        kdtree = cKDTree(object_point_cloud)
        nearest_idx = kdtree.query(tmp)[1]
        remapped_Pa[i] = nearest_idx

    # Find the union of remapped points Pa' and Pb'
    unique_Pa = set(remapped_Pa)
    unique_Pb = set(remapped_Pb)
    union = unique_Pa.union(unique_Pb)
    if union_only:
        return np.array(list(union))

    # Create remapped contact value
    remapped_contact_value = np.zeros(object_point_cloud.shape[0])
    
    # Combine the two remapped sets
    for p in union:
        pa_indices = np.where(remapped_Pa == p)[0]
        pb_indices = np.where(remapped_Pb == p)[0]
        
        if len(pa_indices) > 0 and len(pb_indices) > 0:
            contact_value_Pa = contact_value[Pa[pa_indices[0]], channel_a]
            contact_value_Pb = contact_value[Pb[pb_indices[0]], channel_b]
            combined_value = np.clip(contact_value_Pa + contact_value_Pb, 0, 1)
            # combined_value = contact_value_Pa + contact_value_Pb
            remapped_contact_value[p] = combined_value
        
        elif len(pa_indices) > 0:
            remapped_contact_value[p] = contact_value[Pa[pa_indices[0]], channel_a]
            
        elif len(pb_indices) > 0:
            remapped_contact_value[p] = contact_value[Pb[pb_indices[0]], channel_b]

    return remapped_contact_value

def remap_contact_map(contact_value, object_point_cloud, n_robot_part):
    """
    Remaps the contact map channels from 5 fingers (including palm)
    to a new representation of 3 fingers and palm.

    Args:
        contact_value (np.ndarray): Original contact value, shape (2048, 6).
        object_point_cloud (np.ndarray): Original object point cloud, shape (2048, 3).

    Returns:
        np.ndarray: Remapped contact value, shape (2048, 4).
    """

    if n_robot_part == 16:
        return contact_value
    elif n_robot_part == 13:
        remapped_contact_value = np.zeros((object_point_cloud.shape[0], 4*3+1))
        remapped_contact_value[:, 0:10] = contact_value[:, 0:10]
        remapped_contact_value[:, 10] = remap_single_channel(contact_value, object_point_cloud, 10, 13)
        remapped_contact_value[:, 11] = remap_single_channel(contact_value, object_point_cloud, 11, 14)
        remapped_contact_value[:, 12] = remap_single_channel(contact_value, object_point_cloud, 12, 15)
    elif n_robot_part == 10:
        remapped_contact_value = np.zeros((object_point_cloud.shape[0], 10))
        remapped_contact_value[:, 0:4] = contact_value[:, 0:4]
        remapped_contact_value[:, 4] = remap_single_channel(contact_value, object_point_cloud, 4, 7)
        remapped_contact_value[:, 5] = remap_single_channel(contact_value, object_point_cloud, 5, 8)
        remapped_contact_value[:, 6] = remap_single_channel(contact_value, object_point_cloud, 6, 9)
        remapped_contact_value[:, 7] = remap_single_channel(contact_value, object_point_cloud, 10, 13)
        remapped_contact_value[:, 8] = remap_single_channel(contact_value, object_point_cloud, 11, 14)
        remapped_contact_value[:, 9] = remap_single_channel(contact_value, object_point_cloud, 12, 15)
    elif n_robot_part == 7:
        remapped_contact_value = np.zeros((object_point_cloud.shape[0], 7))
        remapped_contact_value[:, 0:3] = contact_value[:, 0:3]
        remapped_contact_value[:, 3] = remap_single_channel(contact_value, object_point_cloud, 3, 5)
        remapped_contact_value[:, 4] = remap_single_channel(contact_value, object_point_cloud, 4, 6)
        remapped_contact_value[:, 5] = remap_single_channel(contact_value, object_point_cloud, 7, 9)
        remapped_contact_value[:, 6] = remap_single_channel(contact_value, object_point_cloud, 8, 10)

    return remapped_contact_value

def remap_part_labels(part_labels, n_part=11):
    if n_part == 6:
        remap_dict = {
            0: 0, # palm
            1: 2, # index
            2: 2,
            3: 2,
            4: 3, # middle finger
            5: 3,
            6: 3,
            7: 5, # pinky finger
            8: 5,
            9: 5, 
            10: 4, # ring finger
            11: 4,
            12: 4,
            13: 0, # thumb
            14: 1,
            15: 1,
        }
    elif n_part == 11:
        remap_dict = {
            0: 0, # palm
            1: 3, # index
            2: 4,
            3: 4,
            4: 5, # middle finger
            5: 6,
            6: 6,
            7: 9, # pinky finger
            8: 10,
            9: 10, 
            10: 7, # ring finger
            11: 8,
            12: 8,
            13: 0, # thumb
            14: 1,
            15: 2,
        }
    else:
        remap_dict = {
            0: 0, # palm
            1: 4, # index
            2: 5,
            3: 6,
            4: 7, # middle finger
            5: 8,
            6: 9,
            7: 13, # pinky finger
            8: 14,
            9: 15, 
            10: 10, # ring finger
            11: 11,
            12: 12,
            13: 1, # thumb
            14: 2,
            15: 3,
        }
    remapped_labels = torch.zeros_like(part_labels)
    
    for original_id, new_id in remap_dict.items():
        remapped_labels[part_labels == original_id] = new_id
    
    return remapped_labels

def visualize_hand_parts(hand_verts, hand_part_label, highlight_part=0, save_path=None):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if hand_verts.dim() == 3:  # [B, 778, 3]
        hand_verts = hand_verts.squeeze(0)  # [778, 3]
    
    if isinstance(hand_verts, torch.Tensor):
        hand_verts_np = hand_verts.detach().cpu().numpy()
    else:
        hand_verts_np = hand_verts
        
    if isinstance(hand_part_label, torch.Tensor):
        hand_part_label_np = hand_part_label.detach().cpu().numpy()
    else:
        hand_part_label_np = hand_part_label
    
    colors = np.zeros((len(hand_part_label_np), 3))
    colors[hand_part_label_np == highlight_part] = [1, 0, 0]  # red
    colors[hand_part_label_np != highlight_part] = [0, 0, 0]  # black
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hand_verts_np[:, 0], hand_verts_np[:, 1], hand_verts_np[:, 2], 
               c=colors, s=10)
    
    ax.set_title(f"Hand Visualization (Red: Part {highlight_part})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def visualize_hand_surface_and_keypoints(surface_points, keypoints, title=None):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    surface_points_np = surface_points.squeeze(0).detach().cpu().numpy()
    keypoints_np = keypoints.squeeze(0).detach().cpu().numpy()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(surface_points_np[:, 0], surface_points_np[:, 1], surface_points_np[:, 2], 
               c='gray', s=5, alpha=0.3, label='Hand Surface')
    
    ax.scatter(keypoints_np[:, 0], keypoints_np[:, 1], keypoints_np[:, 2], 
               c='blue', s=100, marker='o', label='Global Keypoints')
    
    ax.set_title(title or "Hand Surface and Global Keypoints")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.legend()
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax
