"""
    Implementation of Adam of Contact Map Guide Energy
"""
import collections
import time
from scipy.spatial.transform import Rotation as R
import torch
from utils_model.HandModel import get_handmodel
import torch.nn.functional as F
from utils_model.diffcontact import *
import pytorch3d
from utils_model.HandModel import ERF_loss, SPF_loss, SRF_loss
from utils.my_utils import *
torch.set_printoptions(profile="full")

class CMapAdam:
    def __init__(self, robot_name,
                 num_particles=32, init_rand_scale=0.5,
                 learning_rate=5e-3, running_name=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu', verbose_energy=False):
        if robot_name == 'shadowhand':
            self.n_robot_part = 5 * 3 + 1
            self.n_human_part = 16
        elif robot_name == 'allegro' or 'leaphand':
            self.n_robot_part = 4 * 3 + 1
            self.n_human_part = 16
        elif robot_name == 'barrett':
            self.n_robot_part = 3 * 2 + 1
            self.n_human_part = 11
        elif robot_name == 'robotiq_3finger':
            self.n_robot_part = 3 * 3 + 1
            self.n_human_part = 16
        self.running_name = running_name
        self.device = device
        self.robot_name = robot_name
        self.num_particles = num_particles
        self.init_random_scale = init_rand_scale
        self.learning_rate = learning_rate

        self.verbose_energy = verbose_energy

        self.global_step = None
        self.q_current = None
        self.energy = None
        self.contact_loss = None

        self.contact_value_goal = None
        self.object_point_cloud = None
        self.object_normal_cloud = None

        self.q_global = None
        self.q_local = None
        self.optimizer = None

        self.handmodel = get_handmodel(robot_name, num_particles, device, hand_scale=1.)
        self.q_joint_lower = self.handmodel.revolute_joints_q_lower.detach()
        self.q_joint_upper = self.handmodel.revolute_joints_q_upper.detach()
        
    def reset(self, contact_map_goal, contact_part, running_name):
        self.running_name = running_name
        self.global_step = 0
        self.object_point_cloud = contact_map_goal[:, :3].to(self.device)
        self.object_verts = self.object_point_cloud.unsqueeze(0).repeat(self.num_particles, 1, 1)
        self.object_normal_cloud = contact_map_goal[:, 3:6].to(self.device).unsqueeze(0).repeat(self.num_particles, 1, 1)
        self.contact_value_goal = contact_map_goal[:, 6].to(self.device)
        labels = contact_part.argmax(dim=1)
        if self.n_human_part != 16:
            labels = remap_part_labels(labels, self.n_human_part)
        contact_part = torch.nn.functional.one_hot(labels, num_classes=self.n_human_part).to(self.device)
        contact_map_goal = self.contact_value_goal.unsqueeze(0).unsqueeze(2).repeat(self.num_particles, 1, self.n_human_part) # [B, 2048, 16]
        contact_map_goal = contact_map_goal * contact_part.unsqueeze(0).repeat(self.num_particles, 1, 1) # [2048, 16]
        remapped_contact_map = remap_contact_map(contact_map_goal[0].cpu().numpy(), self.object_point_cloud.cpu().numpy(), self.n_robot_part)
        remapped_contact_map = torch.from_numpy(remapped_contact_map).cuda().unsqueeze(0).repeat(self.num_particles, 1, 1)
        self.contact_labels = torch.argmax(remapped_contact_map, dim=-1)
        
        object_radius = torch.max(torch.norm(self.object_point_cloud, dim=1, p=2))
        self.q_current = torch.zeros(self.num_particles, 3 + 6 + len(self.handmodel.revolute_joints),
                                     device=self.device)
        
        # random initial poses 
        # TODO: You can adjust initial poses to get more stable or more diverse grasp results
        object_center = torch.mean(self.object_point_cloud, dim=0)  # [3]
        cube_size = object_radius * 1.5
        random_offsets = (torch.rand(self.num_particles, 3, device=self.device) - 0.5) * cube_size
        self.q_current[:, :3] = object_center + random_offsets

        new_rotation_matrices = []
        for i in range(self.num_particles):
            hand_normal = object_center - self.q_current[i, :3]
            hand_normal = hand_normal / (torch.norm(hand_normal) + 1e-10)
            
            rotation_matrix = compute_rotation_matrix_from_hand_normal(hand_normal)
            
            angle = torch.tensor(torch.pi / 2)
            if self.robot_name == 'shadowhand':
                rotation_matrix = rotate_around_local_x(rotation_matrix, angle)
            elif self.robot_name == 'allegro':
                rotation_matrix = rotate_around_local_y(rotation_matrix, angle)
            elif self.robot_name == 'barrett' or self.robot_name == 'robotiq_3finger':
                rotation_matrix = rotation_matrix 
            elif self.robot_name == 'leaphand':
                rotation_matrix = rotate_around_local_x(rotation_matrix, angle)
                rotation_matrix = rotate_around_local_x(rotation_matrix, angle)
            
            random_angle = torch.rand(1, device=self.device) * 2 * torch.pi
            if self.robot_name == 'allegro':
                rotation_matrix = rotate_around_local_x(rotation_matrix, random_angle)
            elif self.robot_name == 'shadowhand':
                rotation_matrix = rotate_around_local_y(rotation_matrix, random_angle)
            elif self.robot_name == 'barrett' or self.robot_name == 'robotiq_3finger' or self.robot_name == 'leaphand':
                rotation_matrix = rotate_around_local_z(rotation_matrix, random_angle)
            
            new_rotation_matrices.append(rotation_matrix)

        new_rotation_matrices = torch.stack(new_rotation_matrices)
        self.q_current[:, 3:9] = new_rotation_matrices.reshape(self.num_particles, 9)[:, :6]
        self.q_current[:, 9:] = self.init_random_scale * torch.rand_like(self.q_current[:, 9:]) * (self.q_joint_upper - self.q_joint_lower) + self.q_joint_lower
        self.q_current.requires_grad = True
        # Utils: Visualization hand and object: Set args.num_particles to 1!
        # visualize_point_clouds(self.handmodel.get_surface_points_and_normals(self.q_current)[0].cpu().detach().squeeze(0), self.object_verts.cpu().detach().squeeze(0))
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)

    def compute_energy_sdf_capsule(self, penetration_only=False):  
        hand_surface_points, hand_normals, part_labels = self.handmodel.get_surface_points_and_normals() 

        # robot constraint
        z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(self.q_joint_lower - self.q_current[:, 9:])  

        # physical contraint
        obj_pcd_nor = torch.cat((self.object_verts, self.object_normal_cloud), dim=-1).to(dtype=torch.float32)
        ERF_loss_value = ERF_loss(obj_pcd_nor, hand_surface_points)
        dis_keypoint = self.handmodel.get_dis_keypoints(q= self.q_current)
        SPF_loss_value = SPF_loss(dis_keypoint, self.object_verts, thres_dis=1)
        hand_keypoint = self.handmodel.get_keypoints(q= self.q_current)
        SRF_loss_value = SRF_loss(hand_keypoint)

        # TODO: You can adjust hyperparameter here to get more stable grasp results or grasp results with less penetration
        if penetration_only:
            energy = 1000 * ERF_loss_value + 1000 * SRF_loss_value + 10000 * SPF_loss_value
            if self.robot_name == 'robotiq_3finger':  
                self.energy = energy  
            else:  
                self.energy = energy + z_norm.sum(dim=1)  
            return energy
        
        # TODO: You can adjust hyperparameter here to get more stable grasp results or grasp results with less penetration
        physical_loss = 1000 * ERF_loss_value \
            + 1000 * SRF_loss_value \
            + 1000 * SPF_loss_value
        
        sdf_obj_part, _ = capsule_sdf_per_joint(self.object_verts, self.object_normal_cloud, # TODO
                                                        hand_surface_points, hand_normals, 
                                                        part_labels, 1, 0.0005, -0.0015, self.robot_name)
        pred_p = torch.gather(sdf_obj_part, dim=2, index=self.contact_labels.unsqueeze(dim=-1)).squeeze(-1)
        contact_loss = (torch.abs(pred_p) * self.contact_value_goal).sum(dim=-1)

        energy = contact_loss + + physical_loss

        if self.robot_name == 'robotiq_3finger':  
            self.energy = energy  
        else:  
            self.energy = energy + z_norm.sum(dim=1)  

        return energy

    def step(self, penetration_only=False):
        if penetration_only:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-3
        self.optimizer.zero_grad()
        self.handmodel.update_kinematics(q=self.q_current)
        energy = self.compute_energy_sdf_capsule(penetration_only)
        # energy = self.compute_energy_sdf_capsule(penetration_only=False)
        energy.mean().backward()
        self.optimizer.step()

        self.global_step += 1

    def get_opt_q(self):
        return self.q_current.detach()

    def set_opt_q(self, opt_q):
        self.q_current.copy_(opt_q.detach().to(self.device))

    def get_plotly_data(self, index=0, color='pink', opacity=0.7):
        return self.handmodel.get_plotly_data(q=self.q_current, i=index, color=color, opacity=opacity)