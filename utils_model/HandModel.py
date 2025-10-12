import json
import os
import pytorch_kinematics as pk
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)
from utils.rot6d import *
import trimesh.sample

class HandModel:
    def __init__(self, robot_name, urdf_filename, mesh_path,
                 batch_size=1, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 hand_scale=1.
                 ):
        self.device = device
        self.robot_name = robot_name
        self.batch_size = batch_size
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)

        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}
        
        self.canon_verts = []
        self.canon_faces = []
        self.idx_vert_faces = []
        self.face_normals = []

        removed_links = json.load(open(os.path.join("data", 'urdf/removed_links.json')))[robot_name]

        for i_link, link in enumerate(visual.links):
            if link.name in removed_links:
                continue
            if len(link.visuals) == 0:
                continue
            if type(link.visuals[0].geometry) == Mesh:
                if robot_name == 'shadowhand' or robot_name == 'allegro' or robot_name == 'barrett':
                    filename = link.visuals[0].geometry.filename.split('/')[-1]
                elif robot_name == 'allegro':
                    filename = f"{link.visuals[0].geometry.filename.split('/')[-2]}/{link.visuals[0].geometry.filename.split('/')[-1]}"
                else:
                    filename = link.visuals[0].geometry.filename
                mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
            elif type(link.visuals[0].geometry) == Cylinder:
                mesh = tm.primitives.Cylinder(
                    radius=link.visuals[0].geometry.radius, height=link.visuals[0].geometry.length)
            elif type(link.visuals[0].geometry) == Box:
                mesh = tm.primitives.Box(extents=link.visuals[0].geometry.size)
            elif type(link.visuals[0].geometry) == Sphere:
                mesh = tm.primitives.Sphere(
                    radius=link.visuals[0].geometry.radius)
            else:
                print(type(link.visuals[0].geometry))
                raise NotImplementedError
            try:
                scale = np.array(
                    link.visuals[0].geometry.scale).reshape([1, 3])
            except:
                scale = np.array([[1, 1, 1]])
            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                # print('---')
                # print(link.visuals[0].origin.rpy, rotation)
                # print('---')
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])
                
            if self.robot_name == 'shadowhand':
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=64)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)
            else:
                pts, pts_face_index = trimesh.sample.sample_surface(mesh=mesh, count=128)
                pts_normal = np.array([mesh.face_normals[x] for x in pts_face_index], dtype=float)

            if self.robot_name == 'barrett':
                if link.name in ['bh_base_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'ezgripper':
                if link.name in ['left_ezgripper_palm_link']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[1., 0., 0.] for x in range(pts.shape[0])], dtype=float)
            if self.robot_name == 'robotiq_3finger':
                if link.name in ['gripper_palm']:
                    pts = trimesh.sample.volume_mesh(mesh=mesh, count=1024)
                    pts_normal = np.array([[0., 0., 1.] for x in range(pts.shape[0])], dtype=float)

            pts *= scale
            # pts = mesh.sample(128) * scale
            # print(link.name, len(pts))
            # new
            if robot_name == 'shadowhand':
                pts = pts[:, [0, 2, 1]]
                pts_normal = pts_normal[:, [0, 2, 1]]
                pts[:, 1] *= -1
                pts_normal[:, 1] *= -1

            pts = np.matmul(rotation, pts.T).T + translation
            pts_normal = np.matmul(rotation, pts_normal.T).T
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate([pts_normal, np.ones([len(pts_normal), 1])], axis=-1)
            self.surface_points[link.name] = torch.from_numpy(pts).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)
            self.surface_points_normal[link.name] = torch.from_numpy(pts_normal).to(
                device).float().unsqueeze(0).repeat(batch_size, 1, 1)

            # visualization mesh
            self.mesh_verts[link.name] = np.array(mesh.vertices) * scale
            if robot_name == 'shadowhand':
                self.mesh_verts[link.name] = self.mesh_verts[link.name][:, [0, 2, 1]]
                self.mesh_verts[link.name][:, 1] *= -1
            self.mesh_verts[link.name] = np.matmul(rotation, self.mesh_verts[link.name].T).T + translation
            self.mesh_faces[link.name] = np.array(mesh.faces)

        self.joint_angles = len(self.robot.get_joint_parameter_names())

        self.current_status = None

        self.scale = hand_scale

    def update_kinematics(self, q):
        if q.shape[1] < self.joint_angles + 9:
            self.global_translation = q[:, :3]
            self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(torch.tensor([1., 0., 0., 0., 1., 0.], device='cuda').view(1, 6).repeat(q.shape[0], 1))
            self.current_status = self.robot.forward_kinematics(q[:, 3:])
        else:
            self.global_translation = q[:, :3]
            self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:,3:9])
            self.current_status = self.robot.forward_kinematics(q[:,9:])

    def get_surface_points(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3])
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale
    
    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data
    
    def get_plotly_data(self, q=None, i=0, color='lightblue', opacity=1.):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(),
                                      transformed_v.T).T + np.expand_dims(
                self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(x=transformed_v[:, 0], y=transformed_v[:, 1], z=transformed_v[:, 2], i=f[:, 0], j=f[:, 1],
                          k=f[:, 2], color=color, opacity=opacity))
        return data

def get_handmodel(robot, batch_size, device, hand_scale=1.):
    urdf_assets_meta = json.load(open("data/urdf/urdf_assets_meta.json"))
    urdf_path = urdf_assets_meta['urdf_path'][robot]
    meshes_path = urdf_assets_meta['meshes_path'][robot]
    hand_model = HandModel(robot, urdf_path, meshes_path, batch_size=batch_size, device=device, hand_scale=hand_scale)
    return hand_model