import os
import json
import numpy as np
import trimesh

class TestSetYCB:
    def __init__(self, 
                 obj_root='../data/object/ycb',
                 mesh_relpath='google_16k/nontextured.ply',
                 n_samples=2048,
                 object_list_path='../data/objects.json'):
        self.obj_root = obj_root
        self.mesh_relpath = mesh_relpath
        self.n_samples = n_samples

        with open(object_list_path, 'r') as f:
            object_data = json.load(f)
        
        ycb_objects = object_data.get('YCB', [])
        
        self.object_list = []
        for obj_name in ycb_objects:
            mesh_path = os.path.join(obj_root, obj_name, mesh_relpath)
            if os.path.exists(mesh_path):
                self.object_list.append((obj_name, mesh_path))
            else:
                print(f"Warning: Mesh not found for {obj_name} at {mesh_path}")

        print(f"Found {len(self.object_list)} objects from YCB dataset")
        
    def __len__(self):
        return len(self.object_list)

    def __getitem__(self, item):
        obj_name, obj_mesh_path = self.object_list[item]
        obj_mesh = trimesh.load(obj_mesh_path)
        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        obj_verts = sample[0].astype(np.float32)
        obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)
        return {
            "obj_name": obj_name,
            "obj_verts": obj_verts,
            "obj_vn": obj_vn
        }