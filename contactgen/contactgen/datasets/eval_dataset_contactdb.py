import os
import json
import numpy as np
import trimesh

class TestSetContactDB:
    def __init__(self, 
                 obj_root='../data/object/contactdb',
                 n_samples=2048,
                 object_list_path='../data/objects.json'):
        self.obj_root = obj_root
        self.n_samples = n_samples

        with open(object_list_path, 'r') as f:
            object_data = json.load(f)
        
        contactdb_objects = object_data.get('ContactDB', [])
        
        self.object_list = []
        for obj_name in contactdb_objects:
            mesh_path = os.path.join(obj_root, obj_name, f'{obj_name}.stl')
            if os.path.exists(mesh_path):
                self.object_list.append((obj_name, mesh_path))
            else:
                print(f"Warning: Mesh not found for {obj_name} at {mesh_path}")

        print(f"Found {len(self.object_list)} objects from ContactDB dataset")
        
    def __len__(self):
        return len(self.object_list)

    def __getitem__(self, item):
        obj_name, obj_mesh_path = self.object_list[item]
        obj_mesh = trimesh.load(obj_mesh_path)
        
        if isinstance(obj_mesh, trimesh.Scene):
            if len(obj_mesh.geometry) > 0:
                obj_mesh = list(obj_mesh.geometry.values())[0]
            else:
                raise ValueError(f"No geometry found in scene from {obj_mesh_path}")
        
        if not isinstance(obj_mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid mesh from {obj_mesh_path}")
            
        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        obj_verts = sample[0].astype(np.float32)
        obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)
        return {
            "obj_name": obj_name,
            "obj_verts": obj_verts,
            "obj_vn": obj_vn
        }