from utils_model.CMapAdam import CMapAdam
import torch
from tqdm import tqdm
import time

class AdamGrasp:
    def __init__(self, robot_name,
                 num_particles=32, init_rand_scale=0.5, max_iter=300, steps_per_iter=2,
                 learning_rate=5e-3, device='cuda'):
        self.robot_name = robot_name
        self.max_iter = max_iter
        self.steps_per_iter = steps_per_iter

        self.opt_model = CMapAdam(robot_name=robot_name, num_particles=num_particles,
                                  init_rand_scale=init_rand_scale, learning_rate=learning_rate,
                                  device=device)

    def run_adam(self, contact_map_goal, contact_part, running_name):
        self.opt_model.reset(contact_map_goal, contact_part, running_name)

        # Optimization loop
        for i_iter in tqdm(range(self.max_iter), desc=f'{running_name}'):
        # for i_iter in range(self.max_iter):

            if i_iter < self.max_iter - 50:
                self.opt_model.step()
            else:
                # if self.robot_name == 'shadowhand':
                self.opt_model.step(penetration_only=True)
                # else: 
                # self.opt_model.step(penetration_only=False)

            # if i_iter % 25 == 0 or i_iter == self.max_iter - 1:
            #     min_energy = self.opt_model.energy.min(dim=0)[0]
            #     print(f'min energy: {min_energy:.4f}')
        
        return self.opt_model.get_opt_q(), self.opt_model.energy.detach().cpu().clone()