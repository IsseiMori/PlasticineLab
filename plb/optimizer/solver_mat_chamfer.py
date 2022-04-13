import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN
import os

import torch

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}

class RunSimulation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, env: TaichiEnv, action):
        ctx.env = env
        ctx.params = params
        ctx.action = action

        # Set material parameters in Taichi
        rescaled_params = params.clone()
        rescaled_params[0] = 5 + rescaled_params[0] * 195
        rescaled_params[1] = 100 + rescaled_params[1] * 2900
        rescaled_params[2] = 0 + rescaled_params[2] * 0.45
        env.simulator.material.from_torch(rescaled_params)
        
        env.simulator.set_material_params_kernel()

        # positions = torch.tensor(, requires_grad=True)
        p_pos_seq = torch.empty(len(action)-1, 5000, 3)

        for s in range(len(action)-1):
            env.simulator.step_kernel(action[s])

            for i in range(env.simulator.n_particles):
                for j in ti.static(range(env.simulator.dim)):
                    p_pos_seq[s, i, j] = env.simulator.x[0, i][j]


        # p_pos_seq = env.simulator.x.to_torch()

        return p_pos_seq


    @staticmethod
    def backward(ctx, grad_output):

        ti.clear_all_gradients()

        env = ctx.env
        params = ctx.params
        action = ctx.action

        # env.simulator.x.grad.from_torch(grad_output)

        grad = ti.field(dtype=ti.f32, shape=(len(action)-1, 5000, 3))
        grad.from_torch(grad_output)

        for s in reversed(range(len(action)-1)):
            
            for i in range(env.simulator.n_particles):
                for j in ti.static(range(3)):
                    env.simulator.x.grad[0, i][j] = grad[s, i, j]
            

            env.simulator.step_kernel_grad(action[s])

        params = env.simulator.material.to_torch()

        return params, None, None

class SolverMatChamfer:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def solve(self, init_actions=None, callbacks=()):
        env = self.env

        # initialize material parameters; YS, E, nu
        material_params = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        optimizer = torch.optim.Adam([material_params], lr=0.05)

        init_actions = self.init_actions(env, self.cfg)

        ppos_seq_target = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', env.cfg.loss.ppos_path))
        ppos_seq_target = ppos_seq_target[1:-1] # remove initial state
        ppos_seq_target = torch.tensor(ppos_seq_target)
        print(ppos_seq_target.shape)
        print(init_actions.shape)

        # initialize ...
        # optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        # optim = OPTIMS[self.optim_cfg.type](material_params, self.optim_cfg)


        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward(sim_state, action, material, ppos_seq_target):
            if self.logger is not None:
                self.logger.reset()
            
            env.set_state(sim_state, self.cfg.softness, False)
            
            p_pos_seq = RunSimulation.apply(material, env, action)
            print("ppos")
            print(p_pos_seq.shape)
            print(ppos_seq_target.shape)
            
            loss = ((ppos_seq_target - p_pos_seq)**2).mean()

            return loss

        # best_action = None
        best_material = None
        best_loss = 1e10

        actions = init_actions
        for iter in range(self.cfg.n_iters):
            # self.params = actions.copy() # not doing anything
            loss = forward(env_state['state'], actions, material_params, ppos_seq_target)
            print('material_params', material_params)
            print('loss ', loss)
            if loss < best_loss:
                best_loss = loss
                best_material = material_params.clone()
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            print(material_params)
            optimizer.step()
    

        env.set_state(**env_state)
        return best_material, actions


    def init_actions(self, env, cfg):
        action_dim = env.primitives.action_dim
        horizon = cfg.horizon
        # if cfg.init_sampler == 'uniform':
        #     return np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
        # else:
        #     raise NotImplementedError

        # Import and reshape the action sequence
        states = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', env.cfg.loss.action_path), allow_pickle=True).item()
        actions = states['shape_states'][0]
        actions = (states['shape_states'][0, 1:, :, 0:3] - states['shape_states'][0, 0:-1, :, 0:3]) * 100
        actions = actions.reshape(actions.shape[0], -1)

        print("target material", (states['YS'] - 5)/195, " ", (states['E']-100)/2900, " ", states['nu']/0.45)

        state = self.env.get_state()

        # x, v, F, C, p1, p2, p3 = state['state']
        states_xvfcp = state['state']
        n_grips = states['shape_states'].shape[2]

        shape_states_ = states['shape_states'][0][0]
    
        for i_grip in range(n_grips):
            states_xvfcp[4+i_grip][:3] = shape_states_[i_grip][0:3]
            states_xvfcp[4+i_grip][3:] = shape_states_[i_grip][6:10]

        new_state = {
            'state': states_xvfcp,
            'is_copy': state['is_copy'],
            'softness': state['softness'],
        }
        env.set_state(**new_state)

        return actions

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 38

        cfg.init_range = 0.
        cfg.init_sampler = 'uniform'
        return cfg


def solve_mat_chamfer(env, path, logger, args):
    import os, cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    env._max_episode_steps = 38 # overwrite frame count
    T = env._max_episode_steps



    solver = SolverMatChamfer(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    mat, actions = solver.solve()
    taichi_env.simulator.set_material(mat)

    for idx, act in enumerate(actions):
        env.step(act)
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])
