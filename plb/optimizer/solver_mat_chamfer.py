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
        # rescaled_params = torch.tensor(params, requires_grad=True)

        # rescaled_params = params.clone().detach().requires_grad_(True)
        # rescaled_params[0] = 5 + rescaled_params[0] * 195
        # rescaled_params[1] = 100 + rescaled_params[1] * 2900
        # rescaled_params[2] = 0 + rescaled_params[2] * 0.45
        env.simulator.material.from_torch(params)

        env.simulator.set_material_params_kernel()

        # ti_grids = []

        # for s in range(len(action)-1):
        #     env.simulator.step_kernel(action[s])
        #     env.simulator.compute_grid_m_kernel(0)
        #     ti_grids.append(env.simulator.grid_m.to_torch())
        env.simulator.step_kernel(action[0])

        env.simulator.compute_grid_m_kernel(0)

        return env.simulator.grid_m.to_torch()
        # return np.array(ti_grids)


    @staticmethod
    def backward(ctx, grad_output):

        ti.clear_all_gradients()

        env = ctx.env
        params = ctx.params
        action = ctx.action

        env.simulator.grid_m.grad.from_torch(grad_output)

        env.simulator.compute_grid_m_kernel.grad(0)

        # for s in reversed(range(len(action)-1)):
        #     env.simulator.step_kernel_grad(action[s])
        env.simulator.step_kernel_grad(action[0])

        env.simulator.set_material_params_kernel.grad()
        
    
        mat_grad = torch.empty(3)
        for i in range(3):
            mat_grad[i] = env.simulator.material.grad[i]
            # print('mat grad i ', env.simulator.material.grad[i])
        # params_grad = env.simulator.material.grad.to_torch()

        return mat_grad, None, None

class SolverMatChamfer:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def solve(self, init_actions=None, callbacks=()):
        env = self.env

        # initialize material parameters; YS, E, nu
        # material_params = torch.tensor([0.75, 0.25, 0.75], requires_grad=True)
        m_YS = torch.tensor(0.75, requires_grad=True)
        m_E = torch.tensor(0.5, requires_grad=True)
        m_nu = torch.tensor(0.5, requires_grad=True)
        # material_params = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        # optimizer = torch.optim.Adam([material_params], lr=0.05)
        # optimizer = torch.optim.SGD([material_params], lr=0.00001, momentum=0.9)
        optimizer = torch.optim.SGD([m_YS, m_E, m_nu], lr=1e14, momentum=0.9)


        # print('material_params', material_params)

        init_actions = self.init_actions(env, self.cfg)

        # ppos_seq_target = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', env.cfg.loss.ppos_path))
        ppos_seq_target = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../', env.cfg.loss.grid_path))
        # ppos_seq_target = ppos_seq_target[1:-1] # remove initial state
        ppos_seq_target = torch.tensor(ppos_seq_target)

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

            # with open('grid.npy', 'wb') as f:
            #     np.save(f, p_pos_seq)
            # with open('ppos_seq_target.npy', 'wb') as f:
            #     np.save(f, ppos_seq_target.detach().numpy())
            # exit()
            
            loss = ((ppos_seq_target[0] - p_pos_seq)**2).mean()

            # lo = (ppos_seq_target - p_pos_seq)**2
            # print('loss mean', lo.mean(axis=(1, 2)))

            return loss

        # best_action = None
        best_material = None
        best_loss = 1e10

        steps = []
        ct0_vals = []
        ct1_vals = []
        ct2_vals = []
        loss_vals = []

        actions = init_actions
        for iter in range(self.cfg.n_iters):
            print('iter', iter, '/', self.cfg.n_iters)
            material_params = torch.stack([m_YS, m_E, m_nu])
            # self.params = actions.copy() # not doing anything
            loss = forward(env_state['state'], actions, material_params, ppos_seq_target)
            if loss < best_loss:
                best_loss = loss
                best_material = material_params.clone()
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            print('material_params', material_params)
            print('material_params grad', material_params.grad)
            print('loss ', loss)
            # m_YS.grad = material_params.grad[0]
            # m_E.grad = material_params.grad[1]
            # m_nu.grad = material_params.grad[2]
            print('mat', m_YS, m_E, m_nu)
            print('mat grad', m_YS.grad, m_E.grad, m_nu.grad)

            # exit()
            optimizer.step()

            
            # m_YS = m_YS.clamp(0, 1)
            # m_E = m_E.clamp(0, 1)
            # m_nu = m_nu.clamp(0, 1)

            # with torch.no_grad():
                # material_params[:] = material_params.clamp(0, 1)

            steps.append(iter)
            ct0_vals.append(material_params.detach().numpy()[0])
            ct1_vals.append(material_params.detach().numpy()[1])
            ct2_vals.append(material_params.detach().numpy()[2])
            loss_vals.append(loss.detach().numpy())
        
            import matplotlib.pyplot as plt

            xpoints = np.array(steps)
            ypoints = np.array(ct0_vals)

            plt.clf()
            plt.plot(xpoints, ypoints)
            plt.title('Optimizing YS value')
            plt.xlabel('steps')
            plt.ylabel('YS')
            plt.savefig('output/ct0.png')
            plt.clf()

            xpoints = np.array(steps)
            ypoints = np.array(ct1_vals)

            plt.plot(xpoints, ypoints)
            plt.title('Optimizing E value')
            plt.xlabel('steps')
            plt.ylabel('E')
            plt.savefig('output/ct1.png')
            plt.clf()

            xpoints = np.array(steps)
            ypoints = np.array(ct2_vals)

            plt.plot(xpoints, ypoints)
            plt.title('Optimizing nu value')
            plt.xlabel('steps')
            plt.ylabel('nu')
            plt.savefig('output/ct2.png')
            plt.clf()

            xpoints = np.array(steps)
            ypoints = np.array(loss_vals)

            plt.plot(xpoints, ypoints)
            plt.title('Loss while optimization')
            plt.xlabel('steps')
            plt.ylabel('Loss')
            plt.savefig("output/loss.png")
            plt.clf()
    

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
