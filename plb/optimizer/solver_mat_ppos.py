import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN
import os

import torch

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

import json
from glpointrast import perspective, PointRasterizer
from scipy.spatial.transform import Rotation
from pytorch3d.loss import chamfer_distance

import matplotlib.pyplot as plt

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}

class RunSimulation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, params, env: TaichiEnv, action, n_steps_warmup, n_steps):
        ctx.env = env
        ctx.params = params
        ctx.action = action
        ctx.n_steps_warmup = n_steps_warmup

        # Set material parameters in Taichi
        env.simulator.material.from_torch(params)

        env.simulator.set_material_params_kernel()

        ctx.n_steps = n_steps
        ti_positions = ti.field(dtype=ti.f64, shape=(n_steps, 5000, 3), needs_grad=True)

        for s in range(n_steps):
            env.simulator.step_kernel(action[n_steps_warmup+s], s)

            for i in range(env.simulator.n_particles):
                for j in ti.static(range(env.simulator.dim)):
                    ti_positions[s, i, j] = env.simulator.x[19*s, i][j]
        
        # ti_positions = ti.field(dtype=ti.f64, shape=(1, 5000, 3), needs_grad=True)
        # for i in range(env.simulator.n_particles):
        #     for j in ti.static(range(env.simulator.dim)):
        #         ti_positions[0, i, j] = env.simulator.x[190, i][j]

        ctx.ti_positions = ti_positions

        return ti_positions.to_torch()

    @staticmethod
    def backward(ctx, grad_output):

        ti.clear_all_gradients()

        env = ctx.env
        params = ctx.params
        action = ctx.action
        ti_positions = ctx.ti_positions
        n_steps = ctx.n_steps
        n_steps_warmup = ctx.n_steps_warmup

        ti_positions.grad.from_torch(grad_output)

        # for i in range(env.simulator.n_particles):
        #     for j in ti.static(range(3)):
        #         env.simulator.x.grad[190, i][j] = ti_positions.grad[0, i, j]

        for s in reversed(range(n_steps)):
            for i in range(env.simulator.n_particles):
                for j in ti.static(range(3)):
                    env.simulator.x.grad[19*s, i][j] = ti_positions.grad[s, i, j]
            env.simulator.step_kernel_grad(action[n_steps_warmup+s], s)

        env.simulator.set_material_params_kernel.grad()
    
        mat_grad = torch.empty(3)
        for i in range(3):
            mat_grad[i] = env.simulator.material.grad[i]

        return mat_grad, None, None, None, None

from PIL import Image as im
def save_depth_image(depth_data, file_name):
    _min = np.amin(depth_data[depth_data != 0])
    _max = np.amax(depth_data[depth_data != 0])
    # print(_min)
    # print(_max)
    _min = -0.7
    _max = -0.4
    disp_norm = (depth_data - _min) * 255.0 / (_max - _min)
    disp_norm = np.clip(disp_norm, a_min = 0, a_max = 255)
    disp_norm[depth_data == 0] = 0
    disp_norm = np.uint8(disp_norm)
    data = im.fromarray(disp_norm).convert('RGB')
    data.save(file_name)

def save_optimization_rollout(predicted_depths, ground_truth_depths, step, views, output_path, n_steps, ppos):

    rollout_path = os.path.join(output_path, f'step_{step}')
    os.makedirs(rollout_path, exist_ok=True)

    for vi, view in enumerate(views):
        os.makedirs(os.path.join(rollout_path, view['view']), exist_ok=True)

        with open(os.path.join(rollout_path, 'ppos.npy'), 'wb') as f:
            np.save(f, ppos)

        for step_i in range(len(predicted_depths)):
            save_depth_image(predicted_depths[step_i][vi].to("cpu").detach().numpy(), os.path.join(rollout_path, view['view'], f'predicted_{step_i:05d}.png'))

        for step_i in range(len(predicted_depths)):
            save_depth_image(ground_truth_depths[vi][step_i].to("cpu").detach().numpy(), os.path.join(rollout_path, view['view'], f'true_{step_i:05d}.png'))

class SolverMatPpos:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def forward_wo_grad(self, params, action, n_steps):
        env = self.env

        material_params = params.detach().clone().requires_grad_(False)

        # Set material parameters in Taichi
        env.simulator.material.from_torch(material_params)
        env.simulator.set_material_params_kernel()

        ti_positions = ti.field(dtype=ti.f64, shape=(n_steps, 5000, 3), needs_grad=False)

        for s in range(n_steps):
            env.simulator.step_kernel(action[s])

            for i in range(env.simulator.n_particles):
                for j in ti.static(range(env.simulator.dim)):
                    ti_positions[s, i, j] = env.simulator.x[18, i][j]
        
        return ti_positions.to_torch()


    def solve(self, data_name, data_path, view_file, n_steps_opt, lr, finite_difference, experiment, init_actions=None, callbacks=()):
        env = self.env


        states = np.load(os.path.join(data_path, "raw", data_name + ".npy"), allow_pickle=True).item()

        scene_info = states['scene_info']
        print(scene_info)
        halfEdge1 = scene_info[0]
        halfEdge2 = scene_info[1]

        env.primitives.primitives[0].size[None] = tuple(halfEdge1.tolist())
        env.primitives.primitives[1].size[None] = tuple(halfEdge2.tolist())


        m_YS_target = (states['YS'] - 5)/195
        m_E_target = (states['E']-100)/2900
        m_nu_target = states['nu']/0.45
        print("target material", m_YS_target, " ", m_E_target, " ", m_nu_target)
        target = [m_YS_target, m_E_target, m_nu_target]

        mat_ini = [0.5, 0.5, 0.5]
        n_steps_warmup = 0
        print("data_name", data_name)
        print("lr", lr)

        depth_file_root = os.path.join(data_path, "depth")
        path_output = os.path.join(data_path, "optimizing")

        # initialize material parameters; YS, E, nu
        m_YS = torch.tensor(mat_ini[0], dtype=torch.float64).requires_grad_(True)
        m_E = torch.tensor(mat_ini[1], dtype=torch.float64).requires_grad_(True)
        m_nu = torch.tensor(mat_ini[2], dtype=torch.float64).requires_grad_(True)
        optimizer = torch.optim.Adam([m_YS, m_E, m_nu], lr=lr)
        # optimizer = torch.optim.Adam([m_YS], lr=lr)

        init_actions = self.init_actions(env, self.cfg, states)

        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        output_path = os.path.join(path_output, experiment, data_name)
        
        os.makedirs(output_path, exist_ok=True)


        def just_forward(sim_state, action, material, n_steps):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)

            p_pos_seq = RunSimulation.apply(material, env, action, n_steps_warmup, n_steps).to('cuda')

            # ti_positions = ti.field(dtype=ti.f64, shape=(5000, 3), needs_grad=False)
            # for i in range(env.simulator.n_particles):
            #     for j in ti.static(range(env.simulator.dim)):
            #         ti_positions[i, j] = env.simulator.x[18, i][j]

            # p_pos_seq = torch.cat((ti_positions.to_torch()[None, :].to('cuda'), p_pos_seq), 0)

            return p_pos_seq


        def forward(sim_state, action, material, n_steps, finite_difference, p_pos_seq_target):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)

            # ti_positions = ti.field(dtype=ti.f64, shape=(5000, 3), needs_grad=False)
            # for i in range(env.simulator.n_particles):
            #     for j in ti.static(range(env.simulator.dim)):
            #         ti_positions[i, j] = env.simulator.x[18, i][j]

            if finite_difference:
                p_pos_seq = self.forward_wo_grad(material, action, n_steps).to('cuda')
            else:
                p_pos_seq = RunSimulation.apply(material, env, action, n_steps_warmup, n_steps).to('cuda')

            # loss = torch.sum(torch.abs(p_pos_seq_target[1:] - p_pos_seq))
            loss = torch.sum(torch.abs(p_pos_seq_target[-1] - p_pos_seq[-1]))
            loss_seq = torch.sum(torch.abs(p_pos_seq_target - p_pos_seq), axis=(1, 2))

            # p_pos_seq = torch.cat((ti_positions.to_torch()[None, :].to('cuda'), p_pos_seq), 0)

            return loss, p_pos_seq, loss_seq


        best_material = None
        best_loss = 1e10

        steps = []
        ct0_vals = []
        ct1_vals = []
        ct2_vals = []
        ct0_grad_vals = []
        ct1_grad_vals = []
        ct2_grad_vals = []
        loss_vals = []
        ppos_seq_vals = []


        actions = init_actions

        state = self.env.get_state()
        x_ini = state['state'][0]

        with torch.no_grad():
            print("infer with ground truth material parameters")
            m_YS_gt = torch.tensor(m_YS_target, dtype=torch.float64).requires_grad_(True)
            m_E_gt = torch.tensor(m_E_target, dtype=torch.float64).requires_grad_(True)
            m_nu_gt = torch.tensor(m_nu_target, dtype=torch.float64).requires_grad_(True)
            material_params = torch.stack([m_YS_gt, m_E_gt, m_nu_gt])
            p_pos_seq_target = just_forward(env_state['state'], actions, material_params, n_steps_opt)
            p_pos_seq_target1 = p_pos_seq_target.to("cpu").detach().numpy()
        
            # print("infer with ground truth material parameters")
            
            # n_steps = 200
            # ti_positions = ti.field(dtype=ti.f64, shape=(n_steps, 5000, 3), needs_grad=True)
            # for s in range(n_steps):
            #     print(s)
            #     for i in range(env.simulator.n_particles):
            #         for j in ti.static(range(env.simulator.dim)):
            #             ti_positions[s, i, j] = env.simulator.x[s, i][j]
            # positions = ti_positions.to_torch().to("cpu").detach().numpy()

            # with open(os.path.join("/home/issei/Documents/UCSD/SuLab/Neural_Simulation/tmp/General/sysid/optimizing", 'x.npy'), 'wb') as f:
            #     np.save(f, positions)
            # exit()
        
        
        # with torch.no_grad():
        #     print("infer with ground truth material parameters")
        #     m_YS_gt = torch.tensor(m_YS_target, dtype=torch.float64).requires_grad_(True)
        #     m_E_gt = torch.tensor(m_E_target, dtype=torch.float64).requires_grad_(True)
        #     m_nu_gt = torch.tensor(m_nu_target, dtype=torch.float64).requires_grad_(True)
        #     material_params = torch.stack([m_YS_gt, m_E_gt, m_nu_gt])
        #     p_pos_seq_target = just_forward(env_state['state'], actions, material_params, n_steps_opt)
        #     p_pos_seq_target2 = p_pos_seq_target.to("cpu").detach().numpy()

        # with torch.no_grad():
        #     print("infer with ground truth material parameters")
        #     m_YS_gt = torch.tensor(m_YS_target, dtype=torch.float64).requires_grad_(True)
        #     m_E_gt = torch.tensor(m_E_target, dtype=torch.float64).requires_grad_(True)
        #     m_nu_gt = torch.tensor(m_nu_target, dtype=torch.float64).requires_grad_(True)
        #     material_params = torch.stack([m_YS+0.02, m_E, m_nu])
        #     p_pos_seq_target2 = just_forward(env_state['state'], actions, material_params, n_steps_opt)
        #     p_pos_seq_target2 = p_pos_seq_target2.to("cpu").detach().numpy()
        
        # with torch.no_grad():
        #     print("infer with ground truth material parameters")
        #     m_YS_gt = torch.tensor(m_YS_target, dtype=torch.float64).requires_grad_(True)
        #     m_E_gt = torch.tensor(m_E_target, dtype=torch.float64).requires_grad_(True)
        #     m_nu_gt = torch.tensor(m_nu_target, dtype=torch.float64).requires_grad_(True)
        #     material_params = torch.stack([m_YS-0.02, m_E, m_nu])
        #     p_pos_seq_target3 = just_forward(env_state['state'], actions, material_params, n_steps_opt)
        #     p_pos_seq_target3 = p_pos_seq_target3.to("cpu").detach().numpy()


        for iter in range(self.cfg.n_iters):
            print('iter', iter, '/', self.cfg.n_iters)
            material_params = torch.stack([m_YS, m_E, m_nu])

            if finite_difference:
                with torch.no_grad():
                    loss, p_pos_seq = forward(env_state['state'], actions, material_params, n_steps_opt, finite_difference, p_pos_seq_target)
                    print('loss = ', loss.to("cpu").detach().numpy())

                    eps = torch.tensor(1e-3)
                    material_params_e = torch.stack([m_YS+eps, m_E, m_nu])
                    loss_e, p_pos_seq_e = forward(env_state['state'], actions, material_params_e, n_steps_opt, finite_difference, p_pos_seq_target)
                    fin_diff_YS = (loss_e - loss) / eps
                    print('loss YS = ', loss_e.to("cpu").detach().numpy())
                    print('YS fin_diff = ', fin_diff_YS.to("cpu").detach().numpy())
                    
                    material_params_e = torch.stack([m_YS, m_E+eps, m_nu])
                    loss_e, p_pos_seq_e = forward(env_state['state'], actions, material_params_e, n_steps_opt, finite_difference, p_pos_seq_target)
                    fin_diff_E = (loss_e - loss) / eps
                    print('loss E = ', loss_e.to("cpu").detach().numpy())
                    print('E fin_diff = ', fin_diff_E.to("cpu").detach().numpy())
                    
                    material_params_e = torch.stack([m_YS+eps, m_E, m_nu])
                    loss_e, p_pos_seq_e = forward(env_state['state'], actions, material_params_e, n_steps_opt, finite_difference, p_pos_seq_target)
                    fin_diff_nu = (loss_e - loss) / eps
                    print('loss nu = ', loss_e.to("cpu").detach().numpy())
                    print('nu fin_diff = ', fin_diff_nu.to("cpu").detach().numpy())

                    m_YS -= lr * fin_diff_YS.to("cpu")
                    m_E -= lr * fin_diff_E.to("cpu")
                    m_nu -= lr * fin_diff_nu.to("cpu")

                    m_YS_grad = fin_diff_YS.to("cpu").detach().numpy().item()
                    m_E_grad = fin_diff_E.to("cpu").detach().numpy().item()
                    m_nu_grad = fin_diff_nu.to("cpu").detach().numpy().item()
            
            else:
                loss, p_pos_seq, loss_seq = forward(env_state['state'], actions, material_params, n_steps_opt, finite_difference, p_pos_seq_target)

                # for i_step in range(2, len(loss_seq)):
                #     optimizer.zero_grad()
                #     loss_seq[i_step].backward(retain_graph=True)
                #     m_YS_grad = m_YS.grad.detach().numpy().item()
                #     m_E_grad = m_E.grad.detach().numpy().item()
                #     m_nu_grad = m_nu.grad.detach().numpy().item()
                #     print(i_step, 'loss = ', loss_seq[i_step].to("cpu").detach().numpy(), ' Gradient YS =', m_YS_grad, 'E =', m_E_grad, 'nu =', m_nu_grad)
                
                optimizer.zero_grad()
                # loss_seq[-1].backward(retain_graph=True)
                loss.backward(retain_graph=True)

                m_YS_grad = m_YS.grad.detach().numpy().item()
                m_E_grad = m_E.grad.detach().numpy().item()
                m_nu_grad = m_nu.grad.detach().numpy().item()

                optimizer.step()
            
            if loss < best_loss:
                best_loss = loss
                best_material = material_params.clone()


            print('loss =', loss.to("cpu").detach().numpy())
            print('YS =', m_YS.detach().numpy(), 'E =', m_E.detach().numpy(), 'nu =', m_nu.detach().numpy())
            print('Gradient YS =', m_YS_grad, 'E =', m_E_grad, 'nu =', m_nu_grad)

            # if np.isnan(m_YS.grad.detach().numpy()) or np.isnan(m_E.grad.detach().numpy()) or np.isnan(m_nu.grad.detach().numpy()):
            #     exit()


            with torch.no_grad():
                m_YS.clamp_(0, 1)
                m_E.clamp_(0, 1)
                m_nu.clamp_(0, 1)

            steps.append(iter)
            ct0_vals.append(m_YS.detach().numpy().item())
            ct1_vals.append(m_E.detach().numpy().item())
            ct2_vals.append(m_nu.detach().numpy().item())
            ct0_grad_vals.append(m_YS_grad)
            ct1_grad_vals.append(m_E_grad)
            ct2_grad_vals.append(m_nu_grad)
            loss_vals.append(loss.to("cpu").detach().numpy())
            ppos_seq_vals.append(p_pos_seq.to("cpu").detach().numpy())

            with open(os.path.join(output_path, 'loss.npy'), 'wb') as f:
                loss_info = {
                    "ct0": np.array(ct0_vals),
                    "ct1": np.array(ct1_vals),
                    "ct2": np.array(ct2_vals),
                    "ct0_grad": np.array(ct0_grad_vals),
                    "ct1_grad": np.array(ct1_grad_vals),
                    "ct2_grad": np.array(ct2_grad_vals),
                    "loss": np.array(loss_vals),
                    "p_pos_seq_vals": np.array(ppos_seq_vals),
                    "p_pos_seq_target1": p_pos_seq_target1,
                    # "p_pos_seq_target2": p_pos_seq_target2,
                    # "p_pos_seq_target3": p_pos_seq_target3

                }
                np.save(f, loss_info)

            self.plot_progress('Optimizing YS value', 'steps', 'YS', np.array(steps), np.array(ct0_vals), os.path.join(output_path, 'ct0.png'), target[0])
            self.plot_progress('Optimizing E value', 'steps', 'E', np.array(steps), np.array(ct1_vals), os.path.join(output_path, 'ct1.png'), target[1])
            self.plot_progress('Optimizing nu value', 'steps', 'nu', np.array(steps), np.array(ct2_vals), os.path.join(output_path, 'ct2.png'), target[2])
            self.plot_progress('YS Gradient', 'steps', 'YS grad', np.array(steps), np.array(ct0_grad_vals), os.path.join(output_path, 'ct0_grad.png'))
            self.plot_progress('E Gradient', 'steps', 'E grad', np.array(steps), np.array(ct1_grad_vals), os.path.join(output_path, 'ct1_grad.png'))
            self.plot_progress('nu Gradient', 'steps', 'nu grad', np.array(steps), np.array(ct2_grad_vals), os.path.join(output_path, 'ct2_grad.png'))
            self.plot_progress('Loss while optimization', 'steps', 'Loss', np.array(steps), np.array(loss_vals), os.path.join(output_path, 'loss.png'))

        env.set_state(**env_state)
        return best_material, actions
    
    def plot_progress(self, title, x_label, y_label, x, y, out, target=None):

        plt.plot(x, y)
        if not target == None:
            plt.axhline(y=target, color='r', linestyle='-')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(out)
        plt.clf()

    def init_actions(self, env, cfg, states):
        action_dim = env.primitives.action_dim
        horizon = cfg.horizon
        # if cfg.init_sampler == 'uniform':
        #     return np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
        # else:
        #     raise NotImplementedError

        # Import and reshape the action sequence
        actions = states['shape_states'][0]
        actions = (states['shape_states'][0, 1:, :, 0:3] - states['shape_states'][0, 0:-1, :, 0:3]) * 100
        actions = actions.reshape(actions.shape[0], -1)

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


def solve_mat_ppos(env, path, logger, args):
    import os, cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    env._max_episode_steps = 38 # overwrite frame count
    T = env._max_episode_steps



    solver = SolverMatPpos(taichi_env, logger, None,
                    n_iters=args.num_steps, softness=args.softness, horizon=T,
                    **{
                        "optim.lr": args.lr, 
                        "optim.type": args.optim, 
                        "init_range": 0.0001
                    })

    mat, actions = solver.solve(data_name=args.data_name, 
                                data_path=args.data_path, 
                                view_file=args.views, 
                                n_steps_opt=args.opt_steps, 
                                lr=args.lr, 
                                finite_difference=args.finite_difference,
                                experiment=args.experiment)
