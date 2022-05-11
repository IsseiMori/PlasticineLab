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

        # ti_grids = []
        # ppos = []
        ctx.n_steps = n_steps
        ti_positions = ti.field(dtype=ti.f64, shape=(n_steps, 5000, 3), needs_grad=True)

        for s in range(n_steps):
            env.simulator.step_kernel(action[n_steps_warmup+s])

            # ti_positions = ti.field(dtype=ti.f32, shape=(5000, 3), needs_grad=True)
            for i in range(env.simulator.n_particles):
                for j in ti.static(range(env.simulator.dim)):
                    ti_positions[s, i, j] = env.simulator.x[18, i][j]

                    
        #     ppos.append(ti_positions.to_torch().numpy())
        # env.simulator.step_kernel(action[0])

        # ti_positions = ti.field(dtype=ti.f32, shape=(5000, 3), needs_grad=True)
        # for i in range(env.simulator.n_particles):
        #     for j in ti.static(range(env.simulator.dim)):
        #         ti_positions[i, j] = env.simulator.x[18, i][j]
        
        ctx.ti_positions = ti_positions

        return ti_positions.to_torch()
        # return np.array(ppos)


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


        for s in reversed(range(n_steps)):
            for i in range(env.simulator.n_particles):
                for j in ti.static(range(3)):
                    env.simulator.x.grad[18, i][j] = ti_positions.grad[s, i, j]
            env.simulator.step_kernel_grad(action[n_steps_warmup+s])

        # env.simulator.step_kernel_grad(action[0])

        env.simulator.set_material_params_kernel.grad()
    
        mat_grad = torch.empty(3)
        for i in range(3):
            mat_grad[i] = env.simulator.material.grad[i]
            # params.grad[i] = env.simulator.material.grad[i]
            # print('mat grad i ', env.simulator.material.grad[i])
        # params_grad = env.simulator.material.grad.to_torch()

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

class SolverMatChamfer:
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

        for s in range(n_steps):
            env.simulator.step_kernel(action[s])


    def solve(self, data_name, data_path, view_file, n_steps_opt, lr, finite_difference, init_actions=None, callbacks=()):
        env = self.env


        states = np.load(os.path.join(data_path, "raw", data_name + ".npy"), allow_pickle=True).item()

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
        # optimizer = torch.optim.SGD([m_YS, m_E, m_nu], lr=lr, momentum=0.0)
        # optimizer = torch.optim.SGD([m_YS], lr=1e-2, momentum=0.0)

        init_actions = self.init_actions(env, self.cfg, states)

        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        output_path = os.path.join(path_output, data_name)

        with open(os.path.join(data_path, view_file), 'r') as f:
            views = json.load(f)

        ground_truth_depths_list = []
        for view in views:
            depth_file_name = os.path.join(depth_file_root, view['view'], data_name, data_name+'.npy')
            print('reading ', depth_file_name)
            depth_feature = torch.tensor(np.load(depth_file_name, allow_pickle=True)).to('cuda')
            print(depth_feature.shape)
            ground_truth_depths_list.append(depth_feature)


        raster_funcs = []
        for view in views:
            rot = Rotation.from_euler('xyz', view['rotation'], degrees=True).as_matrix()
            mv_mat = np.zeros((4,4))
            mv_mat[:3, :3] = rot
            mv_mat[:3, 3] = np.array(view['translation'])
            mv_mat[3, 3] = 1

            proj = perspective(np.pi / 3, 1, 0.1, 10)
            raster_func = PointRasterizer(128, 128, 0.01, mv_mat, proj)
            raster_funcs.append(raster_func)
        
        def unproject_torch(proj, proj_inv, depth_image):
            z = proj[2, 2] * depth_image + proj[2, 3]
            w = proj[3, 2] * depth_image
            z_ndc = z / w

            H, W = depth_image.shape
            ndc = torch.stack(
                [
                    torch.tensor(x, dtype=depth_image.dtype, device=depth_image.device)
                    for x in np.meshgrid(
                        np.arange(0.5, W + 0.4) * 2 / W - 1,
                        np.arange(0.5, H + 0.4)[::-1] * 2 / H - 1,
                    )
                ]
                + [z_ndc, torch.ones_like(z_ndc)],
                axis=-1,
            )
            pos = ndc @ proj_inv.T
            return pos[..., :3] / pos[..., [3]]
        
        proj_matrix = torch.Tensor(np.array([
            [ 1.73205081,  0.,          0.,          0.        ],
            [ 0.,          1.73205081,  0.,          0.        ],
            [ 0.,          0.,         -1.,         -0.1       ],
            [ 0.,          0.,         -1.,          0.        ],
        ])).to('cuda')


        proj_matrix_inv = torch.Tensor(np.array([
            [  0.57735027,   0.,          -0.,           0.        ],
            [  0.,           0.57735027,  -0.,           0.        ],
            [  0.,           0.,          -0.,         -10.        ],
            [  0.,           0.,          -1.,          10.        ],
        ])).to('cuda')


        def forward(sim_state, action, material, n_steps, iter, output_path):
            if self.logger is not None:
                self.logger.reset()
            
            # Shuffle initial particle positions
            # n_particles = len(sim_state[0])
            # sim_state[0] = np.random.random_sample((n_particles, 3)) * 0.2 + np.array([0.4, 0.0, 0.4])

            env.set_state(sim_state, self.cfg.softness, False)

            # self.forward_wo_grad(material, action, n_steps_warmup).to('cuda')

            p_pos_seq = RunSimulation.apply(material, env, action, n_steps_warmup, n_steps).to('cuda')

            predicted_depths = []

            loss_seq = []
            for step in range(n_steps):

                points_predicted = p_pos_seq[step]
                points_predicted = points_predicted - torch.tensor([0.5, 0.1, 0.5]).to('cuda')

                points_predicted = points_predicted.float().contiguous()

                predicted_depths_views = []

                for vi, view in enumerate(views):

                    depth_predicted = raster_funcs[vi].apply(points_predicted)

                    predicted_depths_views.append(depth_predicted)

                    depth_true = ground_truth_depths_list[vi][n_steps_warmup+step]

                    # save_depth_image(depth_predicted.to("cpu").detach().numpy(), os.path.join(output_path, "{:0>4}.png".format(str(step))))

                    # Set background to a reasonable depth
                    depth_predicted[depth_predicted <= -100000] = -100
                    depth_true[depth_true==0] = -100

                    points_projected_true = unproject_torch(proj_matrix, proj_matrix_inv, depth_true)
                    points_projected_pred = unproject_torch(proj_matrix, proj_matrix_inv, depth_predicted)

                    points_projected_true = torch.flatten(points_projected_true, start_dim=0, end_dim=1)
                    points_projected_pred = torch.flatten(points_projected_pred, start_dim=0, end_dim=1)
                    points_projected_true = points_projected_true[None, :]
                    points_projected_pred = points_projected_pred[None, :]

                    loss_seq.append(chamfer_distance(points_projected_true, points_projected_pred, point_reduction="sum")[0])

                predicted_depths.append(torch.stack(predicted_depths_views))
            
            loss_seq = torch.stack(loss_seq)
            loss = torch.sum(loss_seq)
            # loss /= n_steps
            # loss /= 5 # number of views

            predicted_depths = torch.stack(predicted_depths)
            ground_truth_depths = torch.stack(ground_truth_depths_list)

            if iter == 0 or iter % 10 == 0:
                save_optimization_rollout(predicted_depths, ground_truth_depths[:, n_steps_warmup:], iter, views, output_path, n_steps, p_pos_seq.to("cpu").detach().numpy())
            return loss, loss_seq, p_pos_seq

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
        loss_seq_vals = []


        actions = init_actions
        # with torch.no_grad():
        for iter in range(self.cfg.n_iters):
            print('iter', iter, '/', self.cfg.n_iters)
            material_params = torch.stack([m_YS, m_E, m_nu])

            loss, loss_seq, p_pos_seq = forward(env_state['state'], actions, material_params, n_steps_opt, iter, output_path)
            if loss < best_loss:
                best_loss = loss
                best_material = material_params.clone()
            
            # print('loss = ', loss.to("cpu").detach().numpy())

            if finite_difference:
                eps = torch.tensor(1e-5)
                material_params_e = torch.stack([m_YS+eps, m_E, m_nu])
                loss_e, loss_seq_e = forward(env_state['state'], actions, material_params_e, n_steps_opt, iter, output_path)
                fin_diff_YS = (loss_e - loss) / eps
                print('loss YS = ', loss_e.to("cpu").detach().numpy())
                print('YS fin_diff = ', fin_diff_YS.to("cpu").detach().numpy())
                
                material_params_e = torch.stack([m_YS, m_E+eps, m_nu])
                loss_e, loss_seq_e = forward(env_state['state'], actions, material_params_e, n_steps_opt, iter, output_path)
                fin_diff_E = (loss_e - loss) / eps
                print('loss E = ', loss_e.to("cpu").detach().numpy())
                print('E fin_diff = ', fin_diff_E.to("cpu").detach().numpy())
                
                material_params_e = torch.stack([m_YS+eps, m_E, m_nu])
                loss_e, loss_seq_e = forward(env_state['state'], actions, material_params_e, n_steps_opt, iter, output_path)
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
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                m_YS_grad = m_YS.grad.detach().numpy().item()
                m_E_grad = m_YS.grad.detach().numpy().item()
                m_nu_grad = m_YS.grad.detach().numpy().item()

                optimizer.step()


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
            loss_seq_vals.append(loss_seq.to("cpu").detach().numpy())

            with open(os.path.join(output_path, 'loss'), 'wb') as f:
                loss_info = {
                    "ct0": np.array(ct0_vals),
                    "ct1": np.array(ct1_vals),
                    "ct2": np.array(ct2_vals),
                    "ct0_grad": np.array(ct0_grad_vals),
                    "ct1_grad": np.array(ct1_grad_vals),
                    "ct2_grad": np.array(ct2_grad_vals),
                    "loss": np.array(loss_vals),
                    "loss_seq": np.array(loss_seq_vals)
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


def solve_mat_chamfer(env, path, logger, args):
    import os, cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    env._max_episode_steps = 38 # overwrite frame count
    T = env._max_episode_steps



    solver = SolverMatChamfer(taichi_env, logger, None,
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
                                finite_difference=args.finite_difference)
    taichi_env.simulator.set_material(mat)

    for idx, act in enumerate(actions):
        env.step(act)
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])
