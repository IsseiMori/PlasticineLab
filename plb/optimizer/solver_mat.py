import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN
import os

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}

class SolverMat:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def solve(self, init_actions=None, callbacks=()):
        env = self.env

        # initialize material parameters; YS, E, nu
        # material_params = np.array([0.75, 0.25, 0.75])
        material_params = np.array([0.0, 1.0, 0.0])

        init_actions = self.init_actions(env, self.cfg)

        # initialize ...
        # optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        optim = OPTIMS[self.optim_cfg.type](material_params, self.optim_cfg)


        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward(sim_state, action, material):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                env.simulator.set_material(material)
                for i in range(len(action)-1):
                    # print(action[i])
                    env.step(action[i])
                    self.total_steps += 1
                    loss_info = env.compute_loss_seq(i)
                    # loss_info = env.compute_loss()
                    if self.logger is not None:
                        self.logger.step(None, None, loss_info['reward'], None, i==len(action)-1, loss_info)
            loss = env.loss.loss[None]
            # return loss, env.primitives.get_grad(len(action))
            return loss, env.simulator.get_grad()

        # best_action = None
        best_material = None
        best_loss = 1e10

        steps = []
        ct0_vals = []
        ct1_vals = []
        ct2_vals = []
        loss_vals = []

        actions = init_actions
        mat = material_params
        for iter in range(self.cfg.n_iters):
            # self.params = actions.copy() # not doing anything
            self.params = mat.copy() # not doing anything
            loss, grad = forward(env_state['state'], actions, mat)
            print('material_params', mat)
            print('grad', grad)
            print('loss ', loss)
            if loss < best_loss:
                best_loss = loss
                # best_action = actions.copy()
                best_material = mat.copy()
            # actions = optim.step(grad)
            mat = optim.step(grad)
            for callback in callbacks:
                callback(self, optim, loss, grad)
            
            steps.append(iter)
            ct0_vals.append(mat[0])
            ct1_vals.append(mat[1])
            ct2_vals.append(mat[2])
            loss_vals.append(loss)
        
        import matplotlib.pyplot as plt

        xpoints = np.array(steps)
        ypoints = np.array(ct0_vals)

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

        print("target materila", (states['YS'] - 5)/195, " ", (states['E']-100)/2900, " ", states['nu']/0.45)

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


def solve_mat(env, path, logger, args):
    import os, cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    env._max_episode_steps = 38 # overwrite frame count
    T = env._max_episode_steps



    solver = SolverMat(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    mat, actions = solver.solve()
    taichi_env.simulator.set_material(mat)

    for idx, act in enumerate(actions):
        env.step(act)
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])
