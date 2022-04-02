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

class Solver:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def solve(self, init_actions=None, callbacks=()):
        env = self.env

        # initialize material parameters; YS, E, nu
        material_params = np.array([0.5, 0.5, 0.5])

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
                for i in range(len(action)):
                    env.step(action[i])
                    self.total_steps += 1
                    loss_info = env.compute_loss_seq(i)
                    if self.logger is not None:
                        self.logger.step(None, None, loss_info['reward'], None, i==len(action)-1, loss_info)
            loss = env.loss.loss[None]
            # return loss, env.primitives.get_grad(len(action))
            return loss, env.simulator.get_grad()

        # best_action = None
        best_material = None
        best_loss = 1e10

        actions = init_actions
        mat = material_params
        for iter in range(self.cfg.n_iters):
            # self.params = actions.copy() # not doing anything
            self.params = mat.copy() # not doing anything
            loss, grad = forward(env_state['state'], actions, mat)
            print('material_params', mat)
            print('env.simulator', env.simulator.yield_stress)
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

        env.set_state(**env_state)
        return best_material


    @staticmethod
    def init_actions(env, cfg):
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

        print((states['YS'] - 5)/195, " ", (states['E']-100)/2900, " ", states['nu']/0.45)

        return actions

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 39

        cfg.init_range = 0.
        cfg.init_sampler = 'uniform'
        return cfg


def solve_mat(env, path, logger, args):
    import os, cv2
    os.makedirs(path, exist_ok=True)
    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    env._max_episode_steps = 39 # overwrite frame count
    T = env._max_episode_steps



    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    action = solver.solve()

    for idx, act in enumerate(action):
        env.step(act)
        img = env.render(mode='rgb_array')
        cv2.imwrite(f"{path}/{idx:04d}.png", img[..., ::-1])
