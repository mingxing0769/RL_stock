import os
import pickle
from typing import Callable

import gymnasium
import torch as th
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 注册环境
register(
    id='StockTradingEnv-v0',
    entry_point='gym_stock_trading.StockTradingEnv:StockTradingEnv'
)

# 定义变量
Learning_rate = 0.0001
N_steps = 2500
N_epochs = 10
Batch_size = 50
Gamma = 0.99
Ent_coef = 0.1
Vf_coef = 0.5
target_kl = 0.05
Clip_range = 0.1
clip_rang_vf = None
Seed = None
Max_grad_norm = 0.5
policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     ortho_init=False,
                     net_arch=dict(pi=[64, 64, 64],
                                   vf=[64, 64, 64]),
                     optimizer_class=th.optim.Adam,
                     optimizer_kwargs=dict(eps=1e-5, weight_decay=1e-4))

model_path = './models/Stock.pkl'
stock_file = './temp/stock_data.pkl'
temp_model_path = './models/'


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def train_trade():
    # 加载股票数据
    with open(stock_file, 'rb') as f:
        df = pickle.load(f)
    print(len(df.loc[:, 'open'].values))

    # 计算训练长度
    Total_time_steps = len(df.loc[:, 'open'].values) * 2
    # Total_time_steps = 2048

    print('训练总步长度：', Total_time_steps)

    # 创建训练gym环境
    train_env = gymnasium.make('StockTradingEnv-v0', df=df)

    # 如果需要加载模型，则从指定路径加载已有模型
    if os.path.exists(model_path):
        model = PPO.load(model_path, train_env)
        # model.learning_rate = 0.00001
        print('已经加载现有模型')
    else:
        print('新建模型')
        model = PPO("MlpPolicy", train_env, learning_rate=linear_schedule(initial_value=Learning_rate), n_steps=N_steps,
                    n_epochs=N_epochs,
                    batch_size=Batch_size, gamma=Gamma, verbose=1, _init_setup_model=True, target_kl=target_kl,
                    ent_coef=Ent_coef, vf_coef=Vf_coef, clip_range=Clip_range, max_grad_norm=Max_grad_norm,
                    device='cuda', policy_kwargs=policy_kwargs, clip_range_vf=clip_rang_vf,
                    tensorboard_log="./log", seed=Seed)

        # 查看./log 终端运行：tensorboard --logdir=./log

    checkpoint_callback = CheckpointCallback(save_freq=N_steps * 50, save_path=f'{temp_model_path}/temp/',
                                             name_prefix='ppo_model', save_replay_buffer=True, save_vecnormalize=True)
    model.learn(total_timesteps=Total_time_steps, callback=checkpoint_callback)

    # 保存模型
    model.save(model_path, include=['policy_kwargs'])
    print(f'模型已经保存')


if __name__ == '__main__':
    # 训练
    train_trade()
