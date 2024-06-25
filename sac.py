import gym
import os
import time

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

from gym.envs.registration import register
import argparse

# Set up the environment
ENV_ID = "RocketLander-v0"  # Replace with your environment name

if ENV_ID not in gym.envs.registry.env_specs:
    register(
        id=ENV_ID,
        entry_point='env:RocketLander',  # Replace with the correct path to your module
    )

# 定义模型文件路径
model_path = "sac_rocketlander.zip"
tmp_path = "./sb3_log/"


# 创建并行向量化环境
cpu_num = os.cpu_count()
env = make_vec_env(ENV_ID, n_envs=8)
model = None
print('env cpu', cpu_num)

# 检查是否存在已保存的模型
if os.path.exists(model_path + ""):
    # 加载模型
    model = SAC.load(model_path, env=env)
    print("模型已加载。")
else:
    # 定义并训练新的PPO模型
    # model = PPO('MlpPolicy', env, verbose=1)
    model = SAC(
        policy = 'MlpPolicy',
        env = env,
        # n_steps = 1024,
        # batch_size = 64,
        # n_epochs = 4,
        # gamma = 0.999,
        # gae_lambda = 0.98,
        # ent_coef = 0.01,
        verbose=1,
        #device="cuda"
        )
    
    print("模型不存在，开始学习")
    model.learn(total_timesteps=200000)  # 您可以修改训练步数
    model.save(model_path)

new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# 测试模型
demo_env = gym.make('RocketLander-v0')



def test_model():
    obs = demo_env.reset()
    for step_index in range(3000):  # 您可以调整步数以适应您的需求
        action, _states = model.predict(obs)
        obs, rewards, done, info = demo_env.step(action)
        demo_env.render()
        print('ts', step_index, "r:%.5f" % rewards, done, "s:%.3f" % info["speed"], "d:%.8f" % info['distance'], "a:%.3f" % info['angle'], "v:%.3f" % info['vel_a'], "p:%.3f" % info['power'])
        if done:
            obs = demo_env.reset()
            break

def train():

    model.learn(total_timesteps=500000)  # 您可以修改训练步数
    model.save(model_path)
    print("模型已训练并保存。")

    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"平均奖励: {mean_reward} +/- {std_reward}")

    # 如果成功率达到80%，打开UI显示降落过程
    if mean_reward >= 80:  # 假设80分为成功率80%的标准
        pass
    else:
        print("模型成功率未达到80%，继续训练。")
    
    test_model()



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Execution mode between train or test", type=str, default="test", choices=["train", "test", "test_loop"])
    # parser.add_argument("-n", "--model_name", help="Name of the model you want to train/test", default="ppo-RocketLander", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    
    if args.mode == "train":
        while True:
            train()
    elif args.mode == "test":
        test_model()
        time.sleep(2)
    elif args.mode == "test_loop":
        while True:
            test_model()
            time.sleep(2)
