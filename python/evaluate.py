from stable_baselines3 import PPO
from sim_shower_env import SimulinkShowerEnv

model = PPO.load("saved_models/ppo_shower")
env = SimulinkShowerEnv(shower_length=60)

obs = env.reset()
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(int(action))
    print("Reward:", reward)
    print("Action:", action)
    env.render()
    total_reward += reward

print("Eval Reward:", total_reward)
env.close()
