import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sim_shower_env import SimulinkShowerEnv

def make_env():
    return SimulinkShowerEnv(shower_length=60)

def main():
    # 1) Verzeichnisse
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)

    # 2) Parallele Envs
    n_envs = 1  # anpassen je nach Lizenz/RAM
    env_fns = [make_env for _ in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # 3) PPO-Agent
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        device="cuda",
        tensorboard_log="tb_logs/",
    )

    # 4) Training
    model.learn(
        total_timesteps=20_000,
        log_interval=10,
        reset_num_timesteps=False,
    )

    # 5) Speichern
    model_path = os.path.join("saved_models", "ppo_shower")
    model.save(model_path)
    print(f"✓ Modell gespeichert als {model_path}.zip")

    # SubprocVecEnv aufräumen
    vec_env.close()

    # 6) Evaluation (separate Instanz)
    eval_env = SimulinkShowerEnv(shower_length=60)
    episodes = 3
    for ep in range(episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        print(f"\n=== Evaluation Episode {ep+1} ===")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(int(action))
            eval_env.render()
            total_reward += reward
        print(f"Episode {ep+1} Reward: {total_reward}")

    eval_env.close()

if __name__ == "__main__":
    main()
