# train.py

import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sim_shower_env import SimulinkShowerEnv

def make_env():
    return SimulinkShowerEnv(shower_length=60)

def main():
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)

    n_envs = 1  # erst mal 1, um Startup-Zeit zu checken
    env_fns = [make_env for _ in range(n_envs)]

    t0 = time.time()
    # vec_env = SubprocVecEnv(env_fns)
    vec_env = DummyVecEnv(env_fns)  # Dummy zum Vergleich
    print(f"VecEnv ready in {time.time()-t0:.1f}s")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        device="cuda",               # GPU für Netz-Updates
        tensorboard_log="tb_logs/",  # TensorBoard-Ordner
    )

    t1 = time.time()
    model.learn(total_timesteps=20000, log_interval=10)
    print(f"Training done in {time.time()-t1:.1f}s")

    model.save("saved_models/ppo_shower")
    print(f"✓ Modell gespeichert!")
    vec_env.close()

if __name__=="__main__":
    main()
