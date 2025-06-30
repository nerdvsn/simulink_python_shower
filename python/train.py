import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sim_shower_env import SimulinkShowerEnv

def main():
    # 1) Verzeichnis für gespeicherte Modelle anlegen
    os.makedirs("saved_models", exist_ok=True)

    # 2) Vectorized Environment erstellen (für stabileres Training)
    vec_env = DummyVecEnv([lambda: SimulinkShowerEnv(shower_length=60)])

    # 3) PPO-Agent initialisieren
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
    )

    # 4) Training
    total_timesteps = 20_000
    model.learn(total_timesteps=total_timesteps)

    # 5) Modell speichern
    model_path = os.path.join("saved_models", "ppo_shower")
    model.save(model_path)
    print(f"✓ Modell gespeichert als {model_path}.zip")

    # 6) Kurze Evaluation mit deterministischer Policy
    eval_env = SimulinkShowerEnv(shower_length=60)
    obs = eval_env.reset()
    done = False
    total_reward = 0

    print("\n=== Evaluation nach Training ===")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(int(action))
        eval_env.render()
        total_reward += reward

    print(f"\nKumulativer Reward (eval): {total_reward}")
    eval_env.close()

if __name__ == "__main__":
    main()
