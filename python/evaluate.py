# eval_simple.py

from stable_baselines3 import PPO
from sim_shower_env import SimulinkShowerEnv

def main():
    # 1) Eval-Umgebung neu erzeugen
    env = SimulinkShowerEnv(shower_length=60)

    # 2) Modell laden und Env direkt übergeben
    model = PPO.load("saved_models/ppo_shower", env=env, device="cpu")

    # 3) Einzelne Episode
    obs = env.reset()    # obs ist np.array([T0])
    done = False
    total_reward = 0.0

    print("\n=== Evaluation ===")
    while not done:
        # 4) Aktion vorhersagen
        action, _ = model.predict(obs, deterministic=True)

        # 5) Simulationsschritt
        obs, reward, done, _ = env.step(int(action))

        # 6) Ausgabe
        print(f"Action: {int(action)} | Reward: {float(reward)}")
        env.render()

        total_reward += float(reward)

    print(f"\nKumulativer Eval-Reward: {total_reward:.2f}")

    # 7) Engine schließen
    env.close()

if __name__ == "__main__":
    main()
