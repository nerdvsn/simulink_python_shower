# main.py

import os
import numpy as np
from sim_shower_env import SimulinkShowerEnv

def build_model_in_matlab(init_state: int):
    """Erzeugt shower_control.slx mit gegebenem Startwert."""
    from pathlib import Path
    import matlab.engine

    project_root      = Path(__file__).resolve().parents[1]
    build_script_dir  = project_root / "matlab"
    if not build_script_dir.is_dir():
        raise FileNotFoundError(f"Build-Skript-Ordner nicht gefunden: {build_script_dir!s}")

    eng = matlab.engine.start_matlab("-nosplash -nodesktop")
    eng.cd(str(build_script_dir), nargout=0)
    eng.build_shower_control(init_state, nargout=0)
    eng.quit()
    print(f"✓ Simulink-Modell erzeugt mit init_state = {init_state}")

def main():
    # 1) Einmal Modell bauen (erstellte .slx landet in matlab/models/)
    init_state = 38 + np.random.randint(-3, 4)
    build_model_in_matlab(init_state)

    # 2) Umgebung initialisieren & Episode laufen lassen
    env = SimulinkShowerEnv(shower_length=60)
    obs = env.reset()
    print(f"Starttemperatur: {obs[0]:.1f} °C\n")

    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"Action: {action} | Reward: {reward}")
        env.render()
        total_reward += reward

    env.close()
    print(f"\nEpisode beendet - kumulativer Reward: {total_reward}")

if __name__ == "__main__":
    main()
