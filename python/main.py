import numpy as np
import matlab.engine
from sim_shower_env import SimulinkShowerEnv


MATLAB_SCRIPT_DIR = "/home/lukelo-tshakulongo/Bakoko/Nkulu/Computer Science/Code/Musualu/simulink_python_shower/matlab"



def build_model_in_matlab(init_state: int):
    """Erzeugt (oder überschreibt) shower_control.slx mit gegebenem Startwert."""
    eng = matlab.engine.start_matlab("-nosplash -nodesktop")
    # 1) Ins MATLAB-Verzeichnis mit build_shower_control.m wechseln
    eng.cd(MATLAB_SCRIPT_DIR, nargout=0)
    # 2) Funktion aufrufen – schreibt shower_control.slx nach matlab/models/
    eng.build_shower_control(init_state, nargout=0)
    eng.quit()
    print(f"✓ Simulink-Modell erzeugt mit init_state = {init_state}")

def main():
    # 1) Modell einmalig bauen (danach kannst du das hier auskommentieren)
    #init_state = 38 + np.random.randint(-3, 4)
    #build_model_in_matlab(init_state)

    # 2) Umgebung initialisieren & Episode laufen lassen
    env = SimulinkShowerEnv(shower_length=60)
    obs = env.reset()
    print(f"Starttemperatur: {obs[0]:.1f} °C\n")

    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"Action: {action}")
        #print(f"Obs: {obs}")
        #print(f"Done: {done}")
        print(f"Rewards: {reward}")
        env.render()
        total_reward += reward

    env.close()
    print(f"\nEpisode beendet - kumulativer Reward: {total_reward}")

if __name__ == "__main__":
    main()
