# sim_shower_env.py

import matlab.engine
import numpy as np
from gymnasium import Env, spaces
from pathlib import Path

class SimulinkShowerEnv(Env):
    """Gym-Wrapper für das Shower-Control Simulink-Modell."""

    def __init__(self, shower_length=60):
        super().__init__()

        # --- Action & Observation Spaces ---
        self.action_space      = spaces.Discrete(3)            # 0,1,2 → ΔT = -1,0,+1
        self.observation_space = spaces.Box(
            low=np.array([0.0]), high=np.array([100.0]), dtype=np.float32
        )

        # --- Verzeichnisse dynamisch ermitteln ---
        project_root       = Path(__file__).resolve().parents[1]
        matlab_models_dir  = project_root / "matlab" / "models"
        if not matlab_models_dir.is_dir():
            raise FileNotFoundError(f"Verzeichnis existiert nicht: {matlab_models_dir!s}")

        # Prüfen, ob das Modell da ist
        model_file = matlab_models_dir / "shower_control.slx"
        if not model_file.is_file():
            raise FileNotFoundError(f"Modell nicht gefunden: {model_file!s}")

        # --- MATLAB Engine & Modell laden ---
        self.eng = matlab.engine.start_matlab("-nosplash -nodesktop")
        # Wechsel in den models-Ordner
        self.eng.cd(str(matlab_models_dir), nargout=0)

        # Modellname (Datei ohne .slx)
        self.model = "shower_control"
        self.eng.load_system(self.model, nargout=0)
        self.eng.set_param(self.model, 'SimulationCommand', 'update', nargout=0)
        self.eng.set_param(self.model, 'LoadExternalInput',   'on', nargout=0)
        self.eng.set_param(self.model, 'ExternalInput',       'u', nargout=0)

        self.shower_length = shower_length

    def reset(self):
        x0 = self.eng.get_param(f"{self.model}/Delay", 'X0')
        self.current_state = float(x0)
        self.steps_left    = self.shower_length
        return np.array([self.current_state], dtype=np.float32)

    def step(self, action):
        delta = float(action - 1)
        ts = self.eng.timeseries(
            matlab.double([delta, delta]),
            matlab.double([0, 1])
        )
        self.eng.workspace['u'] = ts

        out = self.eng.sim(
            self.model,
            'StartTime', '0',
            'StopTime',  '1',
            nargout=1
        )

        sim_state = np.array(self.eng.get(out, 'sim_state'))
        self.current_state = float(sim_state[-1, 0])

        reward = 1 if 37 <= self.current_state <= 39 else -1
        self.steps_left -= 1
        done = self.steps_left <= 0

        return np.array([self.current_state], dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        print(f"Temp: {self.current_state:5.1f} °C | Schritte übrig: {self.steps_left}")

    def close(self):
        if self.eng:
            try:
                status = self.eng.get_param(self.model, 'SimulationStatus')
                if status != 'stopped':
                    self.eng.set_param(self.model, 'SimulationCommand', 'stop', nargout=0)
            finally:
                self.eng.quit()
                self.eng = None
