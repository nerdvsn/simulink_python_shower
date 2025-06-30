import matlab.engine
import numpy as np
from gym import Env, spaces
import os

# Pfad zum Ordner, in dem shower_control.slx liegt
MATLAB_MODEL_DIR = r"C:\Users\lukelo_g\devWORK\RL\shower\matlab\models"

class SimulinkShowerEnv(Env):
    """Gym-Wrapper für das Shower-Control Simulink-Modell."""

    def __init__(self, shower_length=60):
        super().__init__()

        # --- Action & Observation Spaces ------------------------
        self.action_space      = spaces.Discrete(3)            # 0,1,2 → ΔT = -1,0,+1
        self.observation_space = spaces.Box(
            low=np.array([0.0]), high=np.array([100.0]), dtype=np.float32
        )

        # --- MATLAB Engine & Modell laden -----------------------
        self.eng = matlab.engine.start_matlab("-nosplash -nodesktop")
        # Wechsel in den Ordner mit dem .slx-File
        self.eng.cd(MATLAB_MODEL_DIR, nargout=0)
        # Modellname (Datei ohne .slx)
        self.model = 'shower_control'
        # Jetzt lädt Simulink das Modell aus matlab/models/shower_control.slx
        self.eng.load_system(self.model, nargout=0)
        self.eng.set_param(self.model, 'SimulationCommand', 'update',   nargout=0)
        self.eng.set_param(self.model, 'LoadExternalInput',   'on',      nargout=0)
        self.eng.set_param(self.model, 'ExternalInput',       'u',       nargout=0)

        self.shower_length = shower_length

    def reset(self):
        # Liest als Startzustand den initialen X0 aus dem Unit Delay
        # (Alternativ könntest du init_state speichern und hier zurücksetzen)
        # Aber: Wir haben das Modell bereits mit X0 gebaut, also holen wir es so:
        #   get_param('shower_control/Delay','X0')
        x0 = self.eng.get_param(f'{self.model}/Delay', 'X0')
        self.current_state = float(x0)
        self.steps_left    = self.shower_length
        return np.array([self.current_state], dtype=np.float32)

    def step(self, action):
        # ΔT = action–1
        delta = float(action - 1)
        ts = self.eng.timeseries(
            matlab.double([delta, delta]), matlab.double([0, 1])
        )
        self.eng.workspace['u'] = ts

        # Simuliere 1 Sekunde
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
                if self.eng.get_param(self.model, 'SimulationStatus') != 'stopped':
                    self.eng.set_param(self.model, 'SimulationCommand', 'stop', nargout=0)
            finally:
                self.eng.quit()
                self.eng = None
