# RL-Final-Project-CMDE
# RLBench PPO - 
## Setup
```
pip3 install torch torchvision torchaudio
pip install gym PyOpenGL matplotlib numpy wheel setuptools gymnasium imageio[ffmpeg] moviepy

git clone https://github.com/stepjam/PyRep.git .local/PyRep
cd .local/PyRep
pip install .
cd ../..

git clone https://github.com/stepjam/RLBench.git .local/RLBench
cd .local/RLBench
pip install .
cd ../..
```
## Start Training
```
# Xorg 這行只需要跑一次
Xorg -noreset -config ./xorg.conf :99 &

# 每次啟動新的 Terminal 就要 set 一次 DISPLAY
export DISPLAY=:99

# Run Baseline
python ppo_baseline.py
python ppo_action.py
```
