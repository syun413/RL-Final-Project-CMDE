# RL-Final-Project-CMDE
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

## Costmap Generation

### Task Environment
We use the task `reach_target` in RLBench as task environment. To fix the costmap, we set the seed for environent scene generation to be the same as the demo `variation=2, episode=9`

### Costmap generation
We utilize the previously developed work, VoxPoser (https://voxposer.github.io), to generate the cost maps for our approach. The seed for scene generation is set to be the same as our training environment. 

In VoxPoser, the LLM and VLM generate the target map and avoidance map, which indicate regions of interest and regions to be avoided, respectively. In both maps, lower values correspond to areas that the agent should approach more closely, with the target map emphasizing regions to be prioritized and the avoidance map highlighting areas to be avoided. In our experiment, we leverage target map as the cost map for action guidance, and direct the agent towards regions that are deemed more favorable for task completion.



## Start Training
Environment setup
```
# Only need to run once
Xorg -noreset -config ./xorg.conf :99 &

# Need to set the DISPLAY every time when starting a new Terminal.
export DISPLAY=:99
```

Start training
```
# Run Baseline
python ppo_baseline.py

# Run CostMap-Driven Exploration
python ppo_action.py

# Run CostMap-Driven Reward
python ppo_costmap_reward.py
```
