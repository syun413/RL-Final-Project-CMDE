import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import trange

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, RelativeFrame
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

# ============================
# PPO Hyperparameters
# ============================
TOTAL_TIMESTEPS = 10000000
NUM_STEPS = 800
BATCH_SIZE = 64
NUM_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
LR = 3e-4
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_INTERVAL = 10
MAX_EPISODE_STEPS = 50

# whether or not use wandb
wandb_log = True
_shaped_rewards = False

# ============================
# RLBench Environment Interaction Setup
# ============================
DELTA_TH = 5e-2
DELTA_RATIO = 100

# ============================
# RLBench Environment Setup
# ============================
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.front_camera.set_all(True)
obs_config.front_camera.image_size = (112, 112)
obs_config.gripper_pose = True
TASK_NAME = "reach_target"



if wandb_log:
    import wandb
    wandb.init(project="ppo_reach_target_costmap_guidance", config={
        "total_timesteps": TOTAL_TIMESTEPS,
        "num_steps": NUM_STEPS,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "lr": LR,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "max_episode_steps": MAX_EPISODE_STEPS,
    })


# Helper function
def task_file_to_task_class(task_file):
    import importlib
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def move_gripper_downward(task):
    pos = task._scene.robot.arm.get_tip().get_position()
    rot = task._scene.robot.arm.get_tip().get_quaternion()
    pose = np.concatenate([pos[:2], [pos[2] - 0.2], rot, [0.0]])
    _ = task.step(pose)


arm_action_mode = EndEffectorPoseViaPlanning(
    absolute_mode=True,
    frame=RelativeFrame.WORLD,
    collision_checking=False
)

action_mode = MoveArmThenGripper(
    arm_action_mode=arm_action_mode,
    gripper_action_mode=Discrete()
)

env = Environment(dataset_root=f"./",
                  action_mode=action_mode,
                  obs_config=obs_config,
                  headless=True,
                  shaped_rewards=_shaped_rewards)
env.launch()
print("Env launched successfully")

task = env.get_task(task_file_to_task_class(TASK_NAME))

# 每次 reset 前先固定 variation 與 seed
demo = env.get_demos(
    task_name=TASK_NAME,
    variation_number=2,
    amount=1,
    from_episode_number=9,
    random_selection=False
)[0]
task.reset_to_demo(demo)
move_gripper_downward(task)
obs = task.get_observation()

action_dim = int(env.action_shape[0]) - 1 - 4
print("Action dim:", action_dim)

obs_shape = obs.front_rgb.shape
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PolicyNetwork(nn.Module):
    def __init__(self, image_shape, action_dim):
        super().__init__()
        c, h, w = 3, image_shape[0], image_shape[1]

        self.features = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.features.fc = nn.Identity()
        self.rgb_mean = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.rgb_std = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)
        for p in self.features.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.features(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU()
        )
        self.policy_mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim).fill_(1))
        self.value_head = nn.Linear(128, 1)

        nn.init.zeros_(self.policy_mean.bias)

        self.cost_map_tensor = None
        self.reset_cost_map()

    def reset_cost_map(self):
        cost_map = np.load(
            'cost_map_var2.npy')
        cost_map_tensor = torch.tensor(cost_map, dtype=torch.float32)
        if self.cost_map_tensor is None:
            self.cost_map_tensor = nn.Parameter(
                cost_map_tensor, requires_grad=False)
        else:
            self.cost_map_tensor.copy_(cost_map_tensor)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.rgb_mean.reshape(1, -1, 1, 1)) / \
            self.rgb_std.reshape(1, -1, 1, 1)
        feat = self.features(x)
        feat = self.fc(feat)
        return feat

    def evaluate_action(self, obs, action):
        feat = self.forward(obs)
        mean = self.policy_mean(feat)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return log_prob

    def trilinear_interpolation(self, cost_map_tensor, normalized_actions):

        x, y, z = normalized_actions[:, 0, 0], normalized_actions[:,
                                                                  0, 1], normalized_actions[:, 0,  2]

        x0 = torch.floor(x).long()
        x1 = torch.ceil(x).long()
        y0 = torch.floor(y).long()
        y1 = torch.ceil(y).long()
        z0 = torch.floor(z).long()
        z1 = torch.ceil(z).long()

        x0 = torch.clamp(x0, 0, cost_map_tensor.shape[0] - 1)
        x1 = torch.clamp(x1, 0, cost_map_tensor.shape[0] - 1)
        y0 = torch.clamp(y0, 0, cost_map_tensor.shape[1] - 1)
        y1 = torch.clamp(y1, 0, cost_map_tensor.shape[1] - 1)
        z0 = torch.clamp(z0, 0, cost_map_tensor.shape[2] - 1)
        z1 = torch.clamp(z1, 0, cost_map_tensor.shape[2] - 1)

        v000 = cost_map_tensor[x0, y0, z0]
        v001 = cost_map_tensor[x0, y0, z1]
        v010 = cost_map_tensor[x0, y1, z0]
        v011 = cost_map_tensor[x0, y1, z1]
        v100 = cost_map_tensor[x1, y0, z0]
        v101 = cost_map_tensor[x1, y0, z1]
        v110 = cost_map_tensor[x1, y1, z0]
        v111 = cost_map_tensor[x1, y1, z1]

        wx = x - x0.float()
        wy = y - y0.float()
        wz = z - z0.float()

        i000 = (1 - wz) * v000 + wz * v001
        i010 = (1 - wz) * v010 + wz * v011
        i100 = (1 - wz) * v100 + wz * v101
        i110 = (1 - wz) * v110 + wz * v111

        i00 = (1 - wy) * i000 + wy * i010
        i10 = (1 - wy) * i100 + wy * i110

        interpolated_values = (1 - wx) * i00 + wx * i10

        return interpolated_values

    def get_action_and_value(self, obs, action=None, pos=None, scene=None, pcd=None):

        feat = self.forward(obs)
        mean = self.policy_mean(feat)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        best_log_prob = None
        if action is None:
            sampled_actions = dist.sample((1000,))
            actionspace_min = torch.tensor(
                [-10.0, -10.0, -10.0],
                dtype=torch.float32, device=sampled_actions.device)
            actionspace_max = torch.tensor(
                [10.0, 10.0, 10.0],
                dtype=torch.float32, device=sampled_actions.device)
            sampled_actions = torch.clamp(
                sampled_actions, min=actionspace_min, max=actionspace_max
            )

            pos = torch.tensor(pos, dtype=torch.float32, device=obs.device)
            adjusted_actions = sampled_actions / DELTA_RATIO + pos
            cost_map_tensor = self.cost_map_tensor
            # cost_map = np.load('/home/twke/ppo-rl-fp-uber/costmap_near.npy')
            # cost_map = np.load('/home/twke/ppo-rl-fp/cost_map.npy')

            # cost_map_tensor = cost_map_tensor[None, None]
            # cost_map_tensor = F.interpolate(
            #     cost_map_tensor, size=(20, 20, 20), mode='trilinear', align_corners=True
            # )[0, 0]
            # cost_map_tensor = cost_map_tensor[:, :, :]
            # import matplotlib.pyplot as plt
            # fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
            # cmap = plt.get_cmap('jet')
            # rgba_img = cmap(1 - cost_map_tensor.cpu().numpy())
            # rgb_img = rgba_img[..., :3]
            # ax.voxels(cost_map_tensor, facecolors=rgb_img, alpha=0.5)
            # obs_rgb = (obs.cpu().numpy().astype(np.float32)[0] / 255.).reshape(-1, 3)
            # pcd = pcd.reshape(-1, 3)
            # mask = (pcd[:, 0] >= scene[0]) & (pcd[:, 0] <= scene[1]) & (pcd[:, 1] >= scene[2]) & (pcd[:, 1] <= scene[3]) & (pcd[:, 2] >= scene[4]) & (pcd[:, 2] <= scene[5])
            # pcd[:, 0] = (pcd[:, 0] - scene[0]) / (scene[1] - scene[0]) * 19
            # pcd[:, 1] = (pcd[:, 1] - scene[2]) / (scene[3] - scene[2]) * 19
            # pcd[:, 2] = (pcd[:, 2] - scene[4]) / (scene[5] - scene[4]) * 19
            # ax.scatter(pcd[mask, 0], pcd[mask, 1], pcd[mask, 2], c=obs_rgb[mask])
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # plt.savefig('debug.png', dpi=256)
            # import ipdb; ipdb.set_trace()

            workspace_min = torch.tensor(
                [scene[0], scene[2], scene[4]],
                dtype=torch.float32, device=adjusted_actions.device
            )
            workspace_max = torch.tensor(
                [scene[1], scene[3], scene[5]],
                dtype=torch.float32, device=adjusted_actions.device
            )
            adjusted_actions = torch.clamp(
                adjusted_actions, min=workspace_min, max=workspace_max
            )
            normalized_actions = (
                (adjusted_actions - workspace_min) /
                (workspace_max - workspace_min) * 100
            )
            normalized_actions = torch.clamp(
                torch.round(normalized_actions), 0, 99
            ).long()

            cost_values = cost_map_tensor[normalized_actions[:, 0, 0],
                                          normalized_actions[:, 0, 1], normalized_actions[:, 0, 2]]

            log_probs = dist.log_prob(sampled_actions).sum(dim=-1).view(-1)

            scores = cost_values * -1

            top_scores, top_indices = torch.topk(scores, 10)

            # selected_index = torch.multinomial(torch.ones(100, device=scores.device), 1).item()
            # best_index = top_indices[selected_index]

            # action = sampled_actions[best_index]
            # best_log_prob = log_probs[best_index]
            # top_actions = sampled_actions[top_indices]

            # Average the top 100 actions
            # action = top_actions.mean(dim=0)
            rand_ind = torch.randint(10, size=()).to(top_indices.device)
            action = sampled_actions[top_indices[rand_ind]]
            best_log_prob = log_probs[top_indices[rand_ind]].mean()

            # Increase the cost of sampled actions to encourage exploration
            s_x = normalized_actions[top_indices[rand_ind], 0, 0]
            s_y = normalized_actions[top_indices[rand_ind], 0, 1]
            s_z = normalized_actions[top_indices[rand_ind], 0, 2]
            self.cost_map_tensor[s_x, s_y, s_z] += 0.5
        else:
            best_log_prob = dist.log_prob(action).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        value = self.value_head(feat).squeeze(-1)

        return action, best_log_prob, entropy.mean(), value, mean


class RolloutBuffer:
    def __init__(self, num_steps, obs_shape, action_dim, device):
        self.device = device
        self.obs_buf = torch.zeros(
            (num_steps,) + obs_shape, dtype=torch.uint8, device=device)
        self.actions = torch.zeros((num_steps, action_dim), device=device)
        self.log_probs = torch.zeros(num_steps, device=device)
        self.rewards = torch.zeros(num_steps, device=device)
        self.values = torch.zeros(num_steps, device=device)
        self.dones = torch.zeros(num_steps, device=device)
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = num_steps

    def store(self, obs, action, log_prob, reward, value, done):
        if self.ptr >= self.max_size:
            raise IndexError(
                f"RolloutBuffer index out of range. ptr={self.ptr}, max_size={self.max_size}")
        self.obs_buf[self.ptr] = torch.from_numpy(
            obs.front_rgb).to(self.device)
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_value):
        path_slice = slice(self.path_start_idx, self.ptr)
        dones = self.dones[path_slice]
        values = torch.cat([self.values[path_slice], last_value], dim=0)
        rewards = self.rewards[path_slice]

        adv = torch.zeros_like(rewards, device=self.device)
        last_gae_lam = 0.0
        for step in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + GAMMA * values[step+1] * \
                next_non_terminal - values[step]
            last_gae_lam = delta + GAMMA * GAE_LAMBDA * \
                next_non_terminal * last_gae_lam
            adv[step] = last_gae_lam

        returns = adv + values[:-1]
        self.advantages = adv
        self.returns = returns
        self.path_start_idx = self.ptr

    def get(self, batch_size):
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (self.obs_buf[batch_idx], self.actions[batch_idx], self.log_probs[batch_idx],
                   self.returns[batch_idx], self.advantages[batch_idx], self.values[batch_idx])


agent = PolicyNetwork(obs_shape, action_dim).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LR)

global_step = 0
start_time = time.time()

done = False
episode_rewards = []
num_updates = TOTAL_TIMESTEPS // NUM_STEPS

rollout_for_vis = [obs.front_rgb]
rollout_count = 0

print("Starting training loop...")
for update in range(num_updates):
    buffer = RolloutBuffer(NUM_STEPS, obs_shape, action_dim, device=device)
    print(f"Update {update+1}/{num_updates} - Starting data collection")

    episode_steps = 0
    episode_return = 0.0

    workspace_minx = task._scene._workspace_minx + 0.02
    workspace_maxx = task._scene._workspace_maxx - 0.02
    workspace_miny = task._scene._workspace_miny + 0.02
    workspace_maxy = task._scene._workspace_maxy - 0.02
    workspace_minz = task._scene._workspace_minz + 0.02
    workspace_maxz = task._scene._workspace_maxz - 0.02

    scene = [workspace_minx, workspace_maxx, workspace_miny,
             workspace_maxy, workspace_minz, workspace_maxz]

    for step in range(NUM_STEPS):
        if step % 100 == 0:
            print(f"Update {update+1}, collecting step {step}/{NUM_STEPS}")

        global_step += 1
        obs_cuda = torch.from_numpy(obs.front_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, value, _ = agent.get_action_and_value(
                obs_cuda,
                pos=task._scene.robot.arm.get_tip().get_position(),
                scene=scene,
                pcd=obs.front_point_cloud
            )

        action_np = action.cpu().numpy()[0]

        # 在 relative_mode，action 是相對的增減
        dx = np.clip(action_np[0], -10, 10)
        dy = np.clip(action_np[1], -10, 10)
        dz = np.clip(action_np[2], -10, 10)
        # rx = np.clip(action_np[0], -10, 10)
        # ry = np.clip(action_np[1], -10, 10)
        # rz = np.clip(action_np[2], -10, 10)
        # euler_rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)

        # get absolute position
        abs_position = task._scene.robot.arm.get_tip().get_position()
        abs_rot = task._scene.robot.arm.get_tip().get_quaternion()

        new_x = abs_position[0] + dx / DELTA_RATIO
        new_y = abs_position[1] + dy / DELTA_RATIO
        new_z = abs_position[2] + dz / DELTA_RATIO

        # 不要超出 workspace 的範圍
        new_x = max(workspace_minx, min(new_x, workspace_maxx))
        new_y = max(workspace_miny, min(new_y, workspace_maxy))
        new_z = max(workspace_minz, min(new_z, workspace_maxz))

        dx = new_x - abs_position[0]
        dy = new_y - abs_position[1]
        dz = new_z - abs_position[2]

        final_action = torch.from_numpy(
            np.concatenate([[dx * DELTA_RATIO, dy * DELTA_RATIO, dz * DELTA_RATIO]])).to(device)
        with torch.no_grad():
            log_prob = agent.evaluate_action(obs_cuda, final_action)

        try:
            dxyz = np.array([dx, dy, dz])
            if np.linalg.norm(dxyz) > DELTA_TH:
                num_delta = np.ceil(np.linalg.norm(
                    dxyz) / DELTA_TH).astype(np.int32)
                delta = dxyz / num_delta if num_delta > 0 else dxyz
                abs_position = task._scene.robot.arm.get_tip().get_position()
                for delta_i in trange(num_delta):
                    obs_new, reward, terminate = task.step(
                        np.concatenate([delta * (delta_i + 1) + abs_position, abs_rot, [0.0]]))
            else:
                abs_position = task._scene.robot.arm.get_tip().get_position()
                obs_new, reward, terminate = task.step(
                    np.concatenate([dxyz + abs_position, abs_rot, [0.0]]))
            rollout_for_vis.append(obs_new.front_rgb)
        except Exception as e:
            print("Error encountered:", e)
            terminate = True
            reward = 0.0
            obs_new = obs

        episode_steps += 1
        episode_return += reward

        print(task._scene.robot.arm.get_tip().get_position(),
              task._scene.task.target.get_position())
        success, terminate_task = task._task.success()

        if success:
            print("Success! Episode return:", episode_return)
            done = True
        elif episode_steps >= MAX_EPISODE_STEPS:
            done = True
        else:
            done = terminate_task or terminate

        buffer.store(obs, final_action, log_prob, reward, value, float(done))
        obs = obs_new

        if done:
            print(
                f"Episode done: return={episode_return:.2f}, length={episode_steps}, global_step={global_step}")
            episode_rewards.append(episode_return)

            if wandb_log:
                wandb.log({"episode_return": episode_return,
                           "global_step": global_step})

            episode_steps = 0
            episode_return = 0.0

            video_suffix = "_success" if success else ""
            video_filename = f"rollout_videos_cmap/rollout_{rollout_count}{video_suffix}.mp4"

            import moviepy.video.io.ImageSequenceClip
            from moviepy.editor import vfx
            os.makedirs("rollout_videos_cmap", exist_ok=True)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                rollout_for_vis, fps=30)
            clip.write_videofile(video_filename)
            rollout_count += 1
            rollout_for_vis = []

            task.reset_to_demo(demo)
            move_gripper_downward(task)
            obs = task.get_observation()
            rollout_for_vis = [obs.front_rgb]

    # GAE & returns 計算
    with torch.no_grad():
        _, _, _, last_value, _ = agent.get_action_and_value(
            torch.from_numpy(obs.front_rgb).unsqueeze(0).to(device),
            pos=task._scene.robot.arm.get_tip().get_position(),
            scene=scene
        )
    buffer.finish_path(last_value)
    print(f"Update {update+1} - Data collection done, starting PPO update")

    # PPO Update
    advantages = buffer.advantages
    adv_std = advantages.std()
    adv_std = adv_std if adv_std > 1e-8 else 1e-8
    advantages = (advantages - advantages.mean()) / adv_std

    total_pg_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_batches = 0

    for epoch in range(NUM_EPOCHS):
        if epoch == 0:
            print(f"Update {update+1} - PPO Epoch {epoch+1}/{NUM_EPOCHS}")
        for obs_batch, action_batch, old_logprob_batch, return_batch, adv_batch, value_batch in buffer.get(BATCH_SIZE):
            _, new_logprob, entropy, new_value, _ = agent.get_action_and_value(
                obs_batch, action_batch,
                pos=task._scene.robot.arm.get_tip().get_position(),
                scene=scene
            )
            ratio = (new_logprob - old_logprob_batch).exp()

            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE,
                                1.0 + CLIP_RANGE) * adv_batch
            pg_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((new_value - return_batch)**2).mean()
            entropy_loss = entropy.mean()

            loss = pg_loss + VF_COEF * value_loss - ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()
            total_batches += 1

    avg_pg_loss = total_pg_loss / total_batches
    avg_value_loss = total_value_loss / total_batches
    avg_entropy = total_entropy / total_batches
    print(f"Update {update+1} - PPO update done. avg_pg_loss={avg_pg_loss:.4f}, avg_value_loss={avg_value_loss:.4f}, avg_entropy={avg_entropy:.4f}")

    if wandb_log:
        wandb.log({
            "avg_pg_loss": avg_pg_loss,
            "avg_value_loss": avg_value_loss,
            "avg_entropy": avg_entropy,
            "global_step": global_step,
        })

    if (update + 1) % CHECKPOINT_INTERVAL == 0:
        os.makedirs("./cmap_guidance_checkpoints", exist_ok=True)
        checkpoint_path = f"./cmap_guidance_checkpoints/ppo_checkpoint_{update+1}.pth"
        torch.save(agent.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

env.shutdown()
print("Training finished.")
