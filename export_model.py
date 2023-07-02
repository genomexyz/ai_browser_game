import torch
import torchvision
import gym_snakegame
import gymnasium as gym

from PPO import PPO
from PPO import ActorCritic

checkpoint_path = 'snake_prototype.pth'

#has_continuous_action_space = True  # continuous action space; else discrete
has_continuous_action_space = False

max_ep_len = 1000                   # max timesteps in one episode
max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e5)          # save model frequency (in num timesteps)

action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4      # update policy every n timesteps
K_epochs = 80               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)
#####################################################

env = gym.make('gym_snakegame/SnakeGame-v0', size=25, n_target=1, render_mode='human')

# state space dimension
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

# initialize a PPO agent
agent_ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
agent_ppo.load(checkpoint_path)

model = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std)
model.load_state_dict(agent_ppo.policy.state_dict())

print('cek modul', model.modules())

#model.eval()
dummy_input = torch.randn(1, 25*25)

input_names = ["input"]
output_names = ["action", "action_prob", "reward_estimate"]

torch.onnx.export(model, dummy_input, "model.onnx",
                   verbose=True, input_names=input_names,
  output_names=output_names)
