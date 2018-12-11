import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class PolicyNetwork(nn.Module):
    """A policy network for PongDeterministic-v4
    """

    def __init__(self, device=torch.device('cpu')):
        super(PolicyNetwork, self).__init__()
        self.device = device

        # input: 80x80x2, output: 40x40x4
        self.conv1 = nn.Conv2d(2, 8, kernel_size=5, stride=2, padding=2)

        # input: 40x40x4, output: 20x20x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)

        self._fc_in_size = 20 * 20 * 16

        self.fc1 = nn.Linear(self._fc_in_size, 256)
        self.fc2 = nn.Linear(256, 2)

        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self._fc_in_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# preprocess a single frame
# crop image and downsample to 80x80
# stack two frames together as input
def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return img

# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return batch_input

def run_one_episode(env: gym.Env, policy: util.Policy, tmax=1000, render=False):
    """Collects the trajectory of one-episode run of the `policy` upto `tmax` steps
    Returns:
      [(reward, observation, action)]
    """
    episode = []

    # Duplicates the first frame to make up 2 frames for each step
    # only for the initial frame.
    frame0 = env.reset()
    if render:
        env.render()
    frame_stack = np.array([frame0, frame0])
    frame_stack = preprocess_batch(frame_stack)
    # frame_stack.shape = (1, 2, 80, 80)
    reward = 0

    for _ in range(tmax):
        # let the policy decide the action
        action = policy.action(frame_stack)

        # record the transition
        episode.append((reward, frame_stack, action.copy()))

        # we only use RIGHTFIRE(4) and LEFTFIRE(5).
        action += 4

        # we take one action and skip game forward (0 = NOOP)
        frame0, r0, done, _ = env.step(action)
        frame1, r1, done, _ = env.step(action)
        if render:
            env.render()

        frame_stack = preprocess_batch([frame0, frame1])
        reward = r0 + r1

        if done:
            break

    episode.append((reward, frame_stack, -1))
    return episode


def examine_environment():
    env = gym.make('PongDeterministic-v4')

    # examine the environment
    
    print('List of available actions:', env.unwrapped.get_action_meanings())

    env.reset()
    _, _, _, _ = env.step(0)

    # get a frame after 20 steps
    for _ in range(20):
        frame, _, _, _ = env.step(1)

    plt.subplot(1, 2, 1)
    plt.imshow(frame)
    plt.title('original image')

    plt.subplot(1, 2, 2)
    plt.title('preprocessed image')

    # 80 x 80 black and white image
    plt.imshow(preprocess_single(frame), cmap='Greys')
    plt.show()


if __name__ == '__main__':
    # examine_environment()
    # import test_reinforce
    # test_reinforce.test_train_pong_3k_episode(n_episodes=100, render=True)

    import test_ppo
    test_ppo.test_train_pong_3k_episode(n_episodes=100, render=True)
