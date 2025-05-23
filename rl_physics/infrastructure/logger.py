import os
from tensorboardX import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

        self.reward_history = []
        self.reward_std_history = []
        self.reward_steps = []
        self.actor_critic_loss_steps = []
        self.actor_loss_history = []
        self.actor_loss_std_history = []
        self.critic_loss_history = []
        self.critic_loss_std_history = []

    def log_reward(self, reward, reward_std, step):
        self.reward_history.append(reward)
        self.reward_std_history.append(reward_std)
        self.reward_steps.append(step)
        self._summ_writer.add_scalar('reward', reward, step)

    def log_scalar(self, scalar, name, step):
        self._summ_writer.add_scalar(name, scalar, step)
        
    def log_actor_critic_loss(self, actor_loss, actor_loss_std, critic_loss, critic_loss_std, step):
        self.critic_loss_history.append(critic_loss)
        self.critic_loss_std_history.append(critic_loss_std)
        self.actor_loss_history.append(actor_loss)
        self.actor_loss_std_history.append(actor_loss_std)
        self.actor_critic_loss_steps.append(step)
        self._summ_writer.add_scalar('actor_loss', actor_loss, step)
        self._summ_writer.add_scalar('critic_loss', critic_loss, step)

    def get_reward_history(self):
        return np.array(self.reward_history), np.array(self.reward_std_history), np.array(self.reward_steps)
    def get_actor_loss_history(self):
        return np.array(self.actor_loss_history), np.array(self.actor_loss_std_history), np.array(self.actor_critic_loss_steps)
    def get_critic_loss_history(self):
        return np.array(self.critic_loss_history), np.array(self.critic_loss_std_history), np.array(self.actor_critic_loss_steps)
    

    #def log_scalars(self, scalar_dict, group_name, step, phase):
    #    """Will log all scalars in the same plot."""
    #    self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)
    #
    #def log_image(self, image, name, step):
    #    assert(len(image.shape) == 3)  # [C, H, W]
    #    self._summ_writer.add_image('{}'.format(name), image, step)
    #
    #def log_video(self, video_frames, name, step, fps=10):
    #    assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
    #    self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)
    #
    #def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):
    #    # reshape the rollouts
    #    videos = [np.transpose(p['image_obs'], [0, 3, 1, 2]) for p in paths]
    #
    #    # max rollout length
    #    max_videos_to_save = np.min([max_videos_to_save, len(videos)])
    #    max_length = videos[0].shape[0]
    #    for i in range(max_videos_to_save):
    #        if videos[i].shape[0]>max_length:
    #            max_length = videos[i].shape[0]
    #
    #    # pad rollouts to all be same length
    #    for i in range(max_videos_to_save):
    #        if videos[i].shape[0]<max_length:
    #            padding = np.tile([videos[i][-1]], (max_length-videos[i].shape[0],1,1,1))
    #            videos[i] = np.concatenate([videos[i], padding], 0)
    #
    #    # log videos to tensorboard event file
    #    videos = np.stack(videos[:max_videos_to_save], 0)
    #    self.log_video(videos, video_title, step, fps=fps)
    #
    #def log_figures(self, figure, name, step, phase):
    #    """figure: matplotlib.pyplot figure handle"""
    #    assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
    #    self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)
    #
    #def log_figure(self, figure, name, step, phase):
    #    """figure: matplotlib.pyplot figure handle"""
    #    self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)
    #
    #def log_graph(self, array, name, step, phase):
    #    """figure: matplotlib.pyplot figure handle"""
    #    im = plot_graph(array)
    #    self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)
    #
    #def dump_scalars(self, log_path=None):
    #    log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
    #    self._summ_writer.export_scalars_to_json(log_path)
    #
    def flush(self):
        self._summ_writer.flush()




