from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tqdm
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from environment import *
from agent import UvfAgent
from search_policy import SearchPolicy

tf.compat.v1.enable_v2_behavior()
tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

max_episode_steps = 20
env_name = 'FourRooms'  # Choose one of the environments shown above. 
resize_factor = 5  # Inflate the environment to increase the difficulty.

def train_eval(tf_agent, tf_env, eval_tf_env, num_iterations=2000000,
		# Params for collect
		initial_collect_steps=1000, batch_size=64,
		# Params for eval
		num_eval_episodes=100, eval_interval=10000,
		# Params for checkpoints, summaries, and logging
		log_interval=1000, random_seed=0):
	"""A simple train and eval for UVF.  """
	
	tf.logging.info('random_seed = %d' % random_seed)
	np.random.seed(random_seed)
	random.seed(random_seed)
	tf.set_random_seed(random_seed)
	
	max_episode_steps = tf_env.pyenv.envs[0]._duration
	global_step = tf.compat.v1.train.get_or_create_global_step()
	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			tf_agent.collect_data_spec,
			batch_size=tf_env.batch_size)

	eval_metrics = [
		tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
	]
	
	eval_policy = tf_agent.policy
	collect_policy = tf_agent.collect_policy
	initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
			tf_env,
			collect_policy,
			observers=[replay_buffer.add_batch],
			num_steps=initial_collect_steps)
	
	collect_driver = dynamic_step_driver.DynamicStepDriver(
			tf_env,
			collect_policy,
			observers=[replay_buffer.add_batch],
			num_steps=1)
	
	initial_collect_driver.run = common.function(initial_collect_driver.run)
	collect_driver.run = common.function(collect_driver.run)
	tf_agent.train = common.function(tf_agent.train)
	
	initial_collect_driver.run()
	
	time_step = None
	policy_state = collect_policy.get_initial_state(tf_env.batch_size)

	timed_at_step = global_step.numpy()
	time_acc = 0

	# Dataset generates trajectories with shape [Bx2x...]
	dataset = replay_buffer.as_dataset(
			num_parallel_calls=3,
			sample_batch_size=batch_size,
			num_steps=2).prefetch(3)
	iterator = iter(dataset)
	
	for _ in tqdm.tnrange(num_iterations):
		start_time = time.time()
		time_step, policy_state = collect_driver.run(
				time_step=time_step,
				policy_state=policy_state,
		)
		
		experience, _ = next(iterator)
		train_loss = tf_agent.train(experience)
		time_acc += time.time() - start_time

		if global_step.numpy() % log_interval == 0:
			tf.logging.info('step = %d, loss = %f', global_step.numpy(), train_loss.loss)
			steps_per_sec = log_interval / time_acc
			tf.logging.info('%.3f steps/sec', steps_per_sec)
			time_acc = 0

		if global_step.numpy() % eval_interval == 0:
			start = time.time()
			tf.logging.info('step = %d' % global_step.numpy())
			for dist in [2, 5, 10]:
				tf.logging.info('\t dist = %d' % dist)
				eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
					prob_constraint=1.0, min_dist=dist-1, max_dist=dist+1)

				results = metric_utils.eager_compute(
						eval_metrics,
						eval_tf_env,
						eval_policy,
						num_episodes=num_eval_episodes,
						train_step=global_step,
						summary_prefix='Metrics',
				)
				for (key, value) in results.items():
					tf.logging.info('\t\t %s = %.2f', key, value.numpy())
				# For debugging, it's helpful to check the predicted distances for
				# goals of known distance.
				pred_dist = []
				for _ in range(num_eval_episodes):
					ts = eval_tf_env.reset()
					dist_to_goal = agent._get_dist_to_goal(ts)[0]
					pred_dist.append(dist_to_goal.numpy())
				tf.logging.info('\t\t predicted_dist = %.1f (%.1f)' % (np.mean(pred_dist), np.std(pred_dist)))
			tf.logging.info('\t eval_time = %.2f' % (time.time() - start))
				
	return train_loss


tf.reset_default_graph()

tf_env = env_load_fn(env_name, max_episode_steps,
					 resize_factor=resize_factor,
					 terminate_on_timeout=False)

eval_tf_env = env_load_fn(env_name, max_episode_steps,
						  resize_factor=resize_factor,
						  terminate_on_timeout=True)
agent = UvfAgent(
	tf_env.time_step_spec(),
	tf_env.action_spec(),
	max_episode_steps=max_episode_steps,
	use_distributional_rl=True,
	ensemble_size=3)

train_eval(
	agent,
	tf_env,
	eval_tf_env,
	initial_collect_steps=1000,
	eval_interval=1000,
	num_eval_episodes=10,
	num_iterations=30000,
)

# --------- Visualize rollouts. ---------
eval_tf_env.pyenv.envs[0]._duration = 100  # We'll give the agent lots of time to try to find the goal.
difficulty = 0.8 #@param {min:0, max: 1, step: 0.1, type:"slider"}
max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(prob_constraint=1.0, min_dist=max(0, max_goal_dist * (difficulty - 0.05)), max_dist=max_goal_dist * (difficulty + 0.05))


def get_rollout(tf_env, policy, seed=None):
	np.random.seed(seed)  # Use the same task for both policies.
	obs_vec = []
	waypoint_vec = []
	ts = tf_env.reset()
	goal = ts.observation['goal'].numpy()[0]
	for _ in tqdm.tnrange(tf_env.pyenv.envs[0]._duration):
		obs_vec.append(ts.observation['observation'].numpy()[0])
		action = policy.action(ts)
		waypoint_vec.append(ts.observation['goal'].numpy()[0])
		ts = tf_env.step(action)
		if ts.is_last():
			break
	obs_vec.append(ts.observation['observation'].numpy()[0])
	obs_vec = np.array(obs_vec)
	waypoint_vec = np.array(waypoint_vec)
	return obs_vec, goal, waypoint_vec

plt.figure(figsize=(8, 4))
for col_index in range(2):
	plt.subplot(1, 2, col_index + 1)
	plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
	obs_vec, goal, _ = get_rollout(eval_tf_env, agent.policy)

	plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
	plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+', color='red', s=200, label='start')
	plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+', color='green', s=200, label='end')
	plt.scatter([goal[0]], [goal[1]], marker='*', color='green', s=200, label='goal')
	if col_index == 0:
		plt.legend(loc='lower left', bbox_to_anchor=(0.3, 1), ncol=3, fontsize=16)

plt.show()

# --------- Fill the replay buffer with random data ---------  
replay_buffer_size = 1000 #@param {min:100, max: 1000, step: 100, type:"slider"}

eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
	prob_constraint=0.0,
	min_dist=0,
	max_dist=np.inf)
rb_vec = []

for _ in tqdm.tnrange(replay_buffer_size):
	ts = eval_tf_env.reset()
	rb_vec.append(ts.observation['observation'].numpy()[0])
rb_vec = np.array(rb_vec)

plt.figure(figsize=(6, 6))
plt.scatter(*rb_vec.T)
plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
plt.show()

# --------- Compute the pairwise distances ---------
pdist = agent._get_pairwise_dist(rb_vec, aggregate=None).numpy()
plt.figure(figsize=(6, 3))
plt.hist(pdist.flatten(), bins=range(20))
plt.xlabel('predicted distance')
plt.ylabel('number of (s, g) pairs')
plt.show()

# --------- Graph Construction ---------
cutoff = 5 #@param {min:0, max: 20, type:"slider"}
# To make visualization easier, we only display the shortest edges for each
# node. We will use all edges for planning.
edges_to_display = 8
plt.figure(figsize=(6, 6))

plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
pdist_combined = np.max(pdist, axis=0)
plt.scatter(*rb_vec.T)
for i, s_i in enumerate(tqdm.tqdm_notebook(rb_vec)):
	for count, j in enumerate(np.argsort(pdist_combined[i])):
  		if count < edges_to_display and pdist_combined[i, j] < cutoff:
  			s_j = rb_vec[j]
  			plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
      
plt.show()


# --------- Ensemble of Critics --------- 
cutoff = 5 #@param {min:0, max: 20, type:"slider"}
edges_to_display = 8
plt.figure(figsize=(15, 4))

for col_index in range(agent._ensemble_size):
	plt.subplot(1, agent._ensemble_size, col_index + 1)
	plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
	plt.title('critic %d' % (col_index + 1))

	plt.scatter(*rb_vec.T)
	desc='critic %d / %d' % (col_index + 1, agent._ensemble_size)
	for i, s_i in enumerate(tqdm.tqdm_notebook(rb_vec, desc=desc)):
		for count, j in enumerate(np.argsort(pdist[col_index, i])):
			if count < edges_to_display and pdist[col_index, i, j] < cutoff:
				s_j = rb_vec[j]
				plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)
			
plt.show()

# --------- Search Policy ---------
agent.initialize_search(rb_vec, max_search_steps=5)
search_policy = SearchPolicy(agent, open_loop=True)

# --------- Search Path. ---------
difficulty = 0.6 #@param {min:0, max: 1, step: 0.1, type:"slider"}
max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(prob_constraint=1.0, min_dist=max(0, max_goal_dist * (difficulty - 0.05)), max_dist=max_goal_dist * (difficulty + 0.05))
ts = eval_tf_env.reset()
start = ts.observation['observation'].numpy()[0]
goal = ts.observation['goal'].numpy()[0]
search_policy.action(ts)

plt.figure(figsize=(6, 6))
plot_walls(eval_tf_env.pyenv.envs[0].env.walls)

waypoint_vec = [start]
for waypoint_index in search_policy._waypoint_vec:
	waypoint_vec.append(rb_vec[waypoint_index])
waypoint_vec.append(goal)
waypoint_vec = np.array(waypoint_vec)

plt.scatter([start[0]], [start[1]], marker='+', color='red', s=200, label='start')
plt.scatter([goal[0]], [goal[1]], marker='*', color='green', s=200, label='goal')
plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=16)
plt.show()


# --------- Rollouts with Search. ---------
eval_tf_env.pyenv.envs[0]._duration = 300
seed = np.random.randint(0, 1000000)

difficulty = 0.8 #@param {min:0, max: 1, step: 0.1, type:"slider"}
max_goal_dist = eval_tf_env.pyenv.envs[0].gym.max_goal_dist
eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
		prob_constraint=1.0,
		min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
		max_dist=max_goal_dist * (difficulty + 0.05))


plt.figure(figsize=(12, 5))
for col_index in range(2):
	title = 'no search' if col_index == 0 else 'search'
	plt.subplot(1, 2, col_index + 1)
	plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
	use_search = (col_index == 1)
	np.random.seed(seed)
	ts = eval_tf_env.reset()
	goal = ts.observation['goal'].numpy()[0]
	start = ts.observation['observation'].numpy()[0]
	obs_vec = []
	for _ in tqdm.tnrange(eval_tf_env.pyenv.envs[0]._duration, desc='rollout %d / 2' % (col_index + 1)):
		if ts.is_last():
			break
		obs_vec.append(ts.observation['observation'].numpy()[0])
		if use_search:
			action = search_policy.action(ts)
		else:
			action = agent.policy.action(ts)

		ts = eval_tf_env.step(action)
	
	obs_vec = np.array(obs_vec)

	plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
	plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
							color='red', s=200, label='start')
	plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
							color='green', s=200, label='end')
	plt.scatter([goal[0]], [goal[1]], marker='*',
							color='green', s=200, label='goal')
	
	plt.title(title, fontsize=24)
	if use_search:
		waypoint_vec = [start]
		for waypoint_index in search_policy._waypoint_vec:
			waypoint_vec.append(rb_vec[waypoint_index])
		waypoint_vec.append(goal)
		waypoint_vec = np.array(waypoint_vec)

		plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
		plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
plt.show()
