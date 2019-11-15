from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import networkx as nx
import numpy as np
import tensorflow as tf

from tf_agents.policies import tf_policy

class SearchPolicy(tf_policy.Base):
	def __init__(self, agent, open_loop=False):
		self._agent = agent
		self._open_loop = open_loop
		self._g = self._build_graph()
		super(SearchPolicy, self).__init__(agent.policy.time_step_spec,
											 agent.policy.action_spec)
	
	def _build_graph(self):
		g = nx.DiGraph()
		pdist_combined = np.max(pdist, axis=0)
		for i, s_i in enumerate(rb_vec):
			for j, s_j in enumerate(rb_vec):
			length = pdist_combined[i, j]
			if length < self._agent._max_search_steps:
				g.add_edge(i, j, weight=length)
		return g
	
	def _get_path(self, time_step):
		start_to_rb = agent._get_pairwise_dist(ts.observation['observation'],
												 rb_vec,
												 aggregate='min',
												 masked=True).numpy().flatten()
		rb_to_goal = agent._get_pairwise_dist(rb_vec,
												ts.observation['goal'],
												aggregate='min',
												masked=True).numpy().flatten()

		g2 = self._g.copy()
		
		for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb, rb_to_goal)):
			if dist_from_start < self._agent._max_search_steps:
				g2.add_edge('start', i, weight=dist_from_start)
			if dist_to_goal < self._agent._max_search_steps:
				g2.add_edge(i, 'goal', weight=dist_to_goal)
		
		path = nx.shortest_path(g2, 'start', 'goal')
		edge_lengths = []
		
		for (i, j) in zip(path[:-1], path[1:]):
			edge_lengths.append(g2[i][j]['weight'])
		
		wypt_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
		waypoint_vec = list(path)[1:-1]

		return waypoint_vec, wypt_to_goal_dist[1:]
	
	def _action(self, time_step, policy_state=(), seed=None):
		goal = time_step.observation['goal']
		dist_to_goal = self._agent._get_dist_to_goal(time_step)[0].numpy()

		if self._open_loop:
			if time_step.is_first():
				self._waypoint_vec, self._wypt_to_goal_dist_vec = self._get_path(time_step)
				self._waypoint_counter = 0
			waypoint = rb_vec[self._waypoint_vec[self._waypoint_counter]]
			time_step.observation['goal'] = waypoint[None]
			dist_to_waypoint = self._agent._get_dist_to_goal(time_step)[0].numpy()
			if dist_to_waypoint < self._agent._max_search_steps:
				self._waypoint_counter = min(self._waypoint_counter + 1,
											 len(self._waypoint_vec) - 1)
			waypoint = rb_vec[self._waypoint_vec[self._waypoint_counter]]
			time_step.observation['goal'] = waypoint[None]
			dist_to_waypoint = self._agent._get_dist_to_goal(time_step._replace())[0].numpy()
			dist_to_goal_via_wypt = dist_to_waypoint + self._wypt_to_goal_dist_vec[self._waypoint_counter]
		else:
			(waypoint, dist_to_goal_via_wypt) = self._agent._get_waypoint(time_step)
			dist_to_goal_via_wypt = dist_to_goal_via_wypt.numpy()
			
		if (dist_to_goal_via_wypt < dist_to_goal) or \
			(dist_to_goal > self._agent._max_search_steps):
			time_step.observation['goal'] = tf.convert_to_tensor(waypoint[None])
		else:
			time_step.observation['goal'] = goal

		return self._agent.policy.action(time_step, policy_state, seed)
