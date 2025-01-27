'''
Modified from https://github.dev/keirp/stochastic_offline_envs
'''

class BaseSampler:

	def collect_trajectories(self, n_interactions, n_trajs=None):
		"""Sample at most n_interactions data and return"""
		raise NotImplementedError