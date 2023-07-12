import minari

# print(minari.__version__)

# local_datasets = minari.list_local_datasets()
# print(local_datasets.keys())
# remote_datasets = minari.list_remote_datasets()
# print(remote_datasets.keys())

# minari.download_dataset(dataset_id="pointmaze-medium-v1")

dataset = minari.load_dataset('pointmaze-medium-local')
print(dataset.total_episodes)

dataset_itr = dataset.iterate_episodes()
dataset_itr = list(dataset_itr)
print(len(dataset_itr))
episode = dataset_itr[0]
# episode_dict = episode.__dict__
# print(dir(episode))
print(episode.id)
# print(episode.id) # 0
# print(episode.seed) # 123
# print(episode.observations.keys()) # dict_keys(['achieved_goal', 'desired_goal', 'observation'])
# print(episode.rewards)
# achieved_goal = episode.observations['achieved_goal'] # for point maze, this is useless, as it is just current pos
# desired_goal = episode.observations['desired_goal']
# observation = episode.observations['observation']
# print(type(achieved_goal), type(desired_goal))
# for i in range(10):
#     print(f"{achieved_goal[i,:]}, {desired_goal[i,:]}, {observation[i,:]}")

