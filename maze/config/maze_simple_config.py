import json

# config = json.load(open('maze1.json','r'))
# print(config)

start = [1,2]
goal = [2,4]

mid_point = [2,2]

up_left = [1,1]
down_left = [2,1]
# up_right = [1,7]
down_right = [2,4]

maze = {
    "map": [[1,1,1,1,1,1],
            [1,0,0,1,1,1],
            [1,0,0,0,0,1],
            [1,1,1,1,1,1]],
    "start": start,
    "goal": goal
}

horizon = 100
n_trajs = int(1e4) // horizon

repeat = n_trajs // 4

sample_args = {
    'starts': [start, start, start, start], 
    'goals': [[mid_point,goal], [mid_point,goal], [up_left,down_left,goal], [up_left,down_left,goal]], 
    'repeats': [repeat, repeat, repeat, repeat], 
    'randoms': [False, True, False, True]}

json.dump({"maze":maze, "sample_args": sample_args}, open("maze2_simple_expert_slow.json", "w"), indent=4)