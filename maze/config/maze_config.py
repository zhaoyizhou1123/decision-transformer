import json

# config = json.load(open('maze1.json','r'))
# print(config)

start = [1,3]
goal = [3,7]

mid_point = [3,3]

up_left = [1,1]
down_left = [3,1]
up_right = [1,7]
down_right = [3,7]

maze = {
    "map": [[1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,1],
            [1,0,1,0,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1]],
    "start": start,
    "goal": goal
}

horizon = 300
n_trajs = int(1e6) // horizon

repeat = n_trajs // 4

sample_args = {
    'starts': [start, start, start, start], 
    'goals': [mid_point, mid_point, [up_left,down_left,goal], [up_left,down_left,goal]], 
    'repeats': [repeat, repeat, repeat, repeat], 
    'randoms': [False, False, False, True]}

json.dump({"maze":maze, "sample_args": sample_args}, open("maze2.json", "w"), indent=4)