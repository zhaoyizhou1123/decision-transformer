from collections import namedtuple

Trajectory = namedtuple(
    "Trajectory", ["observations", "actions", "rewards", "returns", "timesteps", "terminated", "truncated", "infos"])
'''
Each attribute is also a list, element shape according to env, length equal to horizon
'''

def show_trajectory(traj: Trajectory, timesteps = None):
    '''
    print a trajectory for specified timesteps
    timesteps: None or list
    '''
    traj_finish = False
    # idx = 0

    obss = traj.observations
    acts = traj.actions
    rs = traj.rewards
    rets = traj.returns
    ts = traj.timesteps
    terminateds = traj.terminated
    truncateds = traj.truncated
    infos = traj.infos

    if timesteps is None:
        loop_range = range(len(obss))
    else:
        loop_range = timesteps
    
    for idx in loop_range:
        assert idx in range(len(obss)), f"idx {idx} out of range {len(obss)}!"
        obs = obss[idx]
        act = acts[idx]
        r = rs[idx]
        ret = rets[idx]
        t = ts[idx]
        terminated = terminateds[idx]
        truncated = truncateds[idx]
        info = infos[idx]
        print(f"Timestep {t}, obs {obs}, act {act}, r {r}, ret {ret}")
        if terminated:
            print(f"Terminated")
        if truncated:
            print(f"Truncated")