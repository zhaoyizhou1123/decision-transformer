# Combo style dynamics model

import __init__

import argparse
import os
import pickle
import wandb
import time
import math
import torch
import numpy as np
from copy import deepcopy


from torch.utils.tensorboard import SummaryWriter  
from maze.utils.dataset import TrajCtxDataset, TrajNextObsDataset, ObsActDataset, TrajCtxWeightedDataset
from maze.scripts.rollout import rollout_expand_trajs, test_rollout_combo, rollout_combo, test_rollout_diffusion, rollout_diffusion
from maze.algos.stitch_rcsl.training.trainer_dynamics import EnsembleDynamics 
from maze.algos.stitch_rcsl.training.trainer_base import TrainerConfig
from maze.algos.stitch_rcsl.training.train_mdp import BehaviorPolicyTrainer
from maze.algos.stitch_rcsl.models.mlp_policy import StochasticPolicy
from maze.utils.buffer import ReplayBuffer
from maze.utils.trajectory import Trajs2Dict
from maze.utils.scalar import StandardScaler
from maze.algos.stitch_rcsl.models.dynamics_normal import EnsembleDynamicsModel
from maze.utils.logger import Logger, make_log_dirs
from maze.utils.none_or_str import none_or_str
from maze.utils.setup_logger import setup_logger
from maze.utils.setup_seed import setup_seed

def run(args):
    if args.env_type == 'pointmaze':
        from create_maze_dataset import create_env_dataset
        point_maze_offline = create_env_dataset(args)
        env = point_maze_offline.env_cls()

        # Add a get_true_observation method for Env
        def get_true_observation(obs):
            '''
            obs, obs received from pointmaze Env
            '''
            return obs['observation']
    
        setattr(env, 'get_true_observation', get_true_observation)

        horizon = args.horizon
        obs_shape = env.observation_space['observation'].shape
        obs_dim = env.observation_space['observation'].shape[0]
        action_dim = env.action_space.shape[0]
        trajs = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info

        assert len(trajs[0].observations) == horizon, f"Horizon mismatch: {len(trajs[0].observations)} and {horizon}"
        
        # Get dict type dataset, for dynamics training
        dynamics_dataset = Trajs2Dict(trajs)
    else:
        raise Exception(f"Unimplemented env type {args.env_type}")

    # horizon = env.get_horizon()
    # num_actions = env.get_num_action()
    # print(f"Num_actions = {num_actions}")

    # initialize a trainer instance and kick off training
    # epochs = args.epochs

    # Set up environment
    # if args.hash:
    #     hash_method = state_hash
    # else:
    #     hash_method = None

    cur_time = time.localtime(time.time())
    format_time = f"{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}{cur_time.tm_hour:02d}{cur_time.tm_min:02d}"
    data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"
    if args.tb_path is not None:
        # Create tb log dir
        # data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"
        # if args.algo == 'dt':
        #     tb_dir = f"{args.algo}_{data_file}_ctx{args.ctx}_batch{args.batch}_goal{args.goal}_lr{args.rate}"
        if args.algo == "rcsl-mlp": # 'mlp'
            sample_method = 'sample' if args.sample else 'top'
            if args.arch == '/':
                args.arch = ''
            tb_dir = f"{args.algo}_{data_file}_ctx{args.ctx}_arch{args.arch}_{sample_method}_rep{args.repeat}_batch{args.batch}_goalmul{args.goal_mul}_lr{args.rate}"
            if args.simple_input:
                tb_dir += "_simpleinput"
        else:
            raise Exception(f"Unimplemented algorithm {args.algo}")
        tb_dir_path = os.path.join(args.tb_path,format_time,tb_dir)
        os.makedirs(tb_dir_path, exist_ok=False)
    else:
        tb_dir_path = None



    # Set up wandb
    if args.log_to_wandb:
        wandb.init(
            name=f"{data_file}_{format_time}",
            group=data_file,
            project='rcsl',
            config=args
        )

    # Get current device
    # device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'
    device = args.device

    # Load rollout trajs if possible
    # if args.rollout_ckpt_path is not None and os.path.exists(args.rollout_ckpt_path):
    #     print("Rollout trajs exist, load rollout trajs")
    #     with open(args.rollout_ckpt_path, "rb") as f:
    #         rollout_trajs = pickle.load(args.rollout_ckpt_path)
    # # Load fitted model if possible
    # elif args.mdp_ckpt_dir is not None and os.path.exists(args.mdp_ckpt_dir):
    #     print("MDP model exists, load existing models")
    #     d_model_path = os.path.join(args.mdp_ckpt_dir, "dynamics.pth")
    #     b_model_path = os.path.join(args.mdp_ckpt_dir, "behavior.pth")
    #     i_model_pth = os.path.join(args.mdp_ckpt_dir, "init.pth")
    #     dynamics_model = torch.load(d_model_path, map_location=device)
    #     behavior_model = torch.load(b_model_path, map_location=device)
    #     init_model = torch.load(i_model_pth, map_location=device)
    #     dynamics_model.train(False)
    #     behavior_model.train(False)
    #     init_model.train(False)

    #     print("Rollout")
    #     rollout_trajs = rollout_expand_trajs(dynamics_model, behavior_model, init_state_model,
    #                                         device,args)
    # # Fit model, rollout
    # else:


    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(trajs) * horizon, # number of transitions
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device=device
    )
    real_buffer.load_dataset(dynamics_dataset)

    # log for dynamics (only if we train dynamics)
    if not args.load_dynamics_path:
        log_dirs = make_log_dirs(args.env_type, args.algo_name, args.seed, vars(args))
        # key: output file name, value: output handler type
        output_config = {
            "consoleout_backup": "stdout",
            "policy_training_progress": "csv",
            "dynamics_training_progress": "csv",
            "tb": "tensorboard"
        }
        logger = Logger(log_dirs, output_config)
        logger.log_hyperparameters(vars(args))

    # Create dynamics model
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )

    scaler = StandardScaler()
    dynamics_trainer = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
    )

    # Load trained model
    if args.load_dynamics_path:
        print(f"Loading dynamics model from {args.load_dynamics_path}")
        dynamics_trainer.load(args.load_dynamics_path)
    else:
        print(f"Training dynamics model")
        dynamics_trainer.train(real_buffer.sample_all(), logger, max_epochs_since_update=5, 
                               batch_size=args.dynamics_batch_size)   

    # Get behavior policy model 
    if args.behavior_type == 'mlp': # mlp model
        if args.mdp_ckpt_dir is not None and os.path.exists(os.path.join(args.mdp_ckpt_dir, "behavior.pth")):
            b_model_path = os.path.join(args.mdp_ckpt_dir, "behavior.pth")
            print(f"MDP model exists, load existing models from {b_model_path}")
            # d_model_path = os.path.join(args.mdp_ckpt_dir, "dynamics.pth")
            
            # i_model_pth = os.path.join(args.mdp_ckpt_dir, "init.pth")
            # dynamics_model = torch.load(d_model_path, map_location=device)
            behavior_model = torch.load(b_model_path, map_location=device)
            # init_model = torch.load(i_model_pth, map_location=device)
            # dynamics_model.train(False)
            behavior_model.train(False)
            # init_model.train(False)
        else: # Train behavior policy model
            if args.mdp_ckpt_dir is None:
                b_ckpt_path = os.path.join('./checkpoint',format_time)
            else:
                b_ckpt_path = args.mdp_ckpt_dir
            os.makedirs(b_ckpt_path,exist_ok=True)

            model_dataset = TrajNextObsDataset(trajs)
            print(f"Train behavior policy model")

            conf = TrainerConfig(max_epochs=args.behavior_epoch, batch_size=args.behavior_batch, learning_rate=args.rate,
                                num_workers=args.num_workers, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                                ckpt_path = b_ckpt_path, tb_log = tb_dir_path, r_loss_weight = args.r_loss_weight,
                                weight_decay = 0.1, log_to_wandb = args.log_to_wandb)
            
            behavior_model = StochasticPolicy(obs_dim, action_dim, arch = args.b_arch, n_support = args.n_support)


            behavior_trainer = BehaviorPolicyTrainer(behavior_model, model_dataset, conf)
            behavior_trainer.train()
    elif args.behavior_type == 'diffusion': # diffusion policy
        # config for both training and logger setup
        conf = TrainerConfig(obs_dim = obs_dim,
                             act_dim = action_dim,
                             spectral_norm = False,
                             num_epochs = args.behavior_epoch,
                             num_diffusion_iters = args.num_diffusion_iters,
                             batch_size = args.behavior_batch,
                             algo = args.behavior_type,
                             env_id = args.env_type,
                             expr_name = 'default',
                             seed = args.diffusion_seed,
                             load = args.load_diffusion,
                             save_ckpt_freq = 5,
                             device = args.device)
        
        state_action_dataset = ObsActDataset(trajs)
        # Configure logger
        diffusion_logger = setup_logger(conf)
        print(f"Setup logger")

        from maze.algos.stitch_rcsl.training.trainer_diffusion import DiffusionBC
        diffusion_policy = DiffusionBC(conf, state_action_dataset, diffusion_logger)

        if not args.load_diffusion:
            print(f"Train diffusion behavior policy")
            diffusion_policy.train() # save checkpoint periodically
            diffusion_policy.save_checkpoint(epoch=None) # Save final model
        else:
            print(f"Load diffusion behavior policy")
            diffusion_policy.load_checkpoint(final=True)

    else:
        raise Exception(f"Unimplemented behavior policy type {args.behavior_type}")

    if args.test_rollout:
        # Test rollout
        if args.behavior_type == 'mlp':
            test_rollout_combo(args, dynamics_trainer, real_buffer, behavior_model_raw=behavior_model, based_true_state=args.true_state, init_true_state=args.init_state)
        elif args.behavior_type == 'diffusion':
            print(f"Starting testing diffusion rollout")
            test_rollout_diffusion(args, dynamics_trainer, real_buffer, diffusion_policy, based_true_state=args.true_state, init_true_state=args.init_state)
            # test_rollout_combo(args, dynamics_trainer, real_buffer, behavior_model_raw=None, based_true_state=False, init_true_state=True)
    else:            
        # Get max dataset return
        traj_rets = [traj.returns[0] for traj in trajs]
        max_dataset_return = max(traj_rets)
        print(f"Maximum dataset return is {max_dataset_return}")

        if args.behavior_type == 'mlp':
            rollout_trajs = rollout_combo(args, dynamics_trainer, behavior_model, real_buffer,
                                          threshold=max_dataset_return)
            traj_rets = [traj.returns[0] for traj in (trajs + rollout_trajs)]
            goal = max(traj_rets) * args.goal_mul
        elif args.behavior_type == 'diffusion':
            rollout_trajs = rollout_diffusion(args, dynamics_trainer, diffusion_policy, real_buffer, 
                                              threshold=max_dataset_return)
            traj_rets = [traj.returns[0] for traj in (trajs + rollout_trajs)]
            goal = max(traj_rets) * args.goal_mul
        
        # Output policy
        if args.final_ckpt_path is not None:
            os.makedirs(args.final_ckpt_path, exist_ok=True)
        if args.algo == 'rcsl-mlp':
            offline_train_dataset = TrajCtxDataset(trajs, ctx = args.ctx, single_timestep = False)
            # num_action = env.get_num_action()
            # print(f"num_action={num_action}")
            # model = MlpPolicy(1, num_action, args.ctx, horizon, args.repeat, args.arch, args.n_embd)
            from maze.algos.rcsl.models.mlp_policy import MlpPolicy

            model = MlpPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
                            simple_input=args.simple_input)
            import maze.algos.rcsl.trainer_mlp as trainer_mlp
            goal = offline_train_dataset.get_max_return() * args.goal_mul
            tconf = trainer_mlp.TrainerConfig(max_epochs=args.epochs, batch_size=args.batch, learning_rate=args.rate,
                        lr_decay=True, num_workers=args.num_workers, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                        desired_rtg=goal, ckpt_path= args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
                        ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb)
            output_policy_trainer = trainer_mlp.Trainer(model, offline_train_dataset, tconf)
            print("Begin output policy training")
            output_policy_trainer.train()
        elif args.algo == 'rcsl-dt':
            from maze.algos.decision_transformer.models.decision_transformer import DecisionTransformer
            from maze.algos.decision_transformer.training.seq_trainer import SequenceTrainer
            offline_train_dataset = TrajCtxDataset(trajs, ctx = args.ctx, single_timestep = False, with_mask=True, state_normalize=True)
            goal = offline_train_dataset.get_max_return() * args.goal_mul
            model = DecisionTransformer(
                state_dim=obs_dim,
                act_dim=action_dim,
                max_length=args.ctx,
                max_ep_len=args.horizon,
                hidden_size=args.embed_dim,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_inner=4*args.embed_dim,
                activation_function='relu',
                n_positions=1024,
                resid_pdrop=0.1,
                attn_pdrop=0.1)
            tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch, lr=args.rate,
                        lr_decay=True, num_workers=args.num_workers, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                        desired_rtg=goal, ckpt_path= args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
                        ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb, device=args.device, 
                        debug = args.debug)
            output_policy_trainer = SequenceTrainer(
                config=tconf,
                model=model,
                offline_dataset=offline_train_dataset)
            print("Begin output policy training")
            output_policy_trainer.train()
        elif args.algo == 'stitch-mlp-mix':
            '''
            Upd: mix offline with rollout according to args.offline_ratio
            '''
            # Use CQL logger
            from SimpleSAC.utils import WandBLogger
            from viskit.logging import logger as logger_cql, setup_logger as setup_logger_cql
            # logging_conf = WandBLogger.get_default_config(updates={"prefix":'stitch-mlp',
            #                                                        "project": 'stitch-rcsl',
            #                                                        "output_dir": './mlp_log'})
            # wandb_logger = WandBLogger(config=logging_conf, variant=vars(args))
            setup_logger_cql(
                variant=vars(args),
                exp_id=f"arch{args.arch}--mlp",
                seed=args.seed,
                base_log_dir="./mlp_log/",
                include_exp_prefix_sub_dir=False
            )

            # num_offline = len(trajs)
            # num_rollout = len(rollout_trajs)

            # if (args.offline_ratio == 0): # rollout only
            #     train_dataset = TrajCtxDataset(rollout_trajs, 
            #                                 ctx = args.ctx, single_timestep = False)
            # else:
            #     repeat_rollout = math.ceil(num_offline / args.offline_ratio * (1-args.offline_ratio) / num_rollout)

            #     train_dataset = TrajCtxDataset(trajs + rollout_trajs * repeat_rollout, 
            #                                 ctx = args.ctx, single_timestep = False)

            # Update: use weighted sampling, setup in Trainer
                
            setup_seed(args.seed)
            env.reset(seed=args.seed)
            from maze.algos.stitch_rcsl.models.mlp_policy import RcslPolicy
            model = RcslPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
                            simple_input=args.simple_input)
            

            # Calculate weight
            num_offline_data = len(trajs)
            num_rollout_data = len(rollout_trajs)
            num_total_data = num_offline_data + num_rollout_data
            offline_weight = num_rollout_data / num_total_data # reciprocal coefficient
            rollout_weight = num_offline_data / num_total_data

            # weights = [offline_weight] * num_offline_data + [rollout_weight] * num_rollout_data

            offline_dataset = TrajCtxWeightedDataset(trajs, [offline_weight] * num_offline_data, ctx = args.ctx, single_timestep = False)
            rollout_dataset = TrajCtxWeightedDataset(rollout_trajs, [rollout_weight] * num_rollout_data, ctx = args.ctx, single_timestep = False)

            # train_dataset = TrajCtxWeightedDataset(
            #     trajs + rollout_trajs,
            #     weights)

            # Get max return
            traj_rets = [traj.returns[0] for traj in (trajs + rollout_trajs)]
            goal = max(traj_rets) * args.goal_mul
            # goal = train_dataset.get_max_return() * args.goal_mul

            import maze.algos.stitch_rcsl.training.trainer_mlp as trainer_mlp
            tconf = trainer_mlp.TrainerConfig(max_epochs=args.epochs, batch_size=args.batch, learning_rate=args.rate,
                        lr_decay=True, num_workers=args.num_workers, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                        desired_rtg=goal, ckpt_path = args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
                        ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb, logger = logger_cql,
                        debug=args.debug)
            output_policy_trainer = trainer_mlp.DoubleDataTrainer(model, offline_dataset, rollout_dataset, tconf)
            print("Begin output policy training")
            output_policy_trainer.train()
        elif args.algo == 'stitch-mlp-rolloutonly':
            '''
            Upd: mix offline with rollout according to args.offline_ratio
            '''
            # Use CQL logger
            from SimpleSAC.utils import WandBLogger
            from viskit.logging import logger as logger_cql, setup_logger as setup_logger_cql
            # logging_conf = WandBLogger.get_default_config(updates={"prefix":'stitch-mlp',
            #                                                        "project": 'stitch-rcsl',
            #                                                        "output_dir": './mlp_log'})
            # wandb_logger = WandBLogger(config=logging_conf, variant=vars(args))
            setup_logger_cql(
                variant=vars(args),
                exp_id=f"arch{args.arch}--mlp",
                seed=args.seed,
                base_log_dir="./mlp_log/",
                include_exp_prefix_sub_dir=False
            )

            # num_offline = len(trajs)
            # num_rollout = len(rollout_trajs)

            # if (args.offline_ratio == 0): # rollout only
            #     train_dataset = TrajCtxDataset(rollout_trajs, 
            #                                 ctx = args.ctx, single_timestep = False)
            # else:
            #     repeat_rollout = math.ceil(num_offline / args.offline_ratio * (1-args.offline_ratio) / num_rollout)

            #     train_dataset = TrajCtxDataset(trajs + rollout_trajs * repeat_rollout, 
            #                                 ctx = args.ctx, single_timestep = False)

            # Update: use weighted sampling, setup in Trainer
                
            setup_seed(args.seed)
            env.reset(seed=args.seed)
            from maze.algos.stitch_rcsl.models.mlp_policy import RcslPolicy
            model = RcslPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
                            simple_input=args.simple_input)
            

            # Calculate weight
            num_offline_data = len(trajs)
            num_rollout_data = len(rollout_trajs)
            num_total_data = num_offline_data + num_rollout_data
            offline_weight = num_rollout_data / num_total_data # reciprocal coefficient
            rollout_weight = num_offline_data / num_total_data

            # weights = [offline_weight] * num_offline_data + [rollout_weight] * num_rollout_data

            # offline_dataset = TrajCtxWeightedDataset(trajs, [offline_weight] * num_offline_data, ctx = args.ctx, single_timestep = False)
            rollout_dataset = TrajCtxDataset(rollout_trajs, ctx = args.ctx, single_timestep = False)

            # train_dataset = TrajCtxWeightedDataset(
            #     trajs + rollout_trajs,
            #     weights)

            # Get max return
            traj_rets = [traj.returns[0] for traj in (trajs + rollout_trajs)]
            goal = max(traj_rets) * args.goal_mul
            # goal = train_dataset.get_max_return() * args.goal_mul

            import maze.algos.stitch_rcsl.training.trainer_mlp as trainer_mlp
            tconf = trainer_mlp.TrainerConfig(max_epochs=args.epochs, batch_size=args.batch, learning_rate=args.rate,
                        lr_decay=True, num_workers=args.num_workers, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                        desired_rtg=goal, ckpt_path = args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
                        ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb, logger = logger_cql,
                        debug=args.debug, seed = args.seed)
            output_policy_trainer = trainer_mlp.Trainer(model, rollout_dataset, tconf)
            print("Begin output policy training")
            output_policy_trainer.train()
        elif args.algo == 'stitch-mlp-gaussian':
            '''
            Rollout only, Gaussian policy
            '''
            # Use CQL logger
            from offlinerlkit.policy_trainer import RcslPolicyTrainer
            from offlinerlkit.policy import RcslPolicy
            from offlinerlkit.modules import RcslModule, DiagGaussian, TanhDiagGaussian
            from offlinerlkit.nets import MLP
            from offlinerlkit.utils.logger import make_log_dirs, Logger
            from maze.utils.trajectory import Trajs2RtgDict
            
            setup_seed(args.seed)
            env.reset(seed=args.seed)

            hidden_dims = args.arch.split("-")
            hidden_dims = [int(hidden_dim) for hidden_dim in hidden_dims]
            rcsl_backbone = MLP(input_dim=obs_dim+1, hidden_dims=hidden_dims)
            dist = DiagGaussian(
                latent_dim=getattr(rcsl_backbone, "output_dim"),
                output_dim=action_dim,
                unbounded=True,
                conditioned_sigma=True
            )

            rcsl_module = RcslModule(rcsl_backbone, dist, args.device)
            rcsl_optim = torch.optim.Adam(rcsl_module.parameters(), lr=args.rate)

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(rcsl_optim, args.epochs)

            rcsl_policy = RcslPolicy(
                dynamics_trainer,
                diffusion_policy,
                rcsl_module,
                rcsl_optim,
                device = args.device
            )

            # Creat policy trainer
            rcsl_log_dirs = make_log_dirs(args.env_type, args.algo, f"{args.arch}-s{args.seed}", vars(args), part='rcsl')
            # key: output file name, value: output handler type
            rcsl_output_config = {
                "consoleout_backup": "stdout",
                "policy_training_progress": "csv",
                "dynamics_training_progress": "csv",
                "tb": "tensorboard"
            }
            rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
            rcsl_logger.log_hyperparameters(vars(args))

            output_policy_trainer = RcslPolicyTrainer(
                policy = rcsl_policy,
                eval_env = env,
                offline_dataset = Trajs2RtgDict(trajs),
                rollout_dataset = Trajs2RtgDict(rollout_trajs),
                goal = goal,
                logger = rcsl_logger,
                epoch = args.epochs,
                step_per_epoch = args.step_per_epoch,
                batch_size = args.batch,
                lr_scheduler = lr_scheduler,
                horizon = horizon,
                num_workers = args.num_workers,
                seed = args.seed
                # device = args.device
            )
        
            output_policy_trainer.train()

        elif args.algo == 'stitch-dt':
            pass
        elif args.algo == 'stitch-cql':
            from SimpleSAC.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
            from SimpleSAC.conservative_sac import ConservativeSAC
            from SimpleSAC.utils import Timer, set_random_seed, prefix_metrics, WandBLogger
            from SimpleSAC.replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
            from maze.algos.cql.sampler import TrajSampler
            from viskit.logging import logger as logger_cql, setup_logger as setup_logger_cql

            set_random_seed(args.seed)
            env.reset(seed=args.seed)
            policy = TanhGaussianPolicy(
                obs_dim,
                action_dim,
                args.arch
            )

            qf1 = FullyConnectedQFunction(
                obs_dim,
                action_dim,
                args.arch
            )
            target_qf1 = deepcopy(qf1)

            qf2 = FullyConnectedQFunction(
                obs_dim,
                action_dim,
                args.arch
            )
            target_qf2 = deepcopy(qf2)

            conf = ConservativeSAC.get_default_config()
            sac = ConservativeSAC(conf, policy, qf1, qf2, target_qf1, target_qf2)
            sac.torch_to_device(args.device)

            # Eval module
            sampler_policy = SamplerPolicy(policy, args.device)
            eval_sampler = TrajSampler(env, horizon)

            # dataset
            # num_offline = len(trajs)
            # num_rollout = len(rollout_trajs)

            # if (args.offline_ratio == 0): # rollout only
            #     train_dataset = Trajs2Dict(rollout_trajs)
            # else:
            #     repeat_rollout = math.ceil(num_offline / args.offline_ratio * (1-args.offline_ratio) / num_rollout)
            #     train_dataset = Trajs2Dict(trajs + rollout_trajs * repeat_rollout)

            offline_dataset = Trajs2Dict(trajs)
            rollout_dataset = Trajs2Dict(rollout_trajs)

            # logger
            if args.log_to_wandb:
                logging_conf = WandBLogger.get_default_config()
                wandb_logger = WandBLogger(config=logging_conf, variant=vars(args))
            setup_logger_cql(
                variant=vars(args),
                exp_id=f'arch{args.arch}--cql',
                seed=args.seed,
                base_log_dir=f"./cql_log/",
                include_exp_prefix_sub_dir=False
            )
            viskit_metrics = {}

            # Training
            for epoch in range(args.epochs):
                metrics = {'epoch': epoch}

                with Timer() as train_timer:
                    for batch_idx in range(args.cql_n_train_step_per_epoch):
                        # Use weighted sample from offline_dataset and rollout_dataset
                        num_sample_offline = int(args.batch * args.offline_ratio)
                        num_sample_rollout = args.batch - num_sample_offline

                        if num_sample_offline > 0:
                            batch_offline = subsample_batch(offline_dataset, num_sample_offline)
                        if num_sample_rollout > 0:
                            batch_rollout = subsample_batch(rollout_dataset, num_sample_rollout)
                        
                        if num_sample_offline == 0:
                            batch = batch_rollout
                        elif num_sample_rollout == 0:
                            batch = batch_offline
                        else:
                            batch = {k: np.concatenate([batch_offline[k], batch_rollout[k]], axis=0) for k in batch_offline.keys()}

                        # batch = subsample_batch(train_dataset, args.batch)
                        batch = batch_to_torch(batch, args.device)
                        metrics.update(prefix_metrics(sac.train(batch, bc=False), 'sac'))

                with Timer() as eval_timer:
                    if epoch == 0 or (epoch + 1) % args.cql_eval_period == 0:
                        env.reset(seed=args.seed) # Fix eval seed
                        trajs = eval_sampler.sample(
                            sampler_policy, n_trajs=10, deterministic=True
                        )

                        metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                        metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                        # metrics['average_normalizd_return'] = np.mean(
                        #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                        # )
                        # if args.save_model:
                        if args.log_to_wandb:
                            save_data = {'sac': sac, 'variant': {}, 'epoch': epoch}
                            wandb_logger.save_pickle(save_data, 'model.pkl')

                metrics['train_time'] = train_timer()
                metrics['eval_time'] = eval_timer()
                metrics['epoch_time'] = train_timer() + eval_timer()
                if args.log_to_wandb:
                    wandb_logger.log(metrics)
                viskit_metrics.update(metrics)
                logger_cql.record_dict(viskit_metrics)
                logger_cql.dump_tabular(with_prefix=False, with_timestamp=False)

            if args.log_to_wandb:
                save_data = {'sac': sac, 'variant': {}, 'epoch': epoch}
                wandb_logger.save_pickle(save_data, 'model.pkl')
        else:
            raise(Exception(f"Unimplemented model {args.algo}!"))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=123)

    # Overall configuration
    parser.add_argument('--maze_config_file', type=str, default='./config/maze2.json')
    parser.add_argument('--data_file', type=str, default='./dataset/maze2.dat')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    parser.add_argument('--algo', type=str, default='stitch-mlp-rolloutonly', help="rcsl-mlp, rcsl-dt or stitch-mlp, stitch-cql")
    parser.add_argument('--horizon', type=int, default=200, help="Should be consistent with dataset")
    parser.add_argument('--num_workers', type=int, default=1, help="Dataloader workers")

    # Output policy
    parser.add_argument('--ctx', type=int, default=1)
    # parser.add_argument('--pre_epochs', type=int, default=5, help='epochs to learn the output policy using offline data')
    parser.add_argument('--offline_ratio', type=float, default=0.5, help='ratio of offline data in whole dataset, only useful in stitch')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs to learn the output policy')
    parser.add_argument('--step_per_epoch', type=int, default=1000, help='number of training steps per epoch')
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    # parser.add_argument('--num_steps', type=int, default=500000)
    # parser.add_argument('--num_buffers', type=int, default=50)
    # parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch', type=int, default=256, help='final output policy training batch')
    parser.add_argument('--final_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable. Used to store output policy model" )
    parser.add_argument('--arch', type=str, default='200-200-200-200', help="Hidden layer size of output mlp model" )
    parser.add_argument('--goal_mul', type=float, default=1, help="goal = max_dataset_return * goal_mul")

    # Ouput policy (cql)
    parser.add_argument('--cql_seed', type=int, default=0, help="cql policy seed")
    parser.add_argument('--cql_n_train_step_per_epoch', type=int, default=1000, help="cql policy seed")
    parser.add_argument('--cql_eval_period', type=int, default=1, help="cql eval frequency")
    
    
    
    
    # parser.add_argument('--mdp_batch', type=int, default=128)
    # parser.add_argument('--mdp_epochs', type=int, default=5, help='epochs to learn the mdp model')
    # parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    
    # parser.add_argument('--rollout_data_file', type=str, default=None, help='./dataset/maze_rollout.dat')
    # parser.add_argument('--log_level', type=str, default='WARNING')
    parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of all trainers" )
    parser.add_argument('--hash', action='store_true', help="Hash states if True")
    # parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
    # parser.add_argument('--env_path', type=str, default='./env/env_rev.txt', help='Path to env description file')
    parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension, default -1 for no embedding")
    parser.add_argument('--embed_dim', type=int, default=128, help="dt token embedding dimension")
    parser.add_argument('--n_layer', type=int, default=3, help="Transformer layer")
    parser.add_argument('--n_head', type=int, default=1, help="Transformer head")
    # parser.add_argument('--model', type=str, default='dt', help="mlp or dt")
    parser.add_argument('--n_support_init', type=int, default=5, help="Number of supporting initial states" )
    parser.add_argument('--d_arch', type=str, default='128', help="Hidden layer size of dynamics model" )
    parser.add_argument('--r_loss_weight', type=float, default=0.5, help="[0,1], weight of r_loss" )
    parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
    parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")
    parser.add_argument('--time_depend_s',action='store_true')
    parser.add_argument('--time_depend_a',action='store_true')
    parser.add_argument('--simple_input',action='store_false', help='Only use history rtg info if true')


    # Dynamics model
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    # parser.add_argument("--rollout-freq", type=int, default=1000)
    # parser.add_argument("--rollout-batch-size", type=int, default=50000)
    # parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    # parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=none_or_str, default=None)
    parser.add_argument("--dynamics-batch-size", type=int, default=256, help="batch size for dynamics training")
    parser.add_argument("--seed", type=int, default=1, help="dynamics seed")

    # Behavior policy (diffusion)
    parser.add_argument('--behavior_type', type=str, default='diffusion', help='mlp or diffusion')
    parser.add_argument("--behavior_epoch", type=int, default=100)
    parser.add_argument("--num_diffusion_iters", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion', action="store_true", help="Load final diffusion policy model if true")
    parser.add_argument('--diffusion_seed', type=str, default='0', help="Distinguish runs for diffusion policy")

    # Behavior policy (mlp)
    parser.add_argument('--mdp_ckpt_dir', type=str, default='./checkpoint/mlp_b_policy', help="dir path, used to load/store mlp behavior policy model" )
    parser.add_argument('--b_arch', type=str, default='200-200-200-200', help="Hidden layer size of behavior model" )
    parser.add_argument('--n_support', type=int, default=10, help="Number of supporting action of policy" )

    # Rollout
    parser.add_argument('--true_state', action="store_true", help="Rollout based_true_state")
    parser.add_argument('--init_state', action="store_true", help="Rollout init_true_state")
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")

    # parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--test_rollout", action='store_true')
    args = parser.parse_args()
    print(args)

    run(args=args)