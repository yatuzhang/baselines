# ACKTR

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- Need to set environment variable OPENAI_LOGDIR so that it knows where to output the logs
- NOTE: the rewards output in terminal is the clipped reward, to see actual reward you need to go to the OPENAI_LOGDIR and find the monitor files
- `python3 run_atari.py --env "BreakoutNoFrameskip-v4"  --ckpt_dir "/some/dir/"` runs the algorithm for 40M frames = 10M timesteps on the Breakout Atari game.  Saves the model at the ckpt_dir every 100 timesteps
- `python3 sample_trajectory.py --load_model_path "/some/dir/checkpoint01200" --save_dir /some/random/dir` loads the saved model and samples the trajectories for a default of 1500 episodes
