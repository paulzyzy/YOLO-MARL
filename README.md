# YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning

This is the official repository for the paper YOLO-MARL: You Only LLM Once for Multi-agent Reinforcement Learning [arxiv](https://arxiv.org/abs/2410.03997).

YOLO-MARL supports both ChatGPT and Claude. Please put your api KEY in the YOLO-MARL/ and name it claude_KEY.txt or openai_KEY.txt.

In YOLO-MARL/src/config/default.yaml, you need to set use_llm by yourself. False for running the MARL baselines and True for running the YOLO-MARL method.

All the baselines we used here are based on [Epymarl](https://github.com/uoe-agents/epymarl). For the denpendencies, you could also refer to Epymarl and then install Openai or Claude dependencies:
```sh
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name="3s5z"
```

By default, YOLO-MARL runs experiments with common rewards (as done previously). To run an experiment with individual rewards for all agents, set `common_reward=False`. For example to run MAPPO in a LBF task with individual rewards:
```sh
pip install openai #If you want to use ChatGPT for your LLM API
pip install anthropic #If you want to use Claude for your LLM API
```
When using the `common_reward=True` setup in environments which naturally provide individual rewards, by default we scalarise the rewards into a common reward by summing up all rewards. This is now configurable and we support the mean operation as an alternative scalarisation. To use the mean scalarisation, set `reward_scalarisation="mean"`.

### Weights and Biases (W&B) Logging
We now support logging to W&B! To log data to W&B, you need to install the library with `pip install wandb` and setup W&B (see their [documentation](https://docs.wandb.ai/quickstart)). After, follow [our instructions](#weights-and-biases).

### Plotting script
We have added a simple plotting script under `plot_results.py` to load data from sacred logs and visualise them for executed experiments. For more details, see [the documentation here](#plotting).

# Table of Contents
- [Extended Python MARL framework - EPyMARL](#extended-python-marl-framework---epymarl)
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)
  - [Installing Dependencies](#installing-dependencies)
  - [Benchmark Paper Experiments](#benchmark-paper-experiments)
  - [Experiments in SMACv2 and SMAClite](#experiments-in-smacv2-and-smaclite)
  - [Registering and Running Experiments in Custom Environments](#registering-and-running-experiments-in-custom-environments)
- [Experiment Configurations](#experiment-configurations)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Logging](#logging)
  - [Weights and Biases](#weights-and-biases)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Plotting](#plotting)
- [Citing PyMARL and EPyMARL](#citing-pymarl-and-epymarl)
- [License](#license)

# Installation & Run instructions

## Installing Dependencies

To install the dependencies for the codebase, clone this repo and run:
```sh
pip install -r requirements.txt
```

To install a set of supported environments, you can use the provided `env_requirements.txt`:
```sh
pip install -r env_requirements.txt
```
which will install the following environments:
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging)
- [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse)
- [PettingZoo](https://github.com/semitable/multiagent-particle-envs) (used for the multi-agent particle environment)
- [Matrix games](https://github.com/uoe-agents/matrix-games)
- [SMAC](https://github.com/oxwhirl/smac)
- [SMACv2](https://github.com/oxwhirl/smacv2)
- [SMAClite](https://github.com/uoe-agents/smaclite)

To install these environments individually, please see instructions in the respective repositories. We note that in particular SMAC and SMACv2 require a StarCraft II installation with specific map files. See their documentation for more details.

Note that the [PAC algorithm](#update-as-of-15th-july-2023) introduces separate dependencies. To install these dependencies, use the provided requirements file:
```sh
pip install -r pac_requirements.txt
```


## Benchmark Paper Experiments

In ["Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks"](https://arxiv.org/abs/2006.07869) we introduce the Level-Based Foraging (LBF) and Multi-Robot Warehouse (RWARE) environments, and additionally evaluate in SMAC, Multi-agent Particle environments, and a set of matrix games. After installing these environments (see instructions above), we can run experiments in these environments as follows:

Matrix games:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="matrixgames:penalty-100-nostate-v0"
```

LBF:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

RWARE:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2"
```

MPE:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
```
Note that for the MPE environments tag (predator-prey) and adversary, we provide pre-trained prey and adversary policies. These can be used to control the respective agents to make these tasks fully cooperative (used in the paper) by setting `env_args.pretrained_wrapper="PretrainedTag"` or `env_args.pretrained_wrapper="PretrainedAdversary"`.

SMAC:
```sh
python src/main.py --config=qmix --env-config=sc2 with env_args.map_name="3s5z"
```

Below, we provide the base environment and key / map name for all the environments evaluated in the "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks":

- Matrix games: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - Climbing: `matrixgames:climbing-nostate-v0`
  - Penalty $k=0$: `matrixgames:penalty-0-nostate-v0`
  - Penalty $k=-25$: `matrixgames:penalty-25-nostate-v0`
  - Penalty $k=-50$: `matrixgames:penalty-50-nostate-v0`
  - Penalty $k=-75$: `matrixgames:penalty-75-nostate-v0`
  - Penalty $k=-100$: `matrixgames:penalty-100-nostate-v0`
- LBF: all with `--env-config=gymma with env_args.time_limit=50 env_args.key="..."`
  - 8x8-2p-2f-coop: `lbforaging:Foraging-8x8-2p-2f-coop-v3`
  - 8x8-2p-2f-2s-coop: `lbforaging:Foraging-2s-8x8-2p-2f-coop-v3`
  - 10x10-3p-3f: `lbforaging:Foraging-10x10-3p-3f-v3`
  - 10x10-3p-3f-2s: `lbforaging:Foraging-2s-10x10-3p-3f-v3`
  - 15x15-3p-5f: `lbforaging:Foraging-15x15-3p-5f-v3`
  - 15x15-4p-3f: `lbforaging:Foraging-15x15-4p-3f-v3`
  - 15x15-4p-5f: `lbforaging:Foraging-15x15-4p-5f-v3`
- RWARE: all with `--env-config=gymma with env_args.time_limit=500 env_args.key="..."`
  - tiny 2p: `rware:rware-tiny-2ag-v2`
  - tiny 4p: `rware:rware-tiny-4ag-v2`
  - small 4p: `rware:rware-small-4ag-v2`
- MPE: all with `--env-config=gymma with env_args.time_limit=25 env_args.key="..."`
  - simple speaker listener: `pz-mpe-simple-speaker-listener-v4`
  - simple spread: `pz-mpe-simple-spread-v3`
  - simple adversary: `pz-mpe-simple-adversary-v3` with additional `env_args.pretrained_wrapper="PretrainedAdversary"`
  - simple tag: `pz-mpe-simple-tag-v3` with additional `env_args.pretrained_wrapper="PretrainedTag"`
- SMAC: all with `--env-config=sc2 with env_args.map_name="..."`
  - 2s_vs_1sc: `2s_vs_1sc`
  - 3s5z: `3s5z`
  - corridor: `corridor`
  - MMM2: `MMM2`
  - 3s_vs_5z: `3s_vs_5z`
  
## Experiments in SMACv2 and SMAClite

EPyMARL now supports the new SMACv2 and SMAClite environments. We provide wrappers to integrate these environments into the Gymnasium interface of EPyMARL. To run experiments in these environments, you can use the following exemplary commands:

SMACv2:
```sh
python src/main.py --config=qmix --env-config=sc2v2 with env_args.map_name="protoss_5_vs_5"
```
We provide prepared configs for a range of SMACv2 scenarios, as described in the [SMACv2 repository](https://github.com/oxwhirl/smacv2), under `src/config/envs/smacv2_configs`. These can be run by providing the name of the config file as the `env_args.map_name` argument. To define a new scenario, you can create a new config file in the same format as the provided ones and provide its name as the `env_args.map_name` argument.

SMAClite:
```sh
python src/main.py --config=qmix --env-config=smaclite with env_args.time_limit=150 env_args.map_name="MMM"
```
By default, SMAClite uses a numpy implementation of the RVO2 library for collision avoidance. To instead use a faster optimised C++ RVO2 library, follow the instructions of [this repo](https://github.com/micadam/SMAClite-Python-RVO2) and provide the additional argument `env_args.use_cpp_rvo2=True`.

## Registering and Running Experiments in Custom Environments

EPyMARL supports environments that have been registered with Gymnasium. If you would like to use any other Gymnasium environment, you can do so by using the `gymma` environment with the `env_args.key` argument being provided with the registration ID of the environment. Environments can either provide a single scalar reward to run common reward experiments (`common_reward=True`), or should provide one environment per agent to run experiments with individual rewards (`common_reward=False`) or with common rewards using some reward scalarisation (see [documentation](#support-for-training-in-environments-with-individual-rewards-for-all-agents) for more details). 

To register a custom environment with Gymnasium, use the template below:
```python
from gymnasium import register

register(
  id="my-environment-v1",                         # Environment ID.
  entry_point="myenv.environment:MyEnvironment",  # The entry point for the environment class
  kwargs={
            ...                                   # Arguments that go to MyEnvironment's __init__ function.
        },
    )
```

After, you can run an experiment in this environment using the following command:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="myenv:my-environment-v1"
```
assuming that the environment is registered with the ID `my-environment-v1` in the installed library `myenv`.

# Experiment Configurations

EPyMARL defines yaml configuration files for algorithms and environments under `src/config`. `src/config/default.yaml` defines default values for a range of configuration options, including experiment information (`t_max` for number of timesteps of training etc.) and algorithm hyperparameters.

Further environment configs (provided to the main script via `--env-config=...`) can be found in `src/config/envs`. Algorithm configs specifying algorithms and their hyperparameters (provided to the main script via `--config=...`) can be found in `src/config/algs`. To change hyperparameters or define a new algorithm, you can modify these yaml config files or create new ones.

# Run a hyperparameter search

We include a script named `search.py` which reads a search configuration file (e.g. the included `search.config.example.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
```shell
python search.py run --config=search.config.example.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```shell
python search.py run --config=search.config.example.yaml --seeds 5 single 1
```
where the 1 is an index to the particular hyperparameter configuration and can take values from 1 to the number of different combinations.

# Logging

By default, EPyMARL will use sacred to log results and models to the `results` directory. These logs include configuration files, a json of all metrics, a txt file of all outputs and more. Additionally, EPyMARL can log data to tensorboard files by setting `use_tensorboard: True` in the yaml config. We also added support to log data to [weights and biases (W&B)](https://wandb.ai/) with instructions below.

## Weights and Biases

First, make sure to install W&B and follow their instructions to authenticate and setup your W&B library (see the [quickstart guide](https://docs.wandb.ai/quickstart) for more details).

To tell EPyMARL to log data to W&B, you then need to specify the following parameters in [your configuration](#experiment-configurations):
```yaml
use_wandb: True # Log results to W&B
wandb_team: null # W&B team name
wandb_project: null # W&B project name
```
to specify the team and project you wish to log to within your account, and set `use_wandb=True`. By default, we log all W&B runs in "offline" mode, i.e. the data will only be stored locally and can be uploaded to your W&B account via `wandb sync ...`. To directly log runs online, please specify `wandb_mode="online"` within the config.

We also support logging all stored models directly to W&B so you can download and inspect these from the W&B online dashboard. To do so, use the following config parameters:
```yaml
wandb_save_model: True # Save models to W&B (only done if use_wandb is True and save_model is True)
save_model: True # Save the models to disk
save_model_interval: 50000
```
Note that models are only saved in general if `save_model=True` and to further log them to W&B you need to specify `use_wandb`, `wandb_team`, `wandb_project`, and `wandb_save_model=True`.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` and `load_step` parameters. `checkpoint_path` should point to a directory stored for a run by epymarl as stated above. The pointed-to directory should contain sub-directories for various timesteps at which checkpoints were stored. If `load_step` is not provided (by default `load_step=0`) then the last checkpoint of the pointed-to run is loaded. Otherwise the checkpoint of the closest timestep to `load_step` will be loaded. After loading, the learning will proceed from the corresponding timestep.

To only evaluate loaded models without any training, set the `checkpoint_path` and `load_step` parameters accordingly for the loading, and additionally set `evaluate=True`. Then, the loaded checkpoint will be evaluated for `test_nepisode` episodes before terminating the run.

# Plotting

The plotting script provided as `plot_results.py` supports plotting of any logged metric, can apply simple window-smoothing, aggregates results across multiple runs of the same algorithm, and can filter which results to plot based on algorithm and environment names.

If multiple configs of the same algorithm exist within the loaded data and you only want to plot the best config per algorithm, then add the `--best_per_alg` argument! If this argument is not set, the script will visualise all configs of each (filtered) algorithm and show the values of the hyperparameter config that differ across all present configs in the legend.

# Citing YOLO-MARL

