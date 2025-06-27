# YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning

This is the official repository for the paper ["YOLO-MARL: You Only LLM Once for Multi-agent Reinforcement Learning"](https://arxiv.org/abs/2410.03997).

The paper is officially accepted by IROS 2025🚀🚀🚀

![Framework Diagram](https://github.com/paulzyzy/YOLO-MARL/blob/master/framework.png)

YOLO-MARL supports both ChatGPT and Claude. Please put your api KEY in the YOLO-MARL/ and name it claude_KEY.txt or openai_KEY.txt.

In YOLO-MARL/src/config/default.yaml, you need to set use_llm by yourself. False for running the MARL baselines and True for running the YOLO-MARL method.

All the baselines we used here are based on [Epymarl](https://github.com/uoe-agents/epymarl). For the denpendencies, you could also refer to Epymarl and then install Openai or Claude dependencies:
```sh
pip install openai #If you want to use ChatGPT for your LLM API
pip install anthropic #If you want to use Claude for your LLM API
```

# Baseline Experiments

For our baselines experiments, you could find all the hyperparameters in ["Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks"](https://arxiv.org/abs/2006.07869) for the Level-Based Foraging (LBF) and Multi-Robot Warehouse (RWARE) environments, and MPE. 

For the YOLO-MARL method, please set the environment in LLM-copilot-RL/LBF/src/prompts/config/config.yaml before you generate LLM planning function.

How to use YOLO-MARL to generate planning function for testing environment:
```sh
python YOLO-MARL/src/prompts/codeGeneration.py
```

For YOLO-MARL training, please set the hyperparameters llm_reward and penalty in YOLO-MARL/src/envs/llm_wrapper.py for mixed_constant.

LBF:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

You can run experiments in these environments as follows:

LBF:
```sh
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

RWARE:
```sh
python src/main.py --config=mappo --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v2"
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

# Citing YOLO-MARL
```
@article{zhuang2024yolomarlllmmultiagentreinforcement,
      title={YOLO-MARL: You Only LLM Once for Multi-agent Reinforcement Learning}, 
      author={Yuan Zhuang and Yi Shen and Zhili Zhang and Yuxiao Chen and Fei Miao},
      year={2024},
      eprint={2410.03997},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2410.03997}, 
}
```
