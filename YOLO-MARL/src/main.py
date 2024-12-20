try:
    # until python 3.10
    from collections import Mapping
except:
    # from python 3.10
    from collections.abc import Mapping
from copy import deepcopy
import os
from os.path import dirname, abspath
import sys
import yaml

import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th

from utils.logging import get_logger
from run import run

SETTINGS["CAPTURE_MODE"] = (
    "fd"  # set to "no" if you want to see stdout/stderr in console
)
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    th.cuda.manual_seed_all(config["seed"])
    config["env_args"]["seed"] = config["seed"]
    print(config["seed"])
    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)
    

def update_config_dict(map_name, config_dict):
    map_args = map_name.split("-")
    env_name = ""
    if map_name.startswith("pz-mpe"):
        N = map_args[-1]
        if N.isdigit():
            N = int(N)
        else:
            raise ValueError("N is not a digit")
        config_dict["env_args"]["N"] = N
        map_name = "-".join(map_args[:-1])
        env_name = "_".join(map_args[1: ])
        config_dict["env_name"] = env_name

    elif map_name.startswith("lbforaging:Foraging"):
        env_name = "lbf_" + "_".join(map_args[2:-1])
        config_dict["env_name"] = env_name
    elif map_name.startswith("rware:rware"):
        env_name = "rware_" + "_".join(map_args[1:])
        config_dict["env_name"] = env_name
    else:
        raise ValueError("Unknown map_name or environment not supported")
    return config_dict, map_name


if __name__ == "__main__":
    params = deepcopy(sys.argv)
    th.set_num_threads(1)

    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")

    np.random.seed(config_dict["seed"])
    th.manual_seed(config_dict["seed"])
    th.cuda.manual_seed_all(config_dict["seed"])
    print("seed")
    print(config_dict["seed"])

    config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    config_dict, map_name = update_config_dict(map_name, config_dict)

    for i, param in enumerate(params):
        if param.startswith("env_args.map_name"):
            del params[i]
            params.append(f"env_args.map_name={map_name}")
            break
        elif param.startswith("env_args.key"):
            del params[i]
            params.append(f"env_args.key={map_name}")
            break

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(
        results_path, f"sacred/{config_dict['name']}/{map_name}"
    )

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # ex.observers.append(MongoObserver())

    ex.run_commandline(params)
