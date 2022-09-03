import os, json, yaml
import random as rd
import numpy as np


def write_json(path, out1):
    with open(path, "wt", encoding="utf-8") as f:
        json_str = json.dumps(out1, ensure_ascii=False)
        json_str += "\n"
        f.writelines(json_str)


def write_jsonl(path, out):
    with open(path, "wt", encoding="utf-8") as f:
        for out1 in out:
            json_str = json.dumps(out1, ensure_ascii=False)
            json_str += "\n"
            f.writelines(json_str)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = f.read()
        data = json.loads(data.strip())

    return data


def load_jsonl(filepath, toy_data=False, toy_size=4, shuffle=False):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size:
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print("The data shuffled.")
        seed = 1
        rd.Random(seed).shuffle(data)  # fixed

    return data


def load_yaml(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.full_load(f)

    return data


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


"""
From Detectron2
"""
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
from iopath.common.file_io import PathManager as PathManagerBase


PathManager = PathManagerBase()
"""
This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as itintroduces potential conflicts among other libraries.
"""


class Detectron2Handler(PathHandler):
    """
    Resolve anything that's hosted under detectron2's namespace.
    """
    PREFIX = "detectron2://"
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path, **kwargs):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name, **kwargs)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(Detectron2Handler())


