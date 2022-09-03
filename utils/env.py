# Copyright (c) Facebook, Inc. and its affiliates.
import subprocess
import importlib
import importlib.util
import logging
import os
import sys

import random
import re
import PIL
import numpy as np

from datetime import datetime
from collections import defaultdict

import tabulate
import torch
import torchvision


__all__ = ["seed_all_rng"]


TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
DOC_BUILDING = os.getenv("_DOC_BUILDING", False)  # set in docs/conf.py


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    """
    Configurations for some libraries.
    """
    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    disable_cv2 = int(os.environ.get("DETECTRON2_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable opencl in opencv since its interaction with cuda often has negative effects
        # This envvar is supported after OpenCV 3.4.0
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ModuleNotFoundError:
            # Other types of ImportError, if happened, should not be ignored.
            # Because a failed opencv import could mess up address space
            # https://github.com/skvark/opencv-python/issues/381
            pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))

    # fmt: off
    assert get_version(torch) >= (1, 4), "Requires torch>=1.4"
    import fvcore
    assert get_version(fvcore, 3) >= (0, 1, 2), "Requires fvcore>=0.1.2"
    import yaml
    assert get_version(yaml) >= (5, 1), "Requires pyyaml>=5.1"
    # fmt: on


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $DETECTRON2_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("DETECTRON2_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = _import_file("detectron2.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
    module.setup_environment()


def fixup_module_metadata(module_name, namespace, keys=None):
    """
    Fix the __qualname__ of module members to be their exported api name, so
    when they are referenced in docs, sphinx can find them. Reference:
    https://github.com/python-trio/trio/blob/6754c74eacfad9cc5c92d5c24727a2f3b620624e/trio/_util.py#L216-L241
    """
    if not DOC_BUILDING:
        return
    seen_ids = set()

    def fix_one(qualname, name, obj):
        # avoid infinite recursion (relevant when using
        # typing.Generic, for example)
        if id(obj) in seen_ids:
            return
        seen_ids.add(id(obj))

        mod = getattr(obj, "__module__", None)
        if mod is not None and (mod.startswith(module_name) or mod.startswith("fvcore.")):
            obj.__module__ = module_name
            # Modules, unlike everything else in Python, put fully-qualitied
            # names into their __name__ attribute. We check for "." to avoid
            # rewriting these.
            if hasattr(obj, "__name__") and "." not in obj.__name__:
                obj.__name__ = name
                obj.__qualname__ = qualname
            if isinstance(obj, type):
                for attr_name, attr_value in obj.__dict__.items():
                    fix_one(objname + "." + attr_name, attr_name, attr_value)

    if keys is None:
        keys = namespace.keys()
    for objname in keys:
        if not objname.startswith("_"):
            obj = namespace[objname]
            fix_one(objname, objname, obj)


def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def get_env_module():
    var_name = "DETECTRON2_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )
            output = output.decode("utf-8").strip().split("\n")
            arch = []
            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                arch.append(".".join(line))
            arch = sorted(set(arch))
            return ", ".join(arch)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_env_info():
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    torch_version = torch.__version__

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    has_rocm = False
    if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
        has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))  # check-template.yml depends on it
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import detectron2  # noqa
        data.append(("detectron2", detectron2.__version__ + " @" + os.path.dirname(detectron2.__file__)))
    except ImportError:
        data.append(("detectron2", "failed to import"))

    try:
        import detectron2._C as _C
        
    except ImportError as e:
        data.append(("detectron2._C", f"not built correctly: {e}"))

        # print system compilers when extension fails to build
        if sys.platform != "win32":  # don't know what to do for windows
            try:
                # this is how torch/utils/cpp_extensions.py choose compiler
                cxx = os.environ.get("CXX", "c++")
                cxx = subprocess.check_output("'{}' --version".format(cxx), shell=True)
                cxx = cxx.decode("utf-8").strip().split("\n")[0]
            except subprocess.SubprocessError:
                cxx = "Not found"
            data.append(("Compiler ($CXX)", cxx))

            if has_cuda and CUDA_HOME is not None:
                try:
                    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                    nvcc = subprocess.check_output("'{}' -V".format(nvcc), shell=True)
                    nvcc = nvcc.decode("utf-8").strip().split("\n")[-1]
                except subprocess.SubprocessError:
                    nvcc = "Not found"
                data.append(("CUDA compiler", nvcc))
        if has_cuda and sys.platform != "win32":
            try:
                so_file = importlib.util.find_spec("detectron2._C").origin
            except ImportError:
                pass
            else:
                data.append(("detectron2 arch flags", detect_compute_compatibility(CUDA_HOME, so_file)))
                
    else:
        # print compilers that are used to build extension
        data.append(("Compiler", _C.get_compiler_version()))
        data.append(("CUDA compiler", _C.get_cuda_version()))  # cuda or hip
        if has_cuda and getattr(_C, "has_cuda", lambda: True)():
            data.append(
                ("detectron2 arch flags", detect_compute_compatibility(CUDA_HOME, _C.__file__))
            )

    data.append(get_env_module())
    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("GPU available", has_gpu))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = ".".join((str(x) for x in torch.cuda.get_device_capability(k)))
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        if has_rocm:
            msg = " - invalid!" if not (ROCM_HOME and os.path.isdir(ROCM_HOME)) else ""
            data.append(("ROCM_HOME", str(ROCM_HOME) + msg))
        else:
            msg = " - invalid!" if not (CUDA_HOME and os.path.isdir(CUDA_HOME)) else ""
            data.append(("CUDA_HOME", str(CUDA_HOME) + msg))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    try:
        data.append(("torchvision", str(torchvision.__version__) + " @" + os.path.dirname(torchvision.__file__),))
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except ImportError:
                data.append(("torchvision._C", "Not found"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import fvcore
        data.append(("fvcore", fvcore.__version__))
    except ImportError:
        pass

    try:
        import iopath
        data.append(("iopath", iopath.__version__))
    except (ImportError, AttributeError):
        pass

    try:
        import cv2
        data.append(("cv2", cv2.__version__))
    except ImportError:
        data.append(("cv2", "Not found"))
        
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str



