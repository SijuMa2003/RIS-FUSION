# Copyright (c) Open-MMLab. All rights reserved.
import io
import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory

import torch
import torchvision
from torch.optim import Optimizer
from torch.nn import functional as F

import mmcv

# MMEngine replaces old MMCV utils: file I/O, distributed utils, etc.
from mmengine.fileio import FileClient
from mmengine.fileio import load as load_file
from mmengine.model import is_model_wrapper as is_module_wrapper
from mmengine.utils import mkdir_or_exist
from mmengine.dist import get_dist_info

ENV_MMCV_HOME = "MMCV_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"


def _get_mmcv_home():
    """Get the MMCV cache directory path, ensure existence."""
    mmcv_home = os.path.expanduser(
        os.getenv(
            ENV_MMCV_HOME,
            os.path.join(
                os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "mmcv"
            ),
        )
    )
    mkdir_or_exist(mmcv_home)
    return mmcv_home


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load a state_dict into a module (wrapper-safe).

    Modified from `torch.nn.Module.load_state_dict`. Default strict=False,
    missing/unexpected keys are ignored unless strict=True.

    Args:
        module (nn.Module): Target module.
        state_dict (OrderedDict): Weights to load.
        strict (bool): Whether keys must match exactly. Default False.
        logger (logging.Logger, optional): Logger for warnings/errors.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(m, prefix=""):
        if is_module_wrapper(m):
            m = m.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        m._load_from_state_dict(
            state_dict, prefix, local_metadata, True,
            all_missing_keys, unexpected_keys, err_msg
        )
        for name, child in m._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break recursion cycle

    missing_keys = [k for k in all_missing_keys if "num_batches_tracked" not in k]
    if unexpected_keys:
        err_msg.append("Unexpected keys: " + ", ".join(unexpected_keys))
    if missing_keys:
        err_msg.append("Missing keys: " + ", ".join(missing_keys))

    if strict:
        rank, _ = get_dist_info()
        if err_msg and rank == 0:
            msg = "State dict mismatch:\n" + "\n".join(err_msg)
            if strict:
                raise RuntimeError(msg)
            elif logger:
                logger.warning(msg)
            else:
                print(msg)


def _load_state_dict_from_url(url, model_dir=None, map_location=None):
    """Wrapper for torch.hub.load_state_dict_from_url with caching."""
    return torch.hub.load_state_dict_from_url(
        url, model_dir=model_dir, map_location=map_location
    )


def load_url_dist(url, model_dir=None, map_location=None):
    """Distributed-safe load from URL. Rank0 downloads, others wait."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        checkpoint = _load_state_dict_from_url(url, model_dir, map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = _load_state_dict_from_url(url, model_dir, map_location)
    return checkpoint


def load_pavimodel_dist(model_path, map_location=None):
    """Distributed-safe load from PAVI modelcloud."""
    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError("Please install pavi to use modelcloud.")

    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        model = modelcloud.get(model_path)
        with TemporaryDirectory() as tmp_dir:
            tmp_file = osp.join(tmp_dir, model.name)
            model.download(tmp_file)
            checkpoint = torch.load(tmp_file, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            model = modelcloud.get(model_path)
            with TemporaryDirectory() as tmp_dir:
                tmp_file = osp.join(tmp_dir, model.name)
                model.download(tmp_file)
                checkpoint = torch.load(tmp_file, map_location=map_location)
    return checkpoint


def load_fileclient_dist(filename, backend, map_location):
    """Distributed-safe load from FileClient (e.g., ceph)."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if backend not in ["ceph"]:
        raise ValueError(f"Unsupported backend: {backend}")
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def get_torchvision_models():
    """Collect torchvision model_urls if available (deprecated in new versions)."""
    model_urls = {}
    if hasattr(torchvision.models, "__path__"):
        for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
            if ispkg:
                continue
            zoo = import_module(f"torchvision.models.{name}")
            if hasattr(zoo, "model_urls"):
                model_urls.update(getattr(zoo, "model_urls"))
    return model_urls


def get_external_models():
    """Return open-mmlab:// mapping from JSON config if available."""
    mmcv_home = _get_mmcv_home()
    default_json = osp.join(mmcv.__path__[0], "model_zoo/open_mmlab.json")
    if not osp.exists(default_json):
        raise FileNotFoundError("open_mmlab.json not found in this MMCV installation.")
    urls = load_file(default_json)
    external_json = osp.join(mmcv_home, "open_mmlab.json")
    if osp.exists(external_json):
        external_urls = load_file(external_json)
        urls.update(external_urls)
    return urls


def get_mmcls_models():
    """Return mmcls:// mapping from JSON config if available."""
    mmcls_json = osp.join(mmcv.__path__[0], "model_zoo/mmcls.json")
    if not osp.exists(mmcls_json):
        raise FileNotFoundError("mmcls.json not found in this MMCV installation.")
    return load_file(mmcls_json)


def get_deprecated_model_names():
    """Return deprecated model mappings if available."""
    path = osp.join(mmcv.__path__[0], "model_zoo/deprecated.json")
    if not osp.exists(path):
        return {}
    return load_file(path)


def _process_mmcls_checkpoint(ckpt):
    """Strip `backbone.` prefix from mmcls checkpoints."""
    state_dict = ckpt["state_dict"]
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_dict[k[9:]] = v
    return {"state_dict": new_dict}


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from various sources (URL, local, pavi, etc.)."""
    if filename.startswith("modelzoo://"):
        warnings.warn('"modelzoo://" is deprecated, use "torchvision://" instead.')
        urls = get_torchvision_models()
        name = filename[11:]
        if name not in urls:
            raise KeyError(f"Cannot resolve torchvision URL for {name}.")
        return load_url_dist(urls[name], map_location=map_location)

    elif filename.startswith("torchvision://"):
        urls = get_torchvision_models()
        name = filename[14:]
        if name not in urls:
            raise KeyError(f"Cannot resolve torchvision URL for {name}.")
        return load_url_dist(urls[name], map_location=map_location)

    elif filename.startswith("open-mmlab://"):
        urls = get_external_models()
        name = filename[13:]
        deprecated = get_deprecated_model_names()
        if name in deprecated:
            warnings.warn(f"{name} is deprecated, use {deprecated[name]} instead.")
            name = deprecated[name]
        if name not in urls:
            raise KeyError(f"open-mmlab:// cannot resolve {name}")
        url = urls[name]
        if url.startswith(("http://", "https://")):
            return load_url_dist(url, map_location=map_location)
        filename = osp.join(_get_mmcv_home(), url)
        if not osp.isfile(filename):
            raise IOError(f"{filename} not found.")
        return torch.load(filename, map_location=map_location)

    elif filename.startswith("mmcls://"):
        urls = get_mmcls_models()
        name = filename[8:]
        if name not in urls:
            raise KeyError(f"mmcls:// cannot resolve {name}")
        ckpt = load_url_dist(urls[name], map_location=map_location)
        return _process_mmcls_checkpoint(ckpt)

    elif filename.startswith(("http://", "https://")):
        return load_url_dist(filename, map_location=map_location)

    elif filename.startswith("pavi://"):
        return load_pavimodel_dist(filename[7:], map_location=map_location)

    elif filename.startswith("s3://"):
        return load_fileclient_dist(filename, backend="ceph", map_location=map_location)

    else:
        if not osp.isfile(filename):
            raise IOError(f"{filename} not found.")
        return torch.load(filename, map_location=map_location)


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load checkpoint into a model, handling prefixes and resizing if needed."""
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint: {filename}")

    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint

    # strip prefixes
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith("backbone."):
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
    if sorted(state_dict.keys())[0].startswith("encoder"):
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}

    # reshape absolute position embedding if needed
    if state_dict.get("absolute_pos_embed") is not None:
        abs_pos = state_dict["absolute_pos_embed"]
        N1, L, C1 = abs_pos.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 == N2 and C1 == C2 and L == H * W:
            state_dict["absolute_pos_embed"] = abs_pos.view(N2, H, W, C2).permute(0, 3, 1, 2)
        else:
            warnings.warn("Mismatch in absolute_pos_embed shape, skipped.")

    # interpolate relative position bias tables if needed
    for key in [k for k in state_dict if "relative_position_bias_table" in k]:
        pre = state_dict[key]
        cur = model.state_dict()[key]
        L1, nH1 = pre.size()
        L2, nH2 = cur.size()
        if nH1 == nH2 and L1 != L2:
            S1, S2 = int(L1 ** 0.5), int(L2 ** 0.5)
            pre_resized = F.interpolate(
                pre.permute(1, 0).view(1, nH1, S1, S1),
                size=(S2, S2), mode="bicubic"
            )
            state_dict[key] = pre_resized.view(nH2, L2).permute(1, 0)
        elif nH1 != nH2:
            warnings.warn(f"Head mismatch for {key}, skipped.")

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy state_dict to CPU."""
    return OrderedDict({k: v.cpu() for k, v in state_dict.items()})


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Save parameters & buffers of a module to destination (wrapper-safe)."""
    if is_module_wrapper(module):
        module = module.module
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix="", keep_vars=False):
    """Recursively get state_dict of a module."""
    if is_module_wrapper(module):
        module = module.module
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = dict(version=getattr(module, "_version", None))
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + ".", keep_vars=keep_vars)
    for hook in getattr(module, "_state_dict_hooks", {}).values():
        result = hook(module, destination, prefix, destination._metadata[prefix[:-1]])
        if result is not None:
            destination = result
    return destination


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file or PAVI modelcloud."""
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f"meta must be dict or None, got {type(meta)}")
    meta.update(mmcv_version=getattr(mmcv, "__version__", "unknown"), time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, "CLASSES") and model.CLASSES is not None:
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {"meta": meta, "state_dict": weights_to_cpu(get_state_dict(model))}

    if isinstance(optimizer, Optimizer):
        checkpoint["optimizer"] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint["optimizer"] = {k: v.state_dict() for k, v in optimizer.items()}

    if filename.startswith("pavi://"):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError("Please install pavi to save to modelcloud.")
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model_node = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model_node = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            tmp_file = osp.join(tmp_dir, model_name)
            with open(tmp_file, "wb") as f:
                torch.save(checkpoint, f); f.flush()
            model_node.create_file(tmp_file, name=model_name)
    else:
        mkdir_or_exist(osp.dirname(filename))
        with open(filename, "wb") as f:
            torch.save(checkpoint, f); f.flush()
