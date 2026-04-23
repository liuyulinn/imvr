import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar
import mani_skill
import tyro
import yaml
from omegaconf import OmegaConf

PACKAGE_DIR = Path(__file__).parent.resolve().parent
PACKAGE_ASSETS_DIR = PACKAGE_DIR / "assets"


def _set_maniskill_assets_env():
    if "MANISKILL_ASSETS" in os.environ:
        print(f"MANISKILL_ASSETS={os.environ['MANISKILL_ASSETS']} already exists.")
    else:
        # Try to find the assets directory from the mani_skill package
        package_dir = Path(mani_skill.__file__).parent
        assets_dir = package_dir / "assets"

        if not assets_dir.exists():
            raise FileNotFoundError(f"Expected assets directory not found at {assets_dir}")

        os.environ["MANISKILL_ASSETS"] = str(assets_dir)
        print(f"MANISKILL_ASSETS is set to {assets_dir}")


def _add_assets_env():
    if "ASSET" in os.environ:
        print(f"ASSETS={os.environ.get('ASSETS')} already exists")
    else:
        print(f"ASSETS is set to {PACKAGE_ASSETS_DIR}, you can modify this behavior in miniussd.assets.py")
        os.environ["ASSETS"] = str(PACKAGE_ASSETS_DIR)


def _add_maniskill_cache_env():
    if "MANISKILL_CACHE" in os.environ:
        print(f"MANISKILL_CACHE={os.environ.get('MANISKILL_CACHE')} already exists")
    else:
        mani_skill_cache = os.path.expanduser("~/.maniskill/data")
        print(f"MANISKILL_CACHE is set to {mani_skill_cache}, you can modify this behavior in miniussd.assets.py")
        os.environ["MANISKILL_CACHE"] = str(mani_skill_cache)


_add_assets_env()
_set_maniskill_assets_env()
_add_maniskill_cache_env()


T = TypeVar("T", bound="Config")


class ConfigRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, config_cls: Type[T], name: str):
        def wrapper(cls):
            self._registry[name] = (config_cls, cls)
            return cls

        return wrapper

    def create(self, name: str):  # config_data: Dict[str, Any], *args, **kwargs) -> T:
        if name not in self._registry:
            raise ValueError(f"Config {name} not registered.")

        config_cls, impl_cls = self._registry[name]

        def builder(config: Any, *args, **kwargs):
            return impl_cls(config, *args, **kwargs)

        return builder


def recursive_load_paths(cfg: Any, base_path: str | Path) -> Any:
    if isinstance(cfg, dict):
        return {k: recursive_load_paths(v, base_path) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [recursive_load_paths(item, base_path) for item in cfg]
    elif isinstance(cfg, str) and cfg.endswith((".yaml", ".yml")):
        # Relative to the base file's directory
        sub_path = Path(base_path).parent / cfg
        return OmegaConf.to_container(OmegaConf.load(str(sub_path)), resolve=True)
    else:
        return cfg


@dataclass
class Config:
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        cfg = OmegaConf.create(data)
        return cls.from_omegaconf(cfg)

    @classmethod
    def from_omegaconf(cls: Type[T], cfg: Any) -> T:
        # Merge config then convert to real dataclass instance
        merged = OmegaConf.merge(OmegaConf.structured(cls), cfg)
        return OmegaConf.to_object(merged)
        # return merged

    @classmethod
    def from_yaml(cls: Type[T], path: str) -> T:
        cfg = OmegaConf.load(path)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # recursively load sub-paths
        cfg_dict = recursive_load_paths(cfg_dict, path)
        return cls.from_dict(cfg_dict)

    def to_dict(self):
        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)

    @classmethod
    def from_cc(cls: Type[T]) -> T:
        return tyro.cli(cls)

    def pretty(self, path: str | None = None, env_var_name: str | None = None, json_format: bool = False):
        """Turn the config into a pretty YAML string.
        When path is given, all Path objects will be resolved relative to the path.
        If env_name is given, it will set the relative path into ${env_name:VAR_NAME}/env_name.
        """

        container = OmegaConf.to_container(OmegaConf.structured(self), resolve=False, enum_to_str=True)
        if env_var_name is not None:
            assert path is not None, "env_name requires path to be set"

        if path is not None:

            def do_resolve(p):
                if isinstance(p, Path):
                    rel_path = os.path.relpath(p, path)
                    if env_var_name is not None:
                        rel_path = os.path.join("${oc.env:" + str(env_var_name) + "}", rel_path)

                    return rel_path

                elif isinstance(p, dict):
                    return {k: do_resolve(v) for k, v in p.items()}
                elif isinstance(p, list):
                    return [do_resolve(v) for v in p]
                return p

            container = do_resolve(container)

        if json_format:
            import json

            return json.dumps(container, sort_keys=False)

        return yaml.dump(
            container,
            default_flow_style=None,
            allow_unicode=True,
            sort_keys=False,
        )


@dataclass
class BuilderConfig(Config):
    _type_: str
    config: dict  # raw config data

    def get_cfg(self):
        return self.config  # subclasses will cast this properly
