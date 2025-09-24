import importlib
from pathlib import Path

from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

# Import the config system
from clinical_ts.config import create_default_config

def _string_to_class(_target_):
    """Convert string target to class object."""
    if len(_target_.split(".")) == 1:  # assume global namespace
        cls_ = globals()[_target_]
    else:
        mod_ = importlib.import_module(".".join(_target_.split(".")[:-1]))
        cls_ = getattr(mod_, _target_.split(".")[-1])
    return cls_


def load_model_from_config(config_name="/absolute/path/to/config.yaml", overrides=None):
    """
    Load and instantiate a model from a full-path YAML config.
    
    Args:
        config_name: Full path to YAML config file (including .yaml extension)
        overrides: List of config overrides in dotlist format (e.g., ["trainer.gpus=0", "base.batch_size=16"])
    
    Returns:
        model: Instantiated model ready for inference or training
        hparams: Full configuration object
    """
    if overrides is None:
        overrides = []
    
    # Initialize the config store for schema (useful when composing structured configs)
    cs = create_default_config()

    config_path = Path(config_name)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_name}")

    raw_cfg = OmegaConf.load(config_name)

    # If the file is a Hydra config (contains defaults), compose it properly
    if isinstance(raw_cfg, dict) or (hasattr(raw_cfg, 'get') and raw_cfg.get('defaults') is not None):
        config_dir = str(config_path.parent)
        config_base = config_path.stem
        # Ensure Hydra can find the project's default config groups (e.g., data/, ts/, loss/)
        project_conf_root = (Path(__file__).resolve().parents[2] / "conf").as_posix()
        searchpath_override = f"hydra.searchpath=[{project_conf_root}]"
        composed_overrides = [searchpath_override] + (overrides or [])
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            hparams = compose(config_name=config_base, overrides=composed_overrides)
    else:
        # Plain YAML without Hydra defaults
        hparams = raw_cfg
        if overrides:
            hparams = OmegaConf.merge(hparams, OmegaConf.from_dotlist(overrides))
    
    print("Loaded config:")
    print(OmegaConf.to_yaml(hparams))
    
    # Create the model
    classname = _string_to_class(hparams.task.mainclass)
    model = classname(hparams)

    # Load pretrained weights if specified
    if hparams.trainer.pretrained != "":
        print("Loading pretrained weights from", hparams.trainer.pretrained, "pretrained_keys_filter:", hparams.trainer.pretrained_keys_filter)
        model.load_weights_from_checkpoint(hparams.trainer.pretrained, hparams.trainer.pretrained_keys_filter)
    
    return model, hparams
