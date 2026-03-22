"""
Load and save model parameters from/to configuration files (YAML or JSON).
Users can edit a config file instead of changing Python code.
"""

from pathlib import Path
from typing import Any, Dict
from .params import GrowthParams


def _params_to_dict(p: GrowthParams) -> Dict[str, Any]:
    """Convert GrowthParams to a plain dict for serialization."""
    return {
        "m_young": p.m_young,
        "s_cw": p.s_cw,
        "alph": p.alph,
        "etha": p.etha,
        "thresh": p.thresh,
        "Lw": getattr(p, "Lw", 40.0),
        "t_cell_cycle": p.t_cell_cycle,
        "t_cell_elongation": p.t_cell_elongation,
        "div_threshold": list(p.div_threshold),
        "min_div_cell_size": list(p.min_div_cell_size) if p.min_div_cell_size else None,
        "max_elong_cell_size": list(p.max_elong_cell_size) if p.max_elong_cell_size else None,
        "n_cell_files": p.n_cell_files,
        "cell_width": list(p.cell_width),
        "n_cells_per_file": list(p.n_cells_per_file),
        "max_cell_length": list(p.max_cell_length) if getattr(p, "max_cell_length", None) else None,
        "elong_border": list(p.elong_border) if getattr(p, "elong_border", None) else None,
        "s_division": list(p.s_division) if p.s_division else None,
        "smax": list(p.smax) if p.smax else None,
        "alpha_min": p.alpha_min,
        "alpha_max": p.alpha_max,
        "alpha_mu": p.alpha_mu,
        "alpha_sigma": p.alpha_sigma,
        "koef": p.koef,
    }


def _dict_to_params(d: Dict[str, Any]) -> GrowthParams:
    """Build GrowthParams from a dict (e.g. from YAML/JSON). Only known keys are used."""
    from dataclasses import fields
    allowed = {f.name for f in fields(GrowthParams)}
    kwargs = {}
    for k, v in d.items():
        if k not in allowed:
            continue
        if v is not None and isinstance(v, list):
            v = list(v)
        kwargs[k] = v
    return GrowthParams(**kwargs)


def load_params(path: str) -> GrowthParams:
    """
    Load GrowthParams from a YAML or JSON file.
    Path can be .yaml, .yml, or .json; format is auto-detected by extension.
    Only specify the parameters you want to override; others use defaults.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config. Install with: pip install pyyaml")
        data = yaml.safe_load(text)
    elif suffix == ".json":
        import json
        data = json.loads(text)
    else:
        try:
            import yaml
            data = yaml.safe_load(text)
        except Exception:
            import json
            data = json.loads(text)
    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a single mapping (dict).")
    return _dict_to_params(data)


def save_params(params: GrowthParams, path: str) -> None:
    """
    Save GrowthParams to a YAML or JSON file.
    Format is chosen by extension (.yaml/.yml → YAML, .json → JSON).
    """
    path = Path(path)
    data = _params_to_dict(params)
    suffix = path.suffix.lower()

    with open(path, "w", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config. Install with: pip install pyyaml")
        elif suffix == ".json":
            import json
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            import json
            json.dump(data, f, indent=2, ensure_ascii=False)


def params_to_dict(params: GrowthParams) -> Dict[str, Any]:
    """Export params to a plain dict (e.g. for custom serialization or logging)."""
    return _params_to_dict(params)


def params_from_dict(d: Dict[str, Any]) -> GrowthParams:
    """Build GrowthParams from a plain dict (e.g. from your own config structure)."""
    return _dict_to_params(d)
