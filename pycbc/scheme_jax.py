# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""JAX-specific processing-scheme initialization."""

import os


def initialize_jax_cache(jax):
    """Persist short kernels by default without replacing cache overrides.

    JAX reads environment defaults during import. Apply PyCBC's zero-second
    threshold to the live config when its value is still JAX's stock one-second
    default. Explicit environment thresholds and nondefault configured values
    take precedence; JAX does not expose whether its stock value was explicitly
    selected. A configured directory and disabled caching are also retained.
    """
    env_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_dir = getattr(jax.config, "jax_compilation_cache_dir", None)
    if cache_dir is None:
        cache_dir = env_dir if env_dir is not None else os.path.expanduser(
            "~/.cache/pycbc_jax_cache")
    disabled_dir = env_dir is not None and (
        not env_dir or env_dir.strip().lower() in ("0", "false", "none", "off"))
    try:
        if disabled_dir:
            # JAX otherwise treats strings such as "off" as literal paths.
            jax.config.update("jax_enable_compilation_cache", False)
        elif getattr(jax.config, "jax_enable_compilation_cache", True):
            os.makedirs(cache_dir, exist_ok=True)
            if getattr(jax.config,
                       "jax_persistent_cache_min_compile_time_secs", 1.) == 1.:
                threshold = float(os.environ.get(
                    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0"))
                jax.config.update("jax_persistent_cache_min_compile_time_secs",
                                  threshold)
            from jax.experimental.compilation_cache import compilation_cache as cc
            if hasattr(cc, "set_cache_dir"):
                cc.set_cache_dir(cache_dir)
            else:
                cc.initialize_cache(cache_dir)
    except Exception:
        pass
