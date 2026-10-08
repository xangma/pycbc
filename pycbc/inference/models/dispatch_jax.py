"""Lazy construction of registered JAX inference implementations."""
from importlib import import_module

def implementation(cls):
    families = {'gaussian_noise': ('gaussian_noise_jax', ('BaseGaussianNoise', 'GaussianNoise')), 'marginalized_gaussian_noise': ('marginalized_gaussian_noise_jax', ('MarginalizedPhaseGaussianNoise',))}
    for native_name, (backend_name, names) in families.items():
        if cls.__module__ != f'{__package__}.{native_name}' or cls.__name__ not in names:
            continue
        native = import_module(f'.{native_name}', __package__)
        if cls is getattr(native, cls.__name__):
            backend = import_module(f'.{backend_name}', __package__)
            return getattr(backend, f'JAX{cls.__name__}')
    return None
