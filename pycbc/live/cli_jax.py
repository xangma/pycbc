"""Device scheduling and output boundaries for the JAX Live search."""

import logging

import jax
import jax.numpy as jnp

from pycbc.live.cli import LiveEventManager


class JAXLiveEventManager(LiveEventManager):
    """Share native event handling while retaining numerical device columns."""

    def commit_results(self, results):
        logging.info('Committing triggers')
        self._jax_result_transport.commit(results)

    def gather_results(self):
        if self.rank != 0:
            raise RuntimeError('Not root process')
        logging.info('Gathering triggers')
        received = [value for value in self._jax_result_transport.gather()
                    if value is not None]
        results = [value[0] for value in received]
        combined = {}
        for ifo in results[0]:
            if any(result[ifo] is False for result in results):
                continue
            combined[ifo] = {}
            for key in results[0][ifo]:
                columns = [result[ifo][key] for result in results]
                from pycbc.events.live_pipeline_jax import (
                    live_result_column_backend_jax)
                backend = live_result_column_backend_jax(key, columns[0])
                values = backend.concatenate(
                    [backend.asarray(column) for column in columns])
                if backend is jnp:
                    from pycbc import scheme
                    values = jax.device_put(values, scheme.mgr.state.jax_device)
                combined[ifo][key] = values
        return combined, received[0][1]

    @staticmethod
    def followup_processing_options(args):
        """Preserve the selected device and diagnostic calculation routes."""
        options = f'--processing-scheme {args.processing_scheme} '
        for name in ('jax_chisq_mode', 'jax_highpass_mode'):
            value = getattr(args, name, None)
            if value is not None:
                options += f'--{name.replace("_", "-")} {value} '
        operations = getattr(args, 'jax_reference_operations', None)
        if operations:
            selection = (operations if isinstance(operations, str)
                         else ','.join(sorted(operations)))
            options += '--jax-reference-operations ' + selection + ' '
        return options

    @staticmethod
    def serialize_value(value):
        if isinstance(value, jax.Array):
            value = jax.device_get(value)
        return LiveEventManager.serialize_value(value)

    def dump(self, results, name, **kwargs):
        from pycbc.events.live_output_jax import dump_live_results_jax

        return dump_live_results_jax(self, results, name, **kwargs)
