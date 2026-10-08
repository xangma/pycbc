"""JAX backend support for parameter transformations and constraints."""

import ast

from pycbc import conversions, cosmology
from pycbc.io import record
from pycbc.types.backend import coerce_jax_values, jax_module_for as _jax_module_for

_EXPRESSION_UNSUPPORTED = object()


def scalar_tov_reference(function, transform, maps):
    """Replay the original scalar TOV transform for each new JAX vector row."""
    import numpy

    names = [transform.mass_param]
    if hasattr(transform, "get_eos"):
        names.append("eos")
    if transform.redshift_mass and transform.distance is None:
        names.append("distance")
    values = numpy.broadcast_arrays(*(numpy.asarray(maps[name]) for name in names))
    shape = values[0].shape
    if not shape:
        return function(transform, maps)
    results = []
    for index in numpy.ndindex(shape):
        row = maps.copy()
        row.update((name, value[index]) for name, value in zip(names, values))
        results.append(function(transform, row)[transform.lambda_param])
    # The original scalar interpolator returns double precision, including
    # its zero/nan values. Preserve that dtype for an empty vector too.
    result = numpy.asarray(results, dtype=float).reshape(shape)
    return transform.format_output(maps, {transform.lambda_param: result})


def scalar_logit_jacobian_reference(function, transform, maps):
    """Replay the original scalar Jacobian for the new JAX vector API."""
    import numpy

    values = numpy.asarray(maps[transform.inputvar])
    if not values.ndim:
        return function(transform, maps)
    results = []
    for value in values.flat:
        row = maps.copy()
        row[transform.inputvar] = value
        results.append(function(transform, row))
    if results:
        return numpy.asarray(results).reshape(values.shape)
    row = maps.copy()
    row[transform.inputvar] = numpy.asarray(
        (transform._a + transform._b) / 2, dtype=values.dtype)[()]
    dtype = numpy.asarray(function(transform, row)).dtype
    return numpy.empty(values.shape, dtype=dtype)

_EXPRESSION_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.keyword,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


_EXPRESSION_CONVERSIONS = {
    "det_tc",
    "eta_from_mass1_mass2",
    "eta_from_q",
    "invq_from_mass1_mass2",
    "mass1_from_mchirp_eta",
    "mass1_from_mchirp_q",
    "mass1_from_mtotal_eta",
    "mass1_from_mtotal_q",
    "mass2_from_mchirp_eta",
    "mass2_from_mchirp_q",
    "mass2_from_mtotal_eta",
    "mass2_from_mtotal_q",
    "mchirp_from_mass1_mass2",
    "mtotal_from_mass1_mass2",
    "mtotal_from_mchirp_eta",
    "primary_mass",
    "q_from_mass1_mass2",
    "secondary_mass",
}

_EXPRESSION_KEYWORDS = {
    "det_tc": {"ref_frame", "relative"},
}

_EXPRESSION_COSMOLOGY = {
    "distance_from_comoving_volume",
    "redshift",
    "redshift_from_comoving_volume",
}


def jax_expression_context(maps, input_names):
    """Build the audited expression library for raw JAX array inputs."""
    if not isinstance(maps, dict):
        return None

    reference = next(
        (
            maps[name]
            for name in input_names
            if name in maps and _jax_module_for(maps[name]) is not None
        ),
        None,
    )
    if reference is None:
        return None

    import jax.numpy as jnp

    jax_fn_map = {
        "abs": jnp.abs,
        "absolute": jnp.abs,
        "acos": jnp.arccos,
        "acosh": jnp.arccosh,
        "asin": jnp.arcsin,
        "asinh": jnp.arcsinh,
        "atan": jnp.arctan,
        "atan2": jnp.arctan2,
        "arctan2": jnp.arctan2,
        "ceil": jnp.ceil,
        "cos": jnp.cos,
        "cosh": jnp.cosh,
        "exp": jnp.exp,
        "expm1": jnp.expm1,
        "floor": jnp.floor,
        "hypot": jnp.hypot,
        "log": jnp.log,
        "log10": jnp.log10,
        "log1p": jnp.log1p,
        "log2": jnp.log2,
        "maximum": jnp.maximum,
        "minimum": jnp.minimum,
        "sign": jnp.sign,
        "sin": jnp.sin,
        "sinh": jnp.sinh,
        "sqrt": jnp.sqrt,
        "tan": jnp.tan,
        "tanh": jnp.tanh,
    }

    def tensor_function(fn):
        def evaluate(*args):
            return fn(*(jnp.asarray(arg) for arg in args))

        return evaluate

    context = {
        name: value
        for name, value in record._numpy_function_lib.items()
        if isinstance(value, float)
    }
    context.update(
        {name: tensor_function(fn) for name, fn in jax_fn_map.items()}
    )
    context.update(
        {name: getattr(conversions, name) for name in _EXPRESSION_CONVERSIONS}
    )
    context.update(
        {name: getattr(cosmology, name) for name in _EXPRESSION_COSMOLOGY}
    )
    context.update({name: maps[name] for name in input_names if name in maps})
    return context


def evaluate_raw_expression(
    expression,
    maps,
    input_names,
    code_cache,
    allowed_nodes=_EXPRESSION_NODES,
):
    """Evaluate a structurally audited expression on raw JAX arrays.

    Unknown names and unsupported syntax return a sentinel so callers can
    retain their established FieldArray/NumPy fallback.
    """
    context = jax_expression_context(maps, input_names)
    if context is None:
        return _EXPRESSION_UNSUPPORTED

    cached = code_cache.get(expression)
    if cached is None:
        tree = ast.parse(expression, mode="eval")
        string_constants = set()
        keyword_nodes = set()
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Name):
                code_cache[expression] = False
                return _EXPRESSION_UNSUPPORTED
            allowed_keywords = _EXPRESSION_KEYWORDS.get(call.func.id)
            if call.keywords and allowed_keywords is None:
                code_cache[expression] = False
                return _EXPRESSION_UNSUPPORTED
            for keyword in call.keywords:
                if keyword.arg not in allowed_keywords:
                    code_cache[expression] = False
                    return _EXPRESSION_UNSUPPORTED
                keyword_nodes.add(id(keyword))
            if call.func.id in _EXPRESSION_KEYWORDS:
                string_constants.update(
                    id(value)
                    for value in (
                        *call.args,
                        *(keyword.value for keyword in call.keywords),
                    )
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
        if any(
            not isinstance(node, allowed_nodes)
            or (isinstance(node, ast.keyword) and id(node) not in keyword_nodes)
            or (
                isinstance(node, ast.Constant)
                and not (
                    isinstance(node.value, (int, float))
                    or (isinstance(node.value, str) and id(node) in string_constants)
                )
            )
            or (isinstance(node, ast.Compare) and len(node.ops) != 1)
            for node in ast.walk(tree)
        ):
            code_cache[expression] = False
            return _EXPRESSION_UNSUPPORTED
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        cached = (compile(tree, "<string>", "eval"), names, calls)
        code_cache[expression] = cached
    elif cached is False:
        return _EXPRESSION_UNSUPPORTED

    code, names, calls = cached
    if not names.issubset(context) or any(
        not callable(context[name]) for name in calls
    ):
        return _EXPRESSION_UNSUPPORTED
    return eval(code, {"__builtins__": {}}, context)


def components_from_mass_order_jax(primary, secondary, mass1, mass2, array_mode):
    """Preserve the original scalar tie and array grouping conventions."""
    import jax.numpy as jnp

    _, (p, s, m1, m2) = coerce_jax_values(primary, secondary, mass1, mass2)
    if not array_mode:
        primary_is_one = m1 > m2
        return (jnp.where(primary_is_one, p, s),
                jnp.where(primary_is_one, s, p))
    p, s, m1, m2 = (value.reshape(-1)
                    for value in jnp.broadcast_arrays(p, s, m1, m2))
    primary_is_one = m1 >= m2
    first = jnp.where(primary_is_one, p, s)
    second = jnp.where(primary_is_one, s, p)
    # Boolean indexing in the original NumPy body concatenates these two
    # groups in different orders for component one and component two.
    first_order = jnp.argsort(~primary_is_one, stable=True)
    second_order = jnp.argsort(primary_is_one, stable=True)
    return first[first_order], second[second_order]


def interp_lambda_from_tov_jax(m_src, mass_data, lambda_data):
    """Interpolate Lambda from TOV table without leaving JAX."""
    import jax.numpy as jnp

    if jnp.iscomplexobj(m_src):
        raise TypeError("mass must be real")
    values = jnp.asarray(m_src, dtype=mass_data.dtype)
    shape = values.shape
    values = values.reshape(-1)

    if mass_data.size == 1:
        result = jnp.where(
            values > mass_data[0],
            jnp.zeros_like(values),
            lambda_data[0],
        )
        return result.reshape(shape)

    result = jnp.interp(
        values,
        mass_data,
        lambda_data,
        left=lambda_data[0],
        right=jnp.asarray(0.0, dtype=lambda_data.dtype),
    )
    return result.reshape(shape)
