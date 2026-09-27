# JAX Divergence Viewer

The JAX Divergence Viewer is an interactive HTML/JS diagnostic viewer for inspecting numerical comparisons between pristine CPU reference runs and JAX backend runs.

## Building a Report

Build a browsable report from a benchmark suite receipt or phase directory:

```bash
python tools/build_jax_divergence_report.py \
    --suite /path/to/suite_run_or_archive.tar.gz \
    --output /path/to/report_output_dir
```

This compiles a self-contained report directory containing `index.html`, `app.js`, `style.css`, and the summarized JSON comparison data.

## Viewing Locally

Serve the generated directory using any local HTTP server:

```bash
python -m http.server --directory /path/to/report_output_dir 8765
```

Then open `http://127.0.0.1:8765` in your browser.
