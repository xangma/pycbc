"""Adversarial replay no-op: qualification must detect unwritten outputs."""
from pathlib import Path
import runpy
import sys

import torch

torch.cuda.CUDAGraph.replay = lambda _graph: None
path = Path(__file__).with_name('qualify_production_graph_v2.py')
sys.argv[0] = str(path)
runpy.run_path(str(path), run_name='__main__')
