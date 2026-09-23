"""Execute the review notebooks and export HTML with embedded plot outputs."""
import argparse
import base64
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default='python3')
    parser.add_argument('--previews', type=Path)
    parser.add_argument('names', nargs='*', default=[
        'pi', 'time_indices', 'phase', 'accumulation',
        'arithmetic_real_data', 'real_data', 'allocation'])
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    exporter = HTMLExporter(template_name='classic')
    for name in args.names:
        path = folder / (name + '.ipynb')
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=600,
                                kernel_name=args.kernel,
                                resources={'metadata': {'path': str(folder)}})
        client.execute()
        nbformat.write(notebook, path)
        html, _ = exporter.from_notebook_node(notebook)
        path.with_suffix('.html').write_text(
            '\n'.join(line.rstrip() for line in html.splitlines()) + '\n')
        images = 0
        for cell in notebook.cells:
            for output in cell.get('outputs', []):
                png = output.get('data', {}).get('image/png')
                if png and args.previews:
                    args.previews.mkdir(parents=True, exist_ok=True)
                    image = args.previews / f'{name}-{images}.png'
                    image.write_bytes(base64.b64decode(png))
                    images += 1
        print(f'{name}: executed, exported HTML; {images} plot previews',
              flush=True)


if __name__ == '__main__':
    main()
