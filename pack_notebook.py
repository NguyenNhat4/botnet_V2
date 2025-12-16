import json
import os
import re

# Files to combine in dependency order
FILES = [
    'config.py',
    'utils.py',
    'preprocessing_utils.py',
    'loss.py',
    'model.py',
    'data_loader.py',
    'analyze_csv.py',
    'train.py',
    'evaluate.py'
]

LOCAL_MODULES = [f.replace('.py', '') for f in FILES]

def create_notebook():
    cells = []

    # 1. Install Requirements
    try:
        with open('requirements.txt', 'r') as f:
            reqs = f.read().splitlines()
        req_str = ' '.join([r for r in reqs if r and not r.startswith('#')])
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install requirements\n",
                f"!pip install {req_str}"
            ]
        })
    except FileNotFoundError:
        pass

    # 2. Add files
    for filename in FILES:
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            continue

        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Rename main functions to avoid collisions
        # Use simple word boundary regex to replace 'main(' with 'new_name('
        # This works for both 'def main(' and calls 'main('
        # And it safely ignores 'train_main(' or '__main__'
        if filename == 'train.py':
            content = re.sub(r'\bmain\s*\(', 'train_main(', content)
        elif filename == 'evaluate.py':
            content = re.sub(r'\bmain\s*\(', 'evaluate_main(', content)
        elif filename == 'analyze_csv.py':
            content = re.sub(r'\bmain\s*\(', 'analyze_csv_main(', content)

        lines = content.splitlines(keepends=True)
        processed_lines = []
        skip_block = False

        processed_lines.append(f"# ==========================================\n")
        processed_lines.append(f"# CONTENT FROM: {filename}\n")
        processed_lines.append(f"# ==========================================\n")

        for line in lines:
            # Handle imports
            is_local_import = False
            stripped = line.strip()

            # Regex for "from module import ..." or "import module"
            # We need to be careful not to match substrings inappropriately
            for mod in LOCAL_MODULES:
                # Matches: from config import ...
                if re.match(rf'^from\s+{mod}\s+import', stripped):
                    is_local_import = True
                    break
                # Matches: import config
                if re.match(rf'^import\s+{mod}(\s|$)', stripped):
                    is_local_import = True
                    break

            if is_local_import:
                processed_lines.append(f"# {line.rstrip()}  # Commented out for notebook compatibility\n")
            elif stripped.startswith('if __name__ == "__main__":'):
                 processed_lines.append(f"# {line.rstrip()} # Block disabled for notebook import\n")
                 skip_block = True
            elif skip_block:
                 # Check if the line is indented
                 if line.startswith('    ') or line.startswith('\t') or stripped == '':
                     processed_lines.append(f"# {line.rstrip()}\n")
                 else:
                     # End of block
                     skip_block = False
                     processed_lines.append(line)
            else:
                processed_lines.append(line)

        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "collapsed": True
            },
            "outputs": [],
            "source": processed_lines
        })

    # 3. Add Execution Cells

    # Train
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==========================================\n",
            "# EXECUTION: TRAIN\n",
            "# ==========================================\n",
            "if __name__ == '__main__':\n",
            "    print('Starting Training...')\n",
            "    # Set ispart=False for full training\n",
            "    train_main(ispart=True)\n"
        ]
    })

    # Evaluate
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==========================================\n",
            "# EXECUTION: EVALUATE\n",
            "# ==========================================\n",
            "import sys\n",
            "# Simulate command line arguments. Use empty list for defaults.\n",
            "# To recompute stats: sys.argv = ['evaluate.py', '--recompute-stats']\n",
            "sys.argv = ['evaluate.py'] \n",
            "\n",
            "if __name__ == '__main__':\n",
            "    print('Starting Evaluation...')\n",
            "    evaluate_main()\n"
        ]
    })

    # Analyze CSV Example
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
             "# ==========================================\n",
            "# EXECUTION: ANALYZE CSV (Example)\n",
            "# ==========================================\n",
            "# sys.argv = ['analyze_csv.py', '--file', './CTU-13-Dataset/1/capture20110810.csv']\n",
            "# analyze_csv_main()\n"
        ]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open('botnet_detection_colab.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

    print("Notebook 'botnet_detection_colab.ipynb' created successfully.")

if __name__ == "__main__":
    create_notebook()
