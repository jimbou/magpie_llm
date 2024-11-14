# Quick Start

## Clone the repository

    git clone https://github.com/bloa/magpie.git
    cd magpie


## Run an example

Run a short 60-second local search:

    python3 magpie local_search --scenario examples/triangle-c/_magpie/scenario_slow.txt

Show the diff corresponding to a given patch:

    python3 magpie show_patch --scenario examples/triangle-cpp/_magpie/scenario_slow.txt --patch "SrcmlStmtDeletion(('triangle.cpp.xml', 'stmt', 3))"

Simplify a given patch by removing unnecessary bloat:

    python3 magpie minify_patch --scenario examples/triangle-py/_magpie/scenario_slow.txt --patch "SrcmlStmtInsertion(('triangle.py.xml', '_inter_block', 35), ('triangle.py.xml', 'stmt', 22)) | SrcmlStmtReplacement(('triangle.py.xml', 'stmt', 7), ('triangle.py.xml', 'stmt', 1)) | SrcmlStmtInsertion(('triangle.py.xml', '_inter_block', 24), ('triangle.py.xml', 'stmt', 6)) | SrcmlStmtInsertion(('triangle.py.xml', '_inter_block', 15), ('triangle.py.xml', 'stmt', 21)) | SrcmlStmtDeletion(('triangle.py.xml', 'stmt', 8))"


## Run tests (optional)

You will need [Pytest](https://docs.pytest.org/) to run tests.

    pytest
