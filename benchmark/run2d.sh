#!/bin/sh
source venv/bin/activate
export PYTHONPATH="src"
python benchmark/naive_2d.py

# make copy
cp -r benchmark/figures /home/vlaurent/Project/transverse/active_learning_regression/632d8372c695eeb6792b519a
