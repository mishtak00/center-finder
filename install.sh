#!/bin/sh
mkdir data && mkdir output
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
