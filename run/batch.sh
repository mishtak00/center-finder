#!/bin/bash
#SBATCH --partition=standard --output=output_%a.txt -c 1 -t 8:00:00 --mem-per-cpu=40gb
#SBATCH -a 0-8
python test_scan.py
