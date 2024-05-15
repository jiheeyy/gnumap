#!/bin/bash

# Run the Python script with the specified dataset
python experiment_main.py --name_dataset Cora --filename Cora --jm GNUMAP2 VGAE GRACE
python experiment_main.py --name_dataset road_cal --filename road_cal --jm GNUMAP2 VGAE GRACE
# python experiment_main.py --name_dataset Swissroll --filename Swissroll --jm GNUMAP2 VGAE GRACE
# python experiment_main.py --name_dataset Swissroll --filename Swissroll_a10 --a 10 --jm GNUMAP2 VGAE GRACE
# python experiment_main.py --name_dataset Sphere --filename Sphere --jm GNUMAP2 VGAE GRACE

# In terminal
# chmod +x exp.sh
# bash exp.sh