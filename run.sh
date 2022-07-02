#!/bin/bash

python main.py $1 & \
(
    python -m nltk.downloader punkt; sleep 5;\
    python mincemeatpy/mincemeat.py -p changeme localhost 
)