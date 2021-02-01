#!/bin/bash
HOME=/home/parallels
OBJECT_FILE_LOCATION=$HOME/projects/my_cas-tn_branch/cas-tn

g++-9 -g `$HOME/.exatn/bin/exatn-config --cxxflags` `$HOME/.exatn/bin/exatn-config --includes` -c h4.cpp -o h4.o 
g++-9 h4.o $OBJECT_FILE_LOCATION/simulation.o `$HOME/.exatn/bin/exatn-config --libs` -o h4.x 

