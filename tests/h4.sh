#!/bin/bash

g++-9 -g `/home/parallels/.exatn/bin/exatn-config --cxxflags` `/home/parallels/.exatn/bin/exatn-config --includes` -c h4.cpp -o h4.o 
g++-9 h4.o /home/parallels/projects/my_cas-tn_branch/cas-tn/simulation.o `/home/parallels/.exatn/bin/exatn-config --libs` -o h4.x 

