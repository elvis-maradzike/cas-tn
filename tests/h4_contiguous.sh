#!/bin/bash
HOME=/home/parallels
g++-9 -g `$HOME/.exatn/bin/exatn-config --cxxflags` `$HOME/.exatn/bin/exatn-config --includes` -c h4_contiguous.cpp -o h4_contiguous.o 
g++-9 h4_contiguous.o `$HOME/.exatn/bin/exatn-config --libs` -o h4_contiguous.x 

