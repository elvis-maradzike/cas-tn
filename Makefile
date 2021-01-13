HOME=/home/parallels
CC=g++-9
RM=rm
CFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --cxxflags` `$(HOME)/.exatn/bin/exatn-config --includes`
LFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --libs`
SRC_DIR=/home/parallels/projects/my_cas-tn_branch/cas-tn/src

simulation.o: $(SRC_DIR)/simulation.cpp $(SRC_DIR)/simulation.hpp
	$(CC) $(CFLAGS) -c $(SRC_DIR)/simulation.cpp

clean: 
	$(RM) *.o
