HOME=
CC=g++
RM=rm
CFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --cxxflags` `$(HOME)/.exatn/bin/exatn-config --includes` 
LFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --libs` -llapacke -llapack -lblas -lm
SRC_DIR=$(HOME)/..

simulation.o: $(SRC_DIR)/simulation.cpp $(SRC_DIR)/simulation.hpp
	$(CC) $(CFLAGS) -c $(SRC_DIR)/simulation.cpp

clean: 
	$(RM) *.o
