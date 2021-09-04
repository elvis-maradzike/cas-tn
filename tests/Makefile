HOME=
CC=
RM=rm
CFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --cxxflags` `$(HOME)/.exatn/bin/exatn-config --includes`
LFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --libs` -llapacke -llapack -lblas -lm
SIMULATION_CLASS_OBJECT_DIR=
SRC_DIR=

h4.x: h4.o $(SIMULATION_CLASS_OBJECT_DIR)/simulation.o
	$(CC) $(CFLAGS) h4.o $(SIMULATION_CLASS_OBJECT_DIR)/simulation.o -o h4.x $(LFLAGS)

h4.o: $(SRC_DIR)/h4.cpp
	$(CC) $(CFLAGS) -c $(SRC_DIR)/h4.cpp

clean: 
	$(RM) *.o *.x 

