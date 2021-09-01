HOME=/home/parallels
CC=g++-9
RM=rm
CFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --cxxflags` `$(HOME)/.exatn/bin/exatn-config --includes` -DMKL_ILP64 -m64 -I${MKLROOT}/include
LFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --libs -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl`
OBJ_FILE_DIR=$(HOME)/projects/my_cas-tn_branch/cas-tn
SRC_DIR=$(HOME)/projects/my_cas-tn_branch/cas-tn/tests

h4.x: h4.o $(OBJ_FILE_DIR)/simulation.o
	$(CC) $(CFLAGS) h4.o $(OBJ_FILE_DIR)/simulation.o -o h4.x $(LFLAGS)

h4.o: $(SRC_DIR)/h4.cpp
	$(CC) $(CFLAGS) -c $(SRC_DIR)/h4.cpp

clean: 
	$(RM) *.o *.x 

