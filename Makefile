HOME=/home/parallels
CC=g++-9
RM=rm
CFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --cxxflags` `$(HOME)/.exatn/bin/exatn-config --includes` -DMKL_ILP64 -m64 -I${MKLROOT}/include
LFLAGS=-g `$(HOME)/.exatn/bin/exatn-config --libs -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl`
SRC_DIR=$(HOME)/projects/my_cas-tn_branch/cas-tn/src

simulation.o: $(SRC_DIR)/simulation.cpp $(SRC_DIR)/simulation.hpp
	$(CC) $(CFLAGS) -c $(SRC_DIR)/simulation.cpp

clean: 
	$(RM) *.o
