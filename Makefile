TOPDIR  =  
CC      =   
CFLAGS  =  -g `$(TOPDIR)/.exatn/bin/exatn-config --cxxflags` `$(TOPDIR)/.exatn/bin/exatn-config --includes` 
LFLAGS  =  -g `$(TOPDIR)/.exatn/bin/exatn-config --libs` -llapacke -llapack -lblas -lm
SRC_DIR =  

simulation.o: $(SRC_DIR)/simulation.cpp $(SRC_DIR)/simulation.hpp
	$(CC) $(CFLAGS) -c $(SRC_DIR)/simulation.cpp

clean: 
	rm -f simulation.o

