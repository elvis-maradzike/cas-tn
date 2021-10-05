TOPDIR  =  
CXX      =   
CXXFLAGS  =  -g `$(TOPDIR)/.exatn/bin/exatn-config --cxxflags` `$(TOPDIR)/.exatn/bin/exatn-config --includes` 
LFLAGS  =  -g `$(TOPDIR)/.exatn/bin/exatn-config --libs` -lm
SRC_DIR =  

simulation.o: $(SRC_DIR)/simulation.cpp $(SRC_DIR)/simulation.hpp
	$(CXX) $(CXXFLAGS) -c $(SRC_DIR)/simulation.cpp

clean: 
	rm -f simulation.o

