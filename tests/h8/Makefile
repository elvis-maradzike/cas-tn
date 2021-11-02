export EXATN_DIR ?= EXATN_DIR
CXX = g++
RM = rm
CXXFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --cxxflags` `$(EXATN_DIR)/bin/exatn-config --includes`
LFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --libs` -lm

h8.x: h8.o ../../simulation.o
	$(CXX) $(CXXFLAGS) h8.o ../../simulation.o -o h8.x $(LFLAGS)

h8.o: h8.cpp
	$(CXX) $(CXXFLAGS) -c h8.cpp

clean: 
	$(RM) *.o *.x 

