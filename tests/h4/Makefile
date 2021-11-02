export EXATN_DIR ?= EXATN_DIR
CXX = g++
RM = rm
CXXFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --cxxflags` `$(EXATN_DIR)/bin/exatn-config --includes`
LFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --libs` -lm

h4.x: h4.o ../../simulation.o
	$(CXX) $(CXXFLAGS) h4.o ../../simulation.o -o h4.x $(LFLAGS)

h4.o: h4.cpp
	$(CXX) $(CXXFLAGS) -c h4.cpp

clean: 
	$(RM) *.o *.x 

