TOPDIR = 
CXX    = 
RM     = rm
CXXFLAGS = -g `$(TOPDIR)/.exatn/bin/exatn-config --cxxflags` `$(TOPDIR)/.exatn/bin/exatn-config --includes`
LFLAGS = -g `$(TOPDIR)/.exatn/bin/exatn-config --libs` -lm
TEST_INPUT_DIR=
CAS_TN_OBJECT_DIR=

h8.x: h8.o $(CAS_TN_OBJECT_DIR)/simulation.o
	$(CXX) $(CXXFLAGS) h8.o $(CAS_TN_OBJECT_DIR)/simulation.o -o h8.x $(LFLAGS)

h8.o: $(TEST_INPUT_DIR)/h8.cpp
	$(CXX) $(CXXFLAGS) -c $(TEST_INPUT_DIR)/h8.cpp

clean: 
	$(RM) *.o *.x 

