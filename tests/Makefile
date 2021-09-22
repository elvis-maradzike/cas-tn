TOPDIR = 
CC     = 
RM     = rm
CFLAGS = -g `$(TOPDIR)/.exatn/bin/exatn-config --cxxflags` `$(TOPDIR)/.exatn/bin/exatn-config --includes`
LFLAGS = -g `$(TOPDIR)/.exatn/bin/exatn-config --libs` -llapacke -llapack -lblas -lm
TEST_INPUT_DIR=
CAS_TN_OBJECT_DIR=

h4.x: h4.o $(CAS_TN_OBJECT_DIR)/simulation.o
	$(CC) $(CFLAGS) h4.o $(CAS_TN_OBJECT_DIR)/simulation.o -o h4.x $(LFLAGS)

h4.o: $(TEST_INPUT_DIR)/h4.cpp
	$(CC) $(CFLAGS) -c $(TEST_INPUT_DIR)/h4.cpp

clean: 
	$(RM) *.o *.x 

