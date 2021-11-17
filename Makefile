CXX = g++
RM = rm
CXXFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --cxxflags` `$(EXATN_DIR)/bin/exatn-config --includes`
LFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --libs` -lm

simulation.o: ./src/simulation.cpp ./src/simulation.hpp
	$(CXX) $(CXXFLAGS) -c ./src/simulation.cpp

	make -C ./tests/h4
	make -C ./tests/h8

clean:
	$(RM) *.o
