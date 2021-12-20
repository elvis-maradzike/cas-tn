#Specify the path to the ExaTN install directory:
export EXATN_DIR ?= ~/.exatn

CXX = g++
RM = rm

CXXFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --cxxflags` `$(EXATN_DIR)/bin/exatn-config --includes`
LFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --libs` -lm

particle_number_representation.o: ./src/particle_number_representation.cpp ./src/particle_number_representation.hpp
	$(CXX) $(CXXFLAGS) -c ./src/particle_number_representation.cpp

	 make -C ./tests/h4
	 make -C ./tests/h8
	 make -C ./tests/h4_by_onr
	 #make -C ./tests/h4_by_onr_contiguous_ansatz
	 #cp ./tests/*/*.x ./

clean:
	$(RM) -rf *.x ./tests/*/*.x *.o ./tests/*/*.o
