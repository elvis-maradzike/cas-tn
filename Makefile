#Specify the path to the ExaTN install directory:
export EXATN_DIR ?= ~/.exatn

CXX = g++
RM = rm

CXXFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --cxxflags` `$(EXATN_DIR)/bin/exatn-config --includes`
LFLAGS = -g `$(EXATN_DIR)/bin/exatn-config --libs` -lm

particle_ansatz.o: ./src/particle_ansatz.cpp ./src/particle_ansatz.hpp
	$(CXX) $(CXXFLAGS) -c ./src/particle_ansatz.cpp

	 make -C ./tests/h4
	 #make -C ./tests/h8
	 #make -C ./tests/h4_by_onr
	 #make -C ./tests/h4_by_onr_contiguous_ansatz
	 make -C ./tests/h4_by_onr_pnumber_constrained
	 #cp ./tests/*/*.x ./

clean:
	$(RM) -rf *.x ./tests/*/*.x *.o ./tests/*/*.o
