# Complete-Active-Space Tensor-Network (CAS-TN) Ansatz Library

## Overview
For a user-defined tensor network ansatz for the electronic 
wavefunction, the CAS-TN ansatz library computes the 
optimized tensor network parameters and the corresponing ground state
nergy.

## Dependencies 
An installation of the ExaTN library. 
Visit https://github.com/ORNL-QCI/exatn for details on how 
to download and build the library.

## Building executables and running CAS-TN
To run CAS-TN, you will need to specify your C++ compiler, 
and the location of your exatn install (EXATN\_DIR) in the 
Makefile provided. 

Then: make 

This compiles the CAS-TN source code and 
generates an object files (simulation.o)

Then: make -C ./tests/h4/h4.cpp and/or 
      make -C ./tests/h8/h8.cpp 

This compiles source code defining the input (tensor network ansatzes)
for a system of 4 electrons (H\_4) and another of eight electrons (H\_8),
and links the compiled code with the CAS-TN library.

The generated executables may be run as you would any other executable, e.g:
  cd tests/h4
  ./h4.x > h4.out & 

