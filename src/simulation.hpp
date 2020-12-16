/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation: Main header
REVISION: 2020/12/15

Copyright (C) 2020-2020 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 <Describe what functionality is provided by this header>
**/

#include "exatn.hpp"

#include <iostream>
#include <vector>

#ifndef CASTN_SIMULATION_HPP_
#define CASTN_SIMULATION_HPP_

namespace castn {

class Simulation {

public:

 /** Basic configuration of a quantum many-body system. **/
 Simulation(std::size_t num_orbitals,      //in: number of active orbitals
            std::size_t num_particles,     //in: number of active particles
            std::size_t num_core_orbitals, //in: number of core orbitals (with core particles)
            std::size_t total_orbitals,    //in: total number of orbitals
            std::size_t total_particles):  //in: total number of particles
  num_orbitals_(num_orbitals),
  num_particles_(num_particles),
  num_core_orbitals_(num_core_orbitals),
  total_orbitals_(total_orbitals),
  total_particles_(total_particles)
  {}

 /** Resets the wavefunction ansatz by copying the provided tensor network. **/
 void resetWaveFunctionAnsatz(const exatn::TensorNetwork & wavefunction);

 /** Resets the wavefunction ansatz by copying the provided tensor network expansion. **/
 void resetWaveFunctionAnsatz(const exatn::TensorExpansion & wavefunction);

 /** Resets the wavefunction ansatz by constructing an appropriate tensor network. **/
 void resetWaveFunctionAnsatz(exatn::NetworkBuilder & ansatz_builder);

 /** Resets the quantum Hamiltonian. **/
 void resetHamiltonian(const std::vector<exatn::Tensor> & hamiltonian);

 /** Optimizes the wavefunction ansatz to minimize the energy expectation value. **/
 bool optimize(std::size_t num_states); //in: number of the lowest quantum states to optimize the energy for

private:

 std::size_t num_orbitals_;      //number of active orbitals
 std::size_t num_particles_;     //number of active particles
 std::size_t num_core_orbitals_; //number of core orbitals
 std::size_t total_orbitals_;    //total number of orbitals
 std::size_t total_particles_;   //total number of particles

 exatn::TensorExpansion ansatz_;                           //wavefunction ansatz
 std::vector<std::shared_ptr<exatn::Tensor>> hamiltonian_; //Hamiltonian tensors

 std::size_t num_states_;             //number of lowest-energy states to find
 std::vector<double> state_energies_; //quantum state energies
};

} //namespace castn

#endif //CASTN_SIMULATION_HPP_
