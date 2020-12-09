/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation: Main header
REVISION: 2020/12/09

Copyright (C) 2020-2020 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2020 Oak Ridge National Laboratory (UT-Battelle)

Rationale:
 <Describe what functionality is provided by this header>
**/

#include "exatn.hpp"

#include <iostream>

#ifndef CASTN_SIMULATION_HPP_
#define CASTN_SIMULATION_HPP_

namespace castn {

class Simulation {

public:

 /** Optimizes the wavefunction ansatz to minimize the energy expectation value. **/
 bool optimize();

private:

 std::size_t num_orbitals_;      //number of active orbitals
 std::size_t num_particles_;     //number of active particles
 std::size_t num_core_orbitals_; //number of core orbitals
 std::size_t total_orbitals_;    //total number of orbitals
 std::size_t total_particles_;   //total number of particles

 std::size_t num_states_;        //number of lowest-energy states to find

 exatn::TensorExpansion ansatz_; //wavefunction ansatz

};

} //namespace castn

#endif //CASTN_SIMULATION_HPP_
