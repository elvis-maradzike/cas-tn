/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation
REVISION: 2022/08/19

Copyright (C) 2020-2022 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2022 Oak Ridge National Laboratory (UT-Battelle)

**/

#include "particle_ansatz.hpp"
#include "exatn.hpp"
#include "talshxx.hpp"
#include <unordered_set>
#include "quantum.hpp"

namespace castn {

// reset wavefunction ansatz when input as a tensor network
void ParticleAnsatz::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorNetwork> ansatz)
{
  bra_ansatz_.reset();
  ket_ansatz_.reset();
  ket_ansatz_ = std::make_shared<exatn::TensorExpansion>();
  ket_ansatz_->appendComponent(ansatz,{1.0,0.0});
  return;
}

// reset wavefunction ansatz when input as a tensor network expansion
void ParticleAnsatz::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorExpansion> ansatz)
{
  bra_ansatz_.reset();
  ket_ansatz_.reset();
  ket_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ansatz);
  return;
}

// reset wavefunction ansatz to be built from tensors and 
// tensor connections specified by user.
void ParticleAnsatz::resetWaveFunctionAnsatz(exatn::NetworkBuilder & ansatz_builder)
{
 //`Implement
 return;
}
 
// resets the Hamiltonian operator when using spin-site/qubit representation   
void SpinSiteAnsatz::resetHamiltonianOperator(std::shared_ptr<exatn::TensorOperator> hamiltonian){
  hamiltonian_operator_.reset();
  hamiltonian_operator_ = std::make_shared<exatn::TensorOperator>(*hamiltonian);
  return;
}

// resets the vector of tensors of one- and two-electron integrals used to 
// defined the Hamiltonian operator
void ParticleAnsatz::resetHamiltonian(const std::vector<std::shared_ptr<exatn::Tensor>> & hamiltonian)
{
 hamiltonian_ = hamiltonian;
 return;
}

// marks tensors as optimizable, subsequent tensors added to the network will
// not be optimizable 
void ParticleAnsatz::markOptimizableTensors(){
  for (auto component = ket_ansatz_->begin(); component != ket_ansatz_->end(); ++component){
    component->network->markOptimizableAllTensors();
  }
}

// initializes ordering projectors with a delta function. 
// This code is copied from the ExaTN source code
void ParticleAnsatz::FunctorInitDelta::pack(BytePacket & packet)
{
  return;
}

void ParticleAnsatz::FunctorInitDelta::unpack(BytePacket & packet)
{
  return;
}

int ParticleAnsatz::FunctorInitDelta::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
  unsigned int rank;
  const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
  const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
  const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice
 
  auto init_delta = [&](auto * tensor_body){
  std::vector<exatn::DimOffset> bas(rank); 
  for(unsigned int i = 0; i < rank; ++i) bas[i] = offsets[i]; //tensor slice dimension base offsets
  std::vector<exatn::DimExtent> ext(rank); 
  for(unsigned int i = 0; i < rank; ++i) ext[i] = extents[i]; //tensor slice dimension extents
  exatn::TensorRange tens_range(bas,ext);
  bool not_over = true;
  const auto & multi_index = tens_range.getMultiIndex();
  while(not_over){
    if (tens_range.onDiagonal()){
      tensor_body[tens_range.localOffset()] = 1.0;
    }else{
      tensor_body[tens_range.localOffset()] = 0.0;
    }
    not_over = tens_range.next();
  }
  return 0;
  };
 
  auto access_granted = false;
  {//Try REAL32:
    float * body;
    access_granted = local_tensor.getDataAccessHost(&body);
    if(access_granted) return init_delta(body);
  }
  
  {//Try REAL64:
    double * body; 
    access_granted = local_tensor.getDataAccessHost(&body);
    if(access_granted) return init_delta(body);
  }
  
  {//Try COMPLEX32:
    std::complex<float> * body;
    access_granted = local_tensor.getDataAccessHost(&body);
    if(access_granted) return init_delta(body);
  }
  
  {//Try COMPLEX64:
    std::complex<double> * body;
    access_granted = local_tensor.getDataAccessHost(&body);
    if(access_granted) return init_delta(body);
  }
 
  std::cout << "#ERROR(exatn::numerics::FunctorInitDelta): Unknown data kind in talsh::Tensor!" << std::endl;
  return 1;
}

// initializes tensors with 1.0 and 0.0 depending on 
// whether or not the labels are in ascending order.
void ParticleAnsatz::FunctorInitOrdering::pack(BytePacket & packet)
{
 return;
}

void ParticleAnsatz::FunctorInitOrdering::unpack(BytePacket & packet)
{
 return;
}

int ParticleAnsatz::FunctorInitOrdering::apply(talsh::Tensor & local_tensor) //tensor slice (in general)
{
 unsigned int rank;
 const auto * extents = local_tensor.getDimExtents(rank); //rank is returned by reference
 const auto tensor_volume = local_tensor.getVolume(); //volume of the given tensor slice
 const auto & offsets = local_tensor.getDimOffsets(); //base offsets of the given tensor slice
 
 auto init_ordered = [&](auto * tensor_body){
 std::vector<exatn::DimOffset> bas(rank);
 for(unsigned int i = 0; i < rank; ++i) bas[i] = offsets[i]; //tensor slice dimension base offsets
 std::vector<exatn::DimExtent> ext(rank);
 for(unsigned int i = 0; i < rank; ++i) ext[i] = extents[i]; //tensor slice dimension extents
 exatn::TensorRange tens_range(bas,ext);
 bool not_over = true;
 const auto & multi_index = tens_range.getMultiIndex();
 while(not_over){
   if (tens_range.increasingOrder()){
     tensor_body[tens_range.localOffset()] = 1.0;
   }else{
     tensor_body[tens_range.localOffset()] = 0.0;
   }
   not_over = tens_range.next();
 }
 return 0;
 };
 
 auto access_granted = false;
 {//Try REAL32:
  float * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_ordered(body);
 }
 
 {//Try REAL64:
  double * body; 
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_ordered(body);
 }

 {//Try COMPLEX32:
  std::complex<float> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_ordered(body);
 }

 {//Try COMPLEX64:
  std::complex<double> * body;
  access_granted = local_tensor.getDataAccessHost(&body);
  if(access_granted) return init_ordered(body);
 }

 std::cout << "#ERROR(exatn::numerics::FunctorInitOrdering): Unknown data kind in talsh::Tensor!" << std::endl;
 return 1;
}


bool ParticleAnsatz::optimize(std::size_t num_states, double convergence_thresh){
 //#ifdef CUQUANTUM
  auto success = exatn::sync(); assert(success);
  auto backends = exatn::queryComputationalBackends();
 //if(std::find(backends.cbegin(),backends.cend(),"default") != backends.cend())
 //exatn::switchComputationalBackend("default");
 //#endif
  exatn::resetContrSeqOptimizer("cutnn", false, true);

  bool success = false;
  auto hamiltonian_operator = exatn::makeSharedTensorOperator("HamiltonianOperator");
  //(anti)symmetrization
  success = hamiltonian_operator->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = hamiltonian_operator->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);

  //mark optimizable tensors
  markOptimizableTensors();
 
  //append ordering projectors
  appendOrderingProjectors();

  //set up and call the optimizer in ../src/exatn/..
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer(hamiltonian_operator,ket_ansatz_,convergence_thresh_);
  optimizer.enableParallelization(true);
  optimizer.resetLearningRate(0.5);
  bool multistate = false;
  if (num_states > 1) multistate = true;
  bool converged = optimizer.optimize(multistate);
  success = exatn::sync(); assert(success);
  success = converged;
  if(exatn::getProcessRank() == 0){
   if(converged){
    std::cout << "Optimization succeeded!" << std::endl;
   }else{
    std::cout << "Optimization failed!" << std::endl;
   }
  }
  return success;
}

bool SpinSiteAnsatz::optimize(std::size_t num_states, double convergence_thresh){
  bool success = true;
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer(hamiltonian_operator_,ket_ansatz_,convergence_thresh_);
  optimizer.enableParallelization(true);
  optimizer.resetLearningRate(0.5);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  success = converged;
  if(exatn::getProcessRank() == 0){
    if(converged){
      std::cout << "Optimization succeeded!" << std::endl;
    }else{
      std::cout << "Optimization failed!" << std::endl;
    }
  }
  return success;
}

// appends N layers of ordering projectors to the tensor network (expansion). 
// For an ansatz with num_output_modes=particle_number_, N=particle_number_/2.
void ParticleAnsatz::appendOrderingProjectors(){
  exatn::TensorElementType tens_elem_type;
 
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        tens_elem_type = tensor.getElementType();
      }
    }
  }
 
  /*
  auto created = exatn::createTensor("Q",tens_elem_type, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  exatn::transformTensorSync("Q", std::shared_ptr<exatn::TensorMethod>{new exatn::numerics::FunctorInitProj()});

  auto appended = false;
  int num_layers = num_particles_/2;
  for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
    auto & network = *(iter->network);
    unsigned int tensorIdCounter = 1 + network.getNumTensors();
    do {
     for ( unsigned int i = num_particles_/2 - num_layers; i < num_particles_ - ( 1 + (num_particles_/2 - num_layers)) ; i = i + 2){
       appended = network.appendTensorGate(tensorIdCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
       tensorIdCounter++;
     }
     std::cout << num_layers << std::endl;
     std::cout << tensorIdCounter << std::endl;
     num_layers--;
    }while(num_layers >= 1);
  }
  std::cout << "Done appending ordering projectors " << std::endl;
  */
 
  // Q is expressed as a contraction of three smaller tensors L,M and R 
  auto created = exatn::createTensor("L", tens_elem_type, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  created = exatn::createTensor("M", tens_elem_type, exatn::TensorShape{total_orbitals_,total_orbitals_}); assert(created);
  created = exatn::createTensor("R", tens_elem_type, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);

  // initialize the three tensors
  auto done = transformTensor("L", std::shared_ptr<exatn::TensorMethod>( new FunctorInitDelta())); assert(done);
  done = transformTensor("R", std::shared_ptr<exatn::TensorMethod>( new FunctorInitDelta())); assert(done);
  done = transformTensor("M", std::shared_ptr<exatn::TensorMethod>( new FunctorInitOrdering())); assert(done);

  // define pairing for tensors
  std::vector<std::pair<unsigned int, unsigned int>> pairing1;
  std::vector<std::pair<unsigned int, unsigned int>> pairing2;
  pairing1.emplace_back(std::make_pair(1,0));
  pairing2.emplace_back(std::make_pair(2,0));
  
  // number of sites
  auto num_sites = total_particles_;
  // number of layers needed
  auto num_layers = total_particles_/2;

  auto oleft = exatn::getTensor("L");
  auto omiddle = exatn::getTensor("M");
  auto oright = exatn::getTensor("R");

  // append layers of ordering projectors to tensor network
  auto appended = false;
  // first layer
  for ( auto i = 0; i < total_particles_/2; i++){
    auto tn_op = exatn::makeSharedTensorNetwork("TN_OP");
    appended = tn_op->appendTensor(1, oleft, {},{}, false); assert(appended);
    appended = tn_op->appendTensor(2, omiddle, pairing1, {}, false); assert(appended);
    appended = tn_op->appendTensor(3, oright, pairing2, {}, false); assert(appended);
    for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
      auto & network = *(iter->network);
      appended = network.appendTensorNetwork(std::move(*tn_op),{{0,0},{1,2}}); assert(appended);
    }
  }
  // all subsequent layers
  for ( auto j = 1; j < num_layers; ++j){
    for ( auto i = 0; i < num_sites/2-j; ++i){
      auto tn_op = exatn::makeSharedTensorNetwork("TN_OP");
      appended = tn_op->appendTensor(1, oleft, {},{}, false); assert(appended);
      appended = tn_op->appendTensor(2, omiddle, pairing1, {}, false); assert(appended);
      appended = tn_op->appendTensor(3, oright, pairing2, {}, false); assert(appended);
      for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
        auto & network = *(iter->network);
        appended = network.appendTensorNetwork(std::move(*tn_op),{{2*j-1,0},{2*j,2}}); assert(appended);
      }
    }
  }

  //reorder output modes
  std::vector<unsigned int> reorder;
  unsigned int num_output_modes = total_particles_;
  for ( unsigned int i = 0; i < num_output_modes; i++){
    if (i%2 == 0) reorder.emplace_back(i);
  }
  for ( unsigned int i = num_output_modes-1; i > 0; i--){
    if (i%2 == 1) reorder.emplace_back(i);
  }

  for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
    auto & network = *(iter->network);
    auto done = network.reorderOutputModes(reorder); assert(done);
  }

  for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
    auto & network = *(iter->network);
    network.printIt();
  }

  ket_ansatz_->printIt();
}
} //namespace castn
