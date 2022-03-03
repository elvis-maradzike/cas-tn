/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation
REVISION: 2021/12/08

Copyright (C) 2020-2021 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2021 Oak Ridge National Laboratory (UT-Battelle)

**/

#include "particle_ansatz.hpp"
#include "exatn.hpp"
#include "talshxx.hpp"
#include <unordered_set>

namespace castn {



void ParticleAnsatz::clear()
{
 state_energies_.clear();
 derivatives_.clear();
 expectation_expansion_.reset();
 return;
}


void ParticleAnsatz::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorNetwork> ansatz)
{
  clear();
  bra_ansatz_.reset();
  ket_ansatz_.reset();

  ket_ansatz_ = std::make_shared<exatn::TensorExpansion>();
  ket_ansatz_->appendComponent(ansatz,{1.0,0.0});
  return;
}


void ParticleAnsatz::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorExpansion> ansatz)
{
  clear();
  bra_ansatz_.reset();
  ket_ansatz_.reset();
  ket_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ansatz);
  return;
}


void ParticleAnsatz::resetWaveFunctionAnsatz(exatn::NetworkBuilder & ansatz_builder)
{
 clear();
 //`Implement
 return;
}
    
void SpinSiteAnsatz::resetHamiltonianOperator(std::shared_ptr<exatn::TensorOperator> hamiltonian){
  clear();
  hamiltonian_operator_.reset();
  hamiltonian_operator_ = std::make_shared<exatn::TensorOperator>(*hamiltonian);
  return;
}


void SpinSiteAnsatz::resetConstraintOperator(std::shared_ptr<exatn::TensorOperator> constraint){
  clear();
  constraint_operator_.reset();
  constraint_operator_ = std::make_shared<exatn::TensorOperator>(*constraint);
  return;
}

void ParticleAnsatz::resetHamiltonian(const std::vector<std::shared_ptr<exatn::Tensor>> & hamiltonian)
{
 clear();
 hamiltonian_ = hamiltonian;
 return;
}

void ParticleAnsatz::markOptimizableTensors(){
  for (auto component = ket_ansatz_->begin(); component != ket_ansatz_->end(); ++component){
    component->network->markOptimizableAllTensors();
  }
}


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

  bool success = false;
  auto hamiltonian_operator = exatn::makeSharedTensorOperator("HamiltonianOperator");
  //(anti)symmetrization
  success = hamiltonian_operator->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = hamiltonian_operator->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);

  //mark optimizable tensors
  markOptimizableTensors();
 
  //append ordering projectors
  appendOrderingProjectors();

  //setting up and calling the optimizer in ../src/exatn/..
  exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
  exatn::TensorNetworkOptimizer optimizer(hamiltonian_operator,ket_ansatz_,convergence_thresh_);
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

bool SpinSiteAnsatz::optimize(std::size_t num_states, double convergence_thresh){

  bool success = true;
    //setting up and calling the optimizer in ../src/exatn/..
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
 
  /* 
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
  const auto TENSOR_SHAPE = exatn::TensorShape{};
  bool success = true;
  constructEnergyFunctional();
  constructEnergyDerivatives();
  initWavefunctionAnsatz();
  double lambda = 0.2;
  double c = 1.0;
  double c_update_factor = 2.0;
  double change_in_lagrangian = 0.0;
  std::complex<double> val_constraint = 0.0;
  std::complex<double> val_expectation = 0.0;
  std::complex<double> val_augmentation = 0.0;
  unsigned int max_constraint_updates = 20;
  unsigned int constraint_update = 0;
  // for fixed l and c, minimize L.
  do{
    // minimize augmented Lagrangian
    unsigned int max_iterations = 999;
    double sum_max_abs;
      do {
        sum_max_abs = 0.0;
      // evaluate
      auto created = exatn::createTensorSync("ac_augmentation",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
      auto ac_augmentation = exatn::getTensor("ac_augmentation");
      auto initialized = exatn::initTensor(ac_augmentation->getName(), 0.0); assert(initialized);
      auto evaluated = exatn::evaluateSync(*augmentation_expansion_,ac_augmentation); assert(evaluated);
      auto local_copy = exatn::getLocalTensor(ac_augmentation->getName()); assert(local_copy);
      const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
      auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
      val_augmentation = *body_ptr;
      body_ptr = nullptr;
      created = exatn::createTensorSync("ac_expectation",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
      auto ac_expectation = exatn::getTensor("ac_expectation");
      initialized = exatn::initTensor(ac_expectation->getName(), 0.0); assert(initialized);
      evaluated = exatn::evaluateSync(*expectation_expansion_,ac_expectation); assert(evaluated);
      // get value
      auto talsh_tensor = exatn::getLocalTensor(ac_expectation->getName());
      access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
      if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
        for ( int i = 0; i < talsh_tensor->getVolume(); i++){
          val_expectation = body_ptr[i];
        }
      }
      body_ptr = nullptr;
      created = exatn::createTensorSync("ac_constraint",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
      auto ac_constraint = exatn::getTensor("ac_constraint");
      initialized = exatn::initTensor(ac_constraint->getName(), 0.0); assert(initialized);
      evaluated = exatn::evaluateSync(*constraint_expansion_,ac_constraint); assert(evaluated);
      // get value
      talsh_tensor = exatn::getLocalTensor(ac_constraint->getName());
      access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
      if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
        for ( int i = 0; i < talsh_tensor->getVolume(); i++){
          val_constraint = body_ptr[i];
        }
      }
      body_ptr = nullptr;

      std::cout << "Augmented Lagrangian: " << val_augmentation + lambda * val_constraint + c * val_constraint * val_constraint << ", Expectation: " << val_expectation  <<  ", Constraint: " << val_constraint << std::endl;
     
      evaluateEnergyDerivatives();


       // update wavefunction ansatz 
      int maxiter = 1;
      // for every optimizable tensor
      double max_abs_element = 0.0;
      for (auto & environment: environments_){
        // microiterations
        for ( int microiteration = 0; microiteration < maxiter; microiteration++){
          // compute gradient of augmented lagrangian
          auto gradient = std::make_shared<exatn::Tensor>("Gradient_"+environment.tensor->getName(), environment.gradient_expectation_tensor->getShape(), environment.gradient_expectation_tensor->getSignature());
          created = exatn::createTensorSync(gradient,environment.gradient_expectation_tensor->getElementType()); assert(created);
          initialized = exatn::initTensorSync(gradient->getName(),0.0); assert(initialized);

          std::string add_pattern;
          //auto ac_gradient_expectation_tensor = exatn::getTensor("GradientExpectationTensor_"+environment.tensor->getName());
          auto generated = exatn::generate_addition_pattern(environment.gradient_expectation_tensor->getRank(),add_pattern,true,gradient->getName(), environment.gradient_expectation_tensor->getName()); assert(generated);
          auto factor = 1.0;
          auto added = exatn::addTensors(add_pattern, factor); assert(added);
          add_pattern.clear();
          generated = exatn::generate_addition_pattern(environment.gradient_constraint_tensor->getRank(),add_pattern,true,gradient->getName(),  environment.gradient_constraint_tensor->getName()); assert(generated);
          factor = (1.0 + 2 * c * val_constraint.real());
          added = exatn::addTensors(add_pattern, factor); assert(added);
          add_pattern.clear();
          auto success = exatn::computeMaxAbsSync(gradient->getName(), max_abs_element); assert(success);
          std::cout << "Max. absolute value in gradient tensor: " << max_abs_element << std::endl;
          sum_max_abs += max_abs_element;
          if (max_abs_element < convergence_thresh_) continue;
          // update wavefunction variable
          generated = exatn::generate_addition_pattern(environment.tensor->getRank(),add_pattern,true,environment.tensor->getName(), gradient->getName()); assert(generated);
          std::cout << add_pattern << std::endl;
          auto lr = 0.1;
          added = exatn::addTensors(add_pattern, -lr); assert(added);
          add_pattern.clear();
    
          std::cout << " Done updating tensor "  << environment.tensor->getName() << std::endl;
          auto destroyed = exatn::destroyTensorSync(gradient->getName()); assert(destroyed);
        }
      } 
      environments_.clear();
      std::cout << "sum max abs: " << sum_max_abs << std::endl;
      auto destroyed = exatn::destroyTensorSync(ac_expectation->getName()); assert(destroyed);
      destroyed = exatn::destroyTensorSync(ac_augmentation->getName()); assert(destroyed);
      destroyed = exatn::destroyTensorSync(ac_constraint->getName()); assert(destroyed);
          
     }while(sum_max_abs >  convergence_thresh_);
     lambda = lambda - 2.0 * c * val_constraint.real();
     c = c_update_factor * c; 
     std::cout << "lambda, c: " << lambda << "," << c << std::endl;
   } while(fabs(change_in_lagrangian) > 1e-5);
   */

  return success;
}

/** Appends two layers of ordering projectors to the wavefunction ansatz. **/
void ParticleAnsatz::appendOrderingProjectors(){

  /*
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  auto created = exatn::createTensor("Q",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  exatn::transformTensorSync("Q", std::shared_ptr<exatn::TensorMethod>{new exatn::numerics::FunctorInitProj()});

  auto appended = false;
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    auto & network = *(iter->network);
    unsigned int tensorIdCounter = 1 + network.getNumTensors();
    // applying layer 1
    for ( unsigned int i = 0; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorIdCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      tensorIdCounter++;
    }
    // applying layer 2
    for ( unsigned int i = 1; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorIdCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      tensorIdCounter++;
    }
  }
  std::cout << "Done with ordering projectors " << std::endl;
  */

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  auto created = exatn::createTensor("INTERMEDIATE",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  created = exatn::createTensor("DIFF",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  created = exatn::createTensor("Q",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_, total_orbitals_}); assert(created);
  created = exatn::createTensor("OP",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_, total_orbitals_}); assert(created);
  created = exatn::createTensor("OL",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  created = exatn::createTensor("OM",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_}); assert(created);
  created = exatn::createTensor("OR",TENS_ELEM_TYPE, exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_}); assert(created);
  created = exatn::createTensor("tmp1",TENS_ELEM_TYPE, exatn::TensorShape{4,4,4}); assert(created);
  created = exatn::createTensor("tmp2",TENS_ELEM_TYPE, exatn::TensorShape{4,4}); assert(created);
  created = exatn::createTensor("tmp3",TENS_ELEM_TYPE, exatn::TensorShape{4,4,4,4}); assert(created);

  auto done = transformTensor("OL", std::shared_ptr<exatn::TensorMethod>( new FunctorInitDelta())); assert(done);
  done = transformTensor("OR", std::shared_ptr<exatn::TensorMethod>( new FunctorInitDelta())); assert(done);
  done = transformTensor("OM", std::shared_ptr<exatn::TensorMethod>( new FunctorInitOrdering())); assert(done);
  done = transformTensorSync("Q", std::shared_ptr<exatn::TensorMethod>{new exatn::numerics::FunctorInitProj()}); assert(done);
  //done = transformTensorSync("tmp1", std::shared_ptr<exatn::TensorMethod>{new FunctorInitDelta()}); assert(done);
  //done = transformTensorSync("tmp2", std::shared_ptr<exatn::TensorMethod>{new FunctorInitOrdering()}); assert(done);
  //done = transformTensorSync("tmp3", std::shared_ptr<exatn::TensorMethod>{new exatn::numerics::FunctorInitProj()}); assert(done);

  //done = exatn::printTensorSync("tmp1"); assert(done);
  //done = exatn::printTensorSync("tmp3"); assert(done);

  done = exatn::initTensor("DIFF", 0.0);
  done = exatn::initTensor("OP", 0.0);
  done = exatn::initTensor("INTERMEDIATE", 0.0);
  done = exatn::contractTensorsSync("INTERMEDIATE(i,k,n)+=OL(i,m,k)*OM(m,n)", 1.0); assert(done);
  done = exatn::contractTensorsSync("OP(i,j,k,l)+=INTERMEDIATE(i,k,n)*OR(j,n,l)", 1.0); assert(done);
  done = exatn::addTensors("DIFF(i,j,k,l)+=Q(i,j,k,l)", 1.0); assert(done);
  done = exatn::addTensors("DIFF(i,j,k,l)+=OP(i,j,k,l)", -1.0); assert(done);

  double val = 0.0;
  done = exatn::computeMaxAbsSync("DIFF",val); assert(done);
  std::cout << "val is " << val << std::endl;

  std::vector<std::pair<unsigned int, unsigned int>> pairing1;
  std::vector<std::pair<unsigned int, unsigned int>> pairing2;
  pairing1.emplace_back(std::make_pair(1,0));
  pairing2.emplace_back(std::make_pair(2,0));

  auto appended = false;

  // number of sites
  auto num_sites = total_particles_;
  // number of layers needed
  auto num_layers = total_particles_/2;

  auto oleft = exatn::getTensor("OL");
  auto omiddle = exatn::getTensor("OM");
  auto oright = exatn::getTensor("OR");
  auto op = exatn::getTensor("OP");

  done = exatn::printTensorSync("OR"); assert(done);
    // first layer
  for ( auto i = 0; i < total_particles_/2; i++){
    auto tn_op = exatn::makeSharedTensorNetwork("TN_OP");
    appended = tn_op->appendTensor(1, oleft, {},{}, false); assert(appended);
    appended = tn_op->appendTensor(2, omiddle, pairing1, {}, false); assert(appended);
    appended = tn_op->appendTensor(3, oright, pairing2, {}, false); assert(appended);

    for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
      auto & network = *(iter->network);
      appended = network.appendTensorNetwork(std::move(*tn_op),{{0,0},{1,2}}); assert(appended);
    //  network.printIt();
    }
  }

  // subsequent layers
  for ( auto j = 1; j < num_layers; ++j){
    for ( auto i = 0; i < num_sites/2-j; ++i){
      auto tn_op = exatn::makeSharedTensorNetwork("TN_OP");
      appended = tn_op->appendTensor(1, oleft, {},{}, false); assert(appended);
      appended = tn_op->appendTensor(2, omiddle, pairing1, {}, false); assert(appended);
      appended = tn_op->appendTensor(3, oright, pairing2, {}, false); assert(appended);
      for (auto iter = ket_ansatz_->begin(); iter != ket_ansatz_->end(); ++iter){
        auto & network = *(iter->network);
        appended = network.appendTensorNetwork(std::move(*tn_op),{{2*j-1,0},{2*j,2}}); assert(appended);
      //  network.printIt();
      }
    }
  }

  //reordering output modes
  std::vector<unsigned int> reorder;
  unsigned int num_output_modes = total_particles_;
  //for ( auto i = 0; i < num_output_modes; i++){
  for ( unsigned int i = 0; i < num_output_modes; i++){
    if (i%2 == 0) reorder.emplace_back(i);
  }
  
  for ( unsigned int i = num_output_modes-1; i > 0; i--){
    if (i%2 == 1) reorder.emplace_back(i);
  }

  /*
  unsigned int i = num_output_modes-1; 
  do {
    std::cout << "Done3" << std::endl;
    if (i%2 == 1) reorder.emplace_back(i);
    std::cout << "Done4" << std::endl;
    --i; 
    std::cout << "Done5" << std::endl;
  }while( i > 0);
  */
  std::cout << "reorder contains: ";
  for ( auto& x: reorder){
    std::cout << ' ' <<  x;
  }
  std::cout << std::endl;
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


void ParticleAnsatz::constructEnergyFunctional(){
  
}

void ParticleAnsatz::initWavefunctionAnsatz(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
  // normalize wavefunction ansatz
  //auto success = exatn::balanceNorm2Sync(*ket_ansatz_,1.0,true); assert(success);
}


double ParticleAnsatz::evaluateEnergyFunctional(){
  double energy = 0.0;
  return energy;
}

void ParticleAnsatz::evaluateEnergyDerivatives(){
}

void ParticleAnsatz::updateWavefunctionAnsatzTensors(){
}


} //namespace castn
