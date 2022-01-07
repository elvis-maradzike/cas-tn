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

bool ParticleAnsatz::optimize(std::size_t num_states, double convergence_thresh){

  bool success = true;
  auto hamiltonian_operator = exatn::makeSharedTensorOperator("HamiltonianOperator");
  //(anti)symmetrization
  success = hamiltonian_operator_->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = hamiltonian_operator_->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);

  //mark optimizable tensors
  markOptimizableTensors();
 
  //append ordering projectors
  appendOrderingProjectors();

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
}


void ParticleAnsatz::constructEnergyFunctional(){
  
  // components of Lagrangian Functional
  bra_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ket_ansatz_);
  bra_ansatz_->conjugate();
  bra_ansatz_->rename("BraAnsatz");
  // H expectation
  exatn::TensorExpansion tmp(*bra_ansatz_,*ket_ansatz_,*hamiltonian_operator_);
  expectation_expansion_ = std::make_shared<exatn::TensorExpansion>(tmp);
  expectation_expansion_->rename("ExpectationExpansion");
  //expectation_expansion_->printIt();
  // constraint expectation
  exatn::TensorExpansion tmp2(*bra_ansatz_,*ket_ansatz_, *constraint_operator_);
  constraint_expansion_ = std::make_shared<exatn::TensorExpansion>(tmp2);
  constraint_expansion_->rename("ConstraintExpansion");
  //constraint_expansion_->printIt();
  // augmentation expansion
  
  exatn::TensorExpansion tmp3(tmp2,tmp2);
  augmentation_expansion_ = std::make_shared<exatn::TensorExpansion>(tmp3);
  augmentation_expansion_->rename("AugmentationExpansion");
  //augmentation_expansion_->printIt();
}

void ParticleAnsatz::constructEnergyDerivatives(){
  std::cout << "Hello2" << std::endl; 
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
  std::unordered_set<std::string> tensor_names;
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        auto res = tensor_names.emplace(tensor.getName());
        if (res.second){
          exatn::TensorExpansion tmp(*expectation_expansion_,tensor.getName(),true);
          exatn::TensorExpansion tmp2(*constraint_expansion_,tensor.getName(),true);
          //exatn::TensorExpansion tmp3(*augmentation_expansion_,tensor.getName(),true);
          //augmented_lagrangian_->printIt();
         // exatn::TensorExpansion tmp4(*augmented_lagrangian_,tensor.getName(),true);
          auto gradient_expectation_expansion = std::make_shared<exatn::TensorExpansion>(tmp);
          auto gradient_constraint_expansion = std::make_shared<exatn::TensorExpansion>(tmp2);
          //auto gradient_augmentation_expansion = std::make_shared<exatn::TensorExpansion>(tmp3);
          //auto gradient_augmented_lagrangian = std::make_shared<exatn::TensorExpansion>(tmp4);
          auto gradient_expectation_tensor = std::make_shared<exatn::Tensor>("GradientExpectationTensor_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto gradient_constraint_tensor = std::make_shared<exatn::Tensor>("GradientConstraintTensor_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          //auto gradient_augmentation_tensor = std::make_shared<exatn::Tensor>("GradientAugmentationTensor_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          //auto gradient_augmented_lagrangian_tensor = std::make_shared<exatn::Tensor>("GradientAugmentedLagrangianTensor_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto created = exatn::createTensor("GradientExpectationTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          created = exatn::createTensor("GradientConstraintTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          //created = exatn::createTensor("GradientAugmentationTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          //auto created = exatn::createTensor("GradientAugmentedlagrangianTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
           derivatives_expectation_.push_back(std::make_tuple(tensor.getName(), gradient_expectation_expansion, gradient_expectation_tensor));
           derivatives_constraint_.push_back(std::make_tuple(tensor.getName(), gradient_constraint_expansion, gradient_constraint_tensor));
          //derivatives_augmentation_.push_back(std::make_tuple(tensor.getName(), gradient_augmentation_expansion, gradient_augmentation_tensor));
          //derivatives_augmented_lagrangian_.push_back(std::make_tuple(tensor.getName(), gradient_augmented_lagrangian, gradient_augmented_lagrangian_tensor));
        }
      }
    }
  }
}

void ParticleAnsatz::initWavefunctionAnsatz(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
  // normalize wavefunction ansatz
  auto success = exatn::balanceNorm2Sync(*ket_ansatz_,1.0,true); assert(success);
}


double ParticleAnsatz::evaluateEnergyFunctional(){
  double energy = 0.0;
  /*
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENSOR_SHAPE = exatn::TensorShape{};
  auto created = exatn::createTensorSync("ac_constraint",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
  auto ac_constraint = exatn::getTensor("ac_constraint");
  auto initialized = exatn::initTensor(ac_constraint->getName(), 0.0); assert(initialized);
  auto evaluated = exatn::evaluateSync(*constraint_expansion_,ac_constraint); assert(evaluated);
  auto local_copy = exatn::getLocalTensor(ac_constraint->getName()); assert(local_copy);
  const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
  auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_constraint = *body_ptr;
  body_ptr = nullptr;
  // create accumulator tensor for the closed tensor expansion
  created = exatn::createTensorSync("ac_expectation",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
  auto ac_expectation = exatn::getTensor("ac_expectation");
  initialized = exatn::initTensor(ac_expectation->getName(), 0.0); assert(initialized);
  evaluated = exatn::evaluateSync(*expectation_expansion_,ac_expectation); assert(evaluated);
  // get value
  auto talsh_tensor = exatn::getLocalTensor(ac_expectation->getName());
  access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_expectation = *body_ptr;
  std::cout << val_expectation << ", " << val_constraint << std::endl;
 
  // store value
  state_energies_.clear();
  state_energies_.push_back(val_expectation);

  // destroy accumulator tensor
  auto destroyed = exatn::destroyTensorSync(ac_expectation->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(ac_constraint->getName()); assert(destroyed);
  */
  return energy;
}

void ParticleAnsatz::evaluateEnergyDerivatives(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
  for (unsigned int i = 0; i < derivatives_expectation_.size(); i++){
    auto tensor_name = std::get<0>(derivatives_expectation_[i]);
    auto tensor = exatn::getTensor(tensor_name);
    const auto tensor_shape = tensor->getShape();
    const exatn::TensorSignature tensor_signature = tensor->getSignature();
    unsigned int rank = tensor->getRank();
    auto ac_gradient_expectation_tensor = exatn::getTensor("GradientExpectationTensor_" + tensor_name);
    auto ac_gradient_constraint_tensor = exatn::getTensor("GradientConstraintTensor_" + tensor_name);
    //auto ac_gradient_augmented_lagrangian_tensor = exatn::getTensor("GradientAugmentedLagrangianTensor_" + tensor_name);
    auto gradient_expectation_expansion = std::get<1>(derivatives_expectation_[i]);
    auto gradient_constraint_expansion = std::get<1>(derivatives_constraint_[i]);
    //auto gradient_augmented_lagrangian_expansion = std::get<1>(derivatives_augmented_lagrangian_[i]);
    auto initialized = exatn::initTensorSync(ac_gradient_expectation_tensor->getName(),0.0); assert(initialized);
    initialized = exatn::initTensorSync(ac_gradient_constraint_tensor->getName(),0.0); assert(initialized);
    //auto initialized = exatn::initTensorSync(ac_gradient_augmented_lagrangian_tensor->getName(),0.0); assert(initialized);
    auto evaluated = exatn::evaluateSync(*gradient_expectation_expansion, ac_gradient_expectation_tensor); assert(evaluated);
    evaluated = exatn::evaluateSync(*gradient_constraint_expansion, ac_gradient_constraint_tensor); assert(evaluated);
    //auto evaluated = exatn::evaluateSync(*gradient_augmented_lagrangian_expansion, ac_gradient_augmented_lagrangian_tensor); assert(evaluated);
    std::cout << " Optimizable tensor name is " << tensor_name  << std::endl;
    environments_.emplace_back(Environment{exatn::getTensor(tensor_name),
                 ac_gradient_expectation_tensor,
                 gradient_expectation_expansion,
                 ac_gradient_constraint_tensor,
                 gradient_constraint_expansion
                 });

    std::cout << "Done evaluating derivative ...." << std::endl;
  }
}

void ParticleAnsatz::updateWavefunctionAnsatzTensors(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  /*
  int maxiter = 1;
  // for every optimizable tensor
  for (auto & environment: environments_){
    const std::string tensor_name = environment.tensor->getName();
    auto tensor = exatn::getTensor(tensor_name);
    const exatn::TensorShape tensor_shape = environment.tensor->getShape();
    const exatn::TensorSignature tensor_signature = environment.tensor->getSignature();
    unsigned int rank = environment.tensor->getRank();
  
    const std::string gradient_expectation_name = environment.gradient_expectation->getName();
    const std::string gradient_constraint_name = environment.gradient_constraint->getName();
    
    // query gradient
    double max_abs_element = 0.0;
    auto success = exatn::computeMaxAbsSync(gradient_name, max_abs_element); assert(success);
    std::cout << "Max. absolute value in gradient tensor w.r.t. " << environment.tensor->getName() << " is " << max_abs_element << std::endl;
    if (max_abs_element < convergence_thresh_) continue;
    
    // microiterations
    for ( int microiteration = 0; microiteration < maxiter; microiteration++){
      // update tensor factor
      auto ac_gradient_tensor = exatn::getTensor("GradientTensor_" + tensor_name);
      auto gradient_expansion = environment.gradient_expansion;
      //auto initialized = exatn::initTensorSync(ac_gradient_tensor->getName(),0.0); assert(initialized);
      //auto evaluated = exatn::evaluateSync(gradient_expansion, ac_gradient_tensor); assert(evaluated);
      auto grad = std::make_shared<exatn::Tensor>("Grad_"+tensor_name, tensor_shape, tensor_signature);
      auto created = exatn::createTensorSync(grad,tensor->getElementType()); assert(created);
      auto initialized = exatn::initTensorSync(grad->getName(),0.0); assert(initialized);
      std::string add_pattern;
      auto generated = exatn::generate_addition_pattern(rank,add_pattern,true,grad->getName(), ac_gradient_tensor->getName()); assert(generated);
      auto factor = 1.0; 
      auto added = exatn::addTensors(add_pattern, factor); assert(added);
      add_pattern.clear();
      
      auto success = exatn::computeMaxAbsSync(grad->getName(), max_abs_element); assert(success);
      std::cout << "Max. absolute value in gradient tensor: " << max_abs_element << std::endl;
      auto destroyed = exatn::destroyTensorSync(grad->getName()); assert(destroyed);
      if (max_abs_element < convergence_thresh_) continue;
      auto prior_tensor = std::make_shared<exatn::Tensor>("PriorTensor_"+tensor_name,tensor_shape,tensor_signature);
      created = exatn::createTensorSync(prior_tensor,tensor->getElementType()); assert(created);
      //std::string add_pattern;
      initialized = exatn::initTensor(prior_tensor->getName(), 0.0); assert(initialized);
      generated = exatn::generate_addition_pattern(rank,add_pattern,true,prior_tensor->getName(), tensor_name); assert(generated);
      std::cout << add_pattern << std::endl;
      added = exatn::addTensors(add_pattern,1.0); assert(added);
      add_pattern.clear();
    
      initialized = exatn::initTensor(tensor_name, 0.0); assert(initialized);
      generated = exatn::generate_addition_pattern(rank,add_pattern,true,tensor_name, prior_tensor->getName()); assert(generated);
      std::cout << add_pattern << std::endl;
      added = exatn::addTensors(add_pattern,1.0); assert(added);
      add_pattern.clear();
    
      generated = exatn::generate_addition_pattern(rank,add_pattern,true,tensor->getName(), gradient_name); assert(generated);
      std::cout << add_pattern << std::endl;
      auto lr = 0.5;
      added = exatn::addTensors(add_pattern, -lr); assert(added);
      add_pattern.clear();

      std::cout << " Done updating tensor "  << tensor_name << std::endl;
      destroyed = exatn::destroyTensorSync(prior_tensor->getName()); assert(destroyed);
    }
  } 
  environments_.clear();
  */
}


} //namespace castn
