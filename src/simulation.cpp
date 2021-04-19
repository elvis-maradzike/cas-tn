/** Complete-Active-Space Tensor-Network (CAS-TN) Simulation
REVISION: 2021/01/08

Copyright (C) 2020-2021 Dmitry I. Lyakh (Liakh), Elvis Maradzike
Copyright (C) 2020-2021 Oak Ridge National Laboratory (UT-Battelle)

**/

#include "simulation.hpp"
#include "exatn.hpp"
#include "talshxx.hpp"
#include <unordered_set>

namespace castn {

void Simulation::clear()
{
 state_energies_.clear();
 derivatives_.clear();
 functional_.reset();
 return;
}


void Simulation::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorNetwork> ansatz)
{
 clear();
 bra_ansatz_.reset();
 ket_ansatz_.reset();
 ket_ansatz_ = std::make_shared<exatn::TensorExpansion>();
 ket_ansatz_->appendComponent(ansatz,{1.0,0.0});
 return;
}


void Simulation::resetWaveFunctionAnsatz(std::shared_ptr<exatn::TensorExpansion> ansatz)
{
 clear();
 bra_ansatz_.reset();
 ket_ansatz_.reset();
 ket_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ansatz);
 return;
}


void Simulation::resetWaveFunctionAnsatz(exatn::NetworkBuilder & ansatz_builder)
{
 clear();
 //`Implement
 return;
}


void Simulation::resetHamiltonian(const std::vector<std::shared_ptr<exatn::Tensor>> & hamiltonian)
{
 clear();
 hamiltonian_ = hamiltonian;
 return;
}

void Simulation::markOptimizableTensors(){
  for (auto component = ket_ansatz_->begin(); component != ket_ansatz_->end(); ++component){
    component->network->markOptimizableAllTensors();
  }
}

bool Simulation::optimize(std::size_t num_states, double convergence_thresh){
 
  // print out some optimization parameters 
  std::cout << "Starting the optimization procedure" << std::endl;
  std::cout << "Number of spin orbitals: " << total_orbitals_ << std::endl;
  std::cout << "Number of particles: " << num_particles_ << std::endl;
  std::cout << "Number of states: " << num_states_ << std::endl;
  std::cout << "Convergence threshold: " << convergence_thresh_ << std::endl;

  // hamiltonian
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENS_SHAPE_0INDEX = exatn::TensorShape{};
  const auto TENS_SHAPE_2INDEX = exatn::TensorShape{total_orbitals_, total_orbitals_};
  const auto TENS_SHAPE_4INDEX = exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_};

  auto h1 = std::make_shared<exatn::Tensor>("H1", TENS_SHAPE_2INDEX);
  auto h2 = std::make_shared<exatn::Tensor>("H2", TENS_SHAPE_4INDEX);

  auto created = exatn::createTensorSync("AC_Ham_Ket",TENS_ELEM_TYPE, TENS_SHAPE_4INDEX); assert(created);
  auto ac_ham_ket = exatn::getTensor("AC_Ham_Ket");
  auto rank = ac_ham_ket->getRank();
  created = exatn::createTensorSync("AC_Ket",TENS_ELEM_TYPE, TENS_SHAPE_4INDEX); assert(created);
  auto ac_ket = exatn::getTensor("AC_Ket");

  created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);
  
  auto initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

  hamiltonian_.push_back(h2);
  hamiltonian_.push_back(h1);

  markOptimizableTensors();
  appendOrderingProjectors();
  constructEnergyFunctional();
  constructEnergyDerivatives();
  initWavefunctionAnsatz();

  double maxAbsResidual = 0.0;
  int iter = 0;
  do {
    
    double val_norm = 0.0;
    evaluateEnergyFunctional();
    // evaluate residual tensor
    auto initialized = exatn::initTensor(ac_ham_ket->getName(), 0.0); assert(initialized);
    auto evaluated = exatn::evaluateSync(*ham_ket_,ac_ham_ket); assert(evaluated);
    auto local_copy = exatn::getLocalTensor(ac_ham_ket->getName()); assert(local_copy);
    //auto success = exatn::printTensorSync(ac_ham_ket->getName()); assert(success);
    initialized = exatn::initTensor(ac_ket->getName(), 0.0); assert(initialized);
    evaluated = exatn::evaluateSync(*ket_ansatz_,ac_ket); assert(evaluated);
    
    std::string add_pattern;
    auto generated = exatn::generate_addition_pattern(rank,add_pattern,true,ac_ham_ket->getName(), ac_ket->getName()); assert(generated);
    std::cout << add_pattern << std::endl;
    double factor = state_energies_[0];
    auto added = exatn::addTensors(add_pattern, -factor); assert(added);
    add_pattern.clear();
   
    // update ket
    for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
      for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
        const auto & tensor = tensor_conn->second;
        if (tensor.isOptimizable()){

          created = exatn::createTensorSync("AC_Ket_Prior",TENS_ELEM_TYPE, tensor.getShape()); assert(created);
          auto tensor_prior = exatn::getTensor("AC_Ket_Prior");
          initialized = exatn::initTensor(tensor_prior->getName(), 0.0); assert(initialized);
          std::string add_pattern;
          auto generated = exatn::generate_addition_pattern(tensor_prior->getRank(),add_pattern,true,tensor_prior->getName(), tensor.getName()); assert(generated);
          std::cout << add_pattern << std::endl;
          auto added = exatn::addTensors(add_pattern, 1.0); assert(added);
          add_pattern.clear();
          initialized = exatn::initTensor(tensor.getName(), 0.0); assert(initialized);
          generated = exatn::generate_addition_pattern(tensor.getRank(),add_pattern,true,tensor.getName(), tensor_prior->getName()); assert(generated);
          std::cout << add_pattern << std::endl;
          added = exatn::addTensors(add_pattern, 1.0); assert(added);
          add_pattern.clear();
          generated = exatn::generate_addition_pattern(tensor.getRank(),add_pattern,true,tensor.getName(), ac_ham_ket->getName()); assert(generated);
          std::cout << add_pattern << std::endl;
          double learning_rate = 0.5;
          added = exatn::addTensors(add_pattern, -learning_rate); assert(added);
          add_pattern.clear();    
          
          // renormalize ket ansatz
          auto created = exatn::createTensorSync("AC_Norm",TENS_ELEM_TYPE, TENS_SHAPE_0INDEX); assert(created);
          auto ac_norm = exatn::getTensor("AC_Norm");
          initialized = exatn::initTensor(ac_norm->getName(), 0.0); assert(initialized);
          auto evaluated = exatn::evaluateSync(*norm_,ac_norm); assert(evaluated);
          auto local_copy = exatn::getLocalTensor(ac_norm->getName()); assert(local_copy);
          const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
          auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
          val_norm = *body_ptr;
          body_ptr = nullptr;
          double factor = sqrt(1.0/val_norm);
          auto scaled = exatn::scaleTensor(tensor.getName(), factor); assert(scaled);

          maxAbsResidual = 0.0;          
          auto success = exatn::computeMaxAbsSync(ac_ham_ket->getName(), maxAbsResidual); assert(success);
          auto destroyed = exatn::destroyTensorSync(ac_norm->getName()); assert(destroyed);
          destroyed = exatn::destroyTensorSync(tensor_prior->getName()); assert(destroyed);
        } 
      }
    }

    std::cout << std::setw(7) << std::right <<  " iter: " <<
          std::setw(7) << std::right << iter <<
          std::setw(12) << std::fixed << state_energies_[0] <<
          std::setw(12) << std::fixed << maxAbsResidual <<
          std::setw(12) << std::fixed << val_norm << std::endl;

   iter++;  
  } while (maxAbsResidual > 1e-7);

  return true;
}

/** Appends two layers of ordering projectors to the wavefunction ansatz. **/
void Simulation::appendOrderingProjectors(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENS_SHAPE_2INDEX = exatn::TensorShape{total_orbitals_, total_orbitals_};
  const auto TENS_SHAPE_4INDEX = exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_};
  // array of elements to initialize tensors that comprise the ordering projectors
  std::vector<double> tmpData(total_orbitals_*total_orbitals_*total_orbitals_*total_orbitals_,0.0);
  for ( unsigned int i = 0; i < total_orbitals_; i++){
    for ( unsigned int j = 0; j < total_orbitals_; j++){
      for ( unsigned int k = 0; k < total_orbitals_; k++){
        for ( unsigned int l = 0; l < total_orbitals_; l++){
          if ( i < j){
          tmpData[i*total_orbitals_*total_orbitals_*total_orbitals_
                 +j*total_orbitals_*total_orbitals_
                 +k*total_orbitals_+l] = double((i==k)*(j==l));
          }else{
          tmpData[i*total_orbitals_*total_orbitals_*total_orbitals_
                 +j*total_orbitals_*total_orbitals_
                 +k*total_orbitals_+l] = 0.0;
          }
        }
      }
    }
  }
 
  //create tensors for ordering projectors
  auto created = exatn::createTensor("Q", TENS_ELEM_TYPE, TENS_SHAPE_4INDEX); assert(created);
  auto initialized = exatn::initTensorData("Q", tmpData); assert(initialized);
  
  auto appended = false;
  for (auto iter = (*ket_ansatz_).begin(); iter != (*ket_ansatz_).end(); ++iter){
    auto & network = *(iter->network);
    unsigned int tensorCounter = 1 + network.getNumTensors();
    // applying layer 1
    for ( unsigned int i = 0; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      tensorCounter++;
    }
    // applying layer 2
    for ( unsigned int i = 1; i < num_particles_-1; i=i+2){
      appended = network.appendTensorGate(tensorCounter, exatn::getTensor("Q"),{i,i+1}); assert(appended);
      tensorCounter++;
    }
  }

  ket_ansatz_->printIt();
}


void Simulation::constructEnergyFunctional(){

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  bool created = false, success = false, destroyed = false;
  bra_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ket_ansatz_);
  bra_ansatz_->conjugate();
  bra_ansatz_->printIt();

  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
  auto appended = false;
  // hamiltonian tensors 
  success = ham->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = ham->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);

  // energy functional as closed product
  exatn::TensorExpansion xHx(*bra_ansatz_,*ket_ansatz_,*ham);
  functional_ = std::make_shared<exatn::TensorExpansion>(xHx);
  functional_->printIt();
  
  exatn::TensorExpansion norm(*bra_ansatz_,*ket_ansatz_);
  norm_ = std::make_shared<exatn::TensorExpansion>(norm);
  norm_->printIt();

  exatn::TensorExpansion ham_ket(*ket_ansatz_, *ham);
  ham_ket_ = std::make_shared<exatn::TensorExpansion>(ham_ket);
  ham_ket_->printIt();

}

void Simulation::constructEnergyDerivatives(){
  
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  std::unordered_set<std::string> tensor_names;
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        auto res = tensor_names.emplace(tensor.getName()); 
        if (res.second){
          auto created = exatn::createTensor("GradientTensor_xHx_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          exatn::TensorExpansion tmp1(*functional_,tensor.getName(),true);
          tmp1.rename("GradientTensor_xHx_"+tensor.getName());
          auto gradient_expansion_xHx = std::make_shared<exatn::TensorExpansion>(tmp1);
          auto gradient_tensor_xHx = std::make_shared<exatn::Tensor>((*exatn::getTensor("GradientTensor_xHx_" + tensor.getName())));
          derivatives_.push_back(std::make_tuple(tensor.getName(), gradient_expansion_xHx, gradient_tensor_xHx));

        }
      } 
    }
  }
}

void Simulation::initWavefunctionAnsatz(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        //auto initialized = exatn::initTensorRnd(tensor.getName()); assert(initialized);
        auto initialized = exatn::initTensorFile(tensor.getName(),"h4_tensor.txt"); assert(initialized);
       
      } 
    }
  }
  
  // evaluate norm of wavefunction
  auto created = exatn::createTensorSync("AC_Norm", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
  auto ac_norm = exatn::getTensor("AC_Norm");
  auto initialized = exatn::initTensor(ac_norm->getName(), 0.0); assert(initialized);
  auto evaluated = exatn::evaluateSync(*norm_,ac_norm); assert(evaluated);
  
  auto local_copy = exatn::getLocalTensor(ac_norm->getName()); assert(local_copy);
  const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
  auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_norm = *body_ptr;
  body_ptr = nullptr;
 
  std::cout << "Norm before renormalization is: " << val_norm << std::endl; 
  auto success = exatn::printTensorSync(ac_norm->getName()); assert(success);

  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        double factor = sqrt(1.0/val_norm);
        auto scaled = exatn::scaleTensor(tensor.getName(), factor); assert(scaled);
      } 
    }
  }

  // reevaluate norm functional
  created = exatn::createTensorSync("AC_Norm_Post", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
  auto ac_norm_post = exatn::getTensor("AC_Norm_Post");
  initialized = exatn::initTensor(ac_norm_post->getName(), 0.0); assert(initialized);
  evaluated = exatn::evaluateSync(*norm_,ac_norm_post); assert(evaluated);
  success = exatn::printTensorSync(ac_norm_post->getName()); assert(success);

  auto destroyed = exatn::destroyTensorSync(ac_norm->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(ac_norm_post->getName()); assert(destroyed);
  std::cout << "Wavefunction ansatz has been initialized and normalized ....." << std::endl;
}


double Simulation::evaluateEnergyFunctional(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENSOR_SHAPE = exatn::TensorShape{};
  auto created = exatn::createTensorSync("AC_Norm",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
  auto ac_norm = exatn::getTensor("AC_Norm");
  auto initialized = exatn::initTensor(ac_norm->getName(), 0.0); assert(initialized);
  auto evaluated = exatn::evaluateSync(*norm_,ac_norm); assert(evaluated);
  std::cout << " Norm has been evaluated ....." << std::endl;
  auto local_copy = exatn::getLocalTensor(ac_norm->getName()); assert(local_copy);
  const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
  auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_norm = *body_ptr;
  body_ptr = nullptr;
  std::cout << " Norm value has been recorded ....." << std::endl;
  auto success = exatn::printTensorSync(ac_norm->getName()); assert(success);
  
  // create accumulator tensor for the closed tensor expansion
  created = exatn::createTensorSync("AC_xHx",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
  auto ac_xHx = exatn::getTensor("AC_xHx");
  initialized = exatn::initTensor(ac_xHx->getName(), 0.0); assert(initialized);
  evaluated = exatn::evaluateSync(*functional_,ac_xHx); assert(evaluated);
  std::cout << " Energy value has been evaluated ....." << std::endl;
  // get value 
  double val_xHx = 0.0;

  auto talsh_tensor = exatn::getLocalTensor(ac_xHx->getName());
  access_granted = false;
  if (talsh_tensor->getDataAccessHostConst(&body_ptr)){
    for ( int i = 0; i < talsh_tensor->getVolume(); i++){
      val_xHx = body_ptr[i];
    }
  }
  body_ptr = nullptr;

  //success = exatn::computeNorm2Sync(ac_xHx->getName(), val_xHx); assert(success);
  std::cout << " Energy value has been recorded ....." << std::endl;

  double energy = 0.0;
  energy = val_xHx/val_norm;
  std::cout << " (E, Norm): " << energy << ", " << val_norm << std::endl;

  // store value
  state_energies_.clear();
  state_energies_.push_back(energy);

  // destroy accumulator tensor
  auto destroyed = exatn::destroyTensorSync(ac_xHx->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(ac_norm->getName()); assert(destroyed);
  return energy;
}

void Simulation::evaluateEnergyDerivatives(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  //  derivative: <Psi|H|Psi>
  for (unsigned int i = 0; i < derivatives_.size(); i++){
    auto tensor_name = std::get<0>(derivatives_[i]);
    auto tensor = exatn::getTensor(tensor_name);
    const auto tensor_shape = tensor->getShape();
    const exatn::TensorSignature tensor_signature = tensor->getSignature();
    unsigned int rank = tensor->getRank();
    auto ac_gradient_tensor_xHx = exatn::getTensor("GradientTensor_xHx_" + tensor_name);
    auto gradient_expansion_xHx = std::get<1>(derivatives_[i]);
    std::cout << "Expansion, Tensor: " << gradient_expansion_xHx->getName() << ", " << ac_gradient_tensor_xHx->getName() << std::endl;
    auto initialized = exatn::initTensorSync(ac_gradient_tensor_xHx->getName(),0.0); assert(initialized);
    auto evaluated = exatn::evaluateSync(*gradient_expansion_xHx, ac_gradient_tensor_xHx); assert(evaluated);
    std::cout << " Optimizable tensor name is " << tensor_name  << std::endl;
    environments_.emplace_back(Environment{exatn::getTensor(tensor_name),
                 ac_gradient_tensor_xHx,
                 *gradient_expansion_xHx
                 });
   
    std::cout << "Done evaluating derivative ...." << std::endl;   
    auto destroyed = exatn::destroyTensorSync(ac_gradient_tensor_xHx->getName()); assert(destroyed);
  }
}
  
void Simulation::updateWavefunctionAnsatzTensors(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  // for every optimizable tensoer
  for (auto & environment: environments_){
    const std::string tensor_name = environment.tensor->getName();
    auto tensor = exatn::getTensor(tensor_name);
    const std::string gradient_name = environment.gradient->getName();
    const exatn::TensorShape tensor_shape = environment.tensor->getShape();
    const exatn::TensorSignature tensor_signature = environment.tensor->getSignature();
    unsigned int rank = environment.tensor->getRank();

    std::cout << " Done updating tensors "  << tensor_name << std::endl;
  } 
  std::cout << " Clearing environments "  << std::endl;
  environments_.clear();
  std::cout << " Done Clearing environments "  << std::endl;
}
 
} //namespace castn
