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
 
  //printing out some optimization parameters 
  std::cout << "Starting the optimization procedure" << std::endl;
  std::cout << "Number of spin orbitals: " << total_orbitals_ << std::endl;
  std::cout << "Number of particles: " << num_particles_ << std::endl;
  std::cout << "Number of states: " << num_states_ << std::endl;
  std::cout << "Convergence threshold: " << convergence_thresh_ << std::endl;

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENS_SHAPE_0INDEX = exatn::TensorShape{};
  const auto TENS_SHAPE_2INDEX = exatn::TensorShape{total_orbitals_, total_orbitals_};
  const auto TENS_SHAPE_4INDEX = exatn::TensorShape{total_orbitals_,total_orbitals_,total_orbitals_,total_orbitals_};

  //hamiltonian tensors
  auto h1 = std::make_shared<exatn::Tensor>("H1", TENS_SHAPE_2INDEX);
  auto h2 = std::make_shared<exatn::Tensor>("H2", TENS_SHAPE_4INDEX);

  auto created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
  created = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(created);
  
  auto initialized = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(initialized);
  initialized = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(initialized);

  hamiltonian_.push_back(h2);
  hamiltonian_.push_back(h1);

  auto ham = exatn::makeSharedTensorOperator("Hamiltonian");
  auto appended = false;
 
  //(anti)symmetrization 
  auto success = ham->appendSymmetrizeComponent(hamiltonian_[0],{0,1},{2,3}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);
  success = ham->appendSymmetrizeComponent(hamiltonian_[1],{0},{1}, num_particles_, num_particles_,{1.0,0.0},true); assert(success);

  // derivative tensors:
    //<x|H|x>
  created = exatn::createTensorSync("AC_Deriv_xHx",TENS_ELEM_TYPE, TENS_SHAPE_4INDEX); assert(created);
  auto ac_d_xHx = exatn::getTensor("AC_Deriv_xHx");
  auto rank = ac_d_xHx->getRank();
    //<x|S|x>
  created = exatn::createTensorSync("AC_Deriv_xSx",TENS_ELEM_TYPE, TENS_SHAPE_4INDEX); assert(created);
  auto ac_d_xSx = exatn::getTensor("AC_Deriv_xSx");

  // functional tensors:
    //<x|H|x>
  created = exatn::createTensorSync("AC_xHx",TENS_ELEM_TYPE, TENS_SHAPE_0INDEX); assert(created);
  auto ac_xHx = exatn::getTensor("AC_xHx");
    //<x|S|x>
  created = exatn::createTensorSync("AC_xSx",TENS_ELEM_TYPE, TENS_SHAPE_0INDEX); assert(created);
  auto ac_xSx = exatn::getTensor("AC_xSx");
  //marking optimizable tensors
  markOptimizableTensors();
  
  //appending ordering projectors
  appendOrderingProjectors();
 
  ket_ansatz_->rename("VectorExpansion"); 
  //bra ansatz
  bra_ansatz_ = std::make_shared<exatn::TensorExpansion>(*ket_ansatz_);
  bra_ansatz_->rename(ket_ansatz_->getName()+"Bra");
  bra_ansatz_->conjugate();
  bra_ansatz_->printIt();
  
  //functional expansions as closed products
    //<x|H|x>
  exatn::TensorExpansion xHx(*bra_ansatz_,*ket_ansatz_,*ham);
  functional_ = std::make_shared<exatn::TensorExpansion>(xHx);
  functional_->printIt();
    //<x|S|x>
  exatn::TensorExpansion l2_norm_squared(*bra_ansatz_,*ket_ansatz_);
  l2_norm_squared_ = std::make_shared<exatn::TensorExpansion>(l2_norm_squared);




  //initializing optimizable tensors
  initWavefunctionAnsatz();

  double energy = 0.0;
  initialized = exatn::initTensor(ac_xSx->getName(), 0.0); assert(initialized);
  auto evaluated = exatn::evaluateSync(*l2_norm_squared_,ac_xSx); assert(evaluated);
  std::cout << " Norm has been evaluated ....." << std::endl;
  auto local_copy = exatn::getLocalTensor(ac_xSx->getName()); assert(local_copy);
  const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
  auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_norm = *body_ptr;
  body_ptr = nullptr;
  std::cout << " Norm value has been recorded ....." << std::endl;
  //auto success = exatn::printTensorSync(ac_norm->getName()); assert(success);
  
  
  // create accumulator tensor for the closed tensor expansion
  initialized = exatn::initTensor(ac_xHx->getName(), 0.0); assert(initialized);
  evaluated = exatn::evaluateSync(*functional_,ac_xHx); assert(evaluated);
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

  energy = val_xHx/val_norm;

  // evaluating derivative, w.r.t. optimizable tensors available 
  std::unordered_set<std::string> tensor_names;
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        auto res = tensor_names.emplace(tensor.getName()); 
        if (res.second){
          exatn::TensorExpansion tmp1d(*functional_,tensor.getName(),true);
          exatn::TensorExpansion tmp2d(*l2_norm_squared_,tensor.getName(),true);
          initialized = exatn::initTensor(ac_d_xHx->getName(), 0.0); assert(initialized);
          auto evaluated = exatn::evaluateSync(tmp1d,ac_d_xHx); assert(evaluated);
          initialized = exatn::initTensor(ac_d_xSx->getName(), 0.0); assert(initialized);
          evaluated = exatn::evaluateSync(tmp2d,ac_d_xSx); assert(evaluated);
          std::cout << ".......... deriv(x|H|x>) ............" << std::endl; 
          success = exatn::printTensorSync(ac_d_xHx->getName()); assert(success);
          std::cout << ".......... deriv(x|S|x>) ............" << std::endl; 
          success = exatn::printTensorSync(ac_d_xSx->getName()); assert(success);
          std::string add_pattern;
          auto generated = exatn::generate_addition_pattern(rank,add_pattern,true,ac_d_xHx->getName(), ac_d_xSx->getName()); assert(generated);
          std::cout << add_pattern << std::endl;
          auto added = exatn::addTensors(add_pattern,-energy); assert(added);
          add_pattern.clear();
          //printing out derivative
          std::cout << ".......... deriv(x|H|x>) ............" << std::endl; 
          success = exatn::printTensorSync(ac_d_xHx->getName()); assert(success);
          std::cout << ".......... deriv(x|H|x>)-deriv(<x|S|x>) ............" << std::endl; 
          //success = exatn::printTensorSync(ac_d_xHx->getName()); assert(success);
          // computing norm of derivative
          double dum = 0.;
          success = exatn::computeMaxAbsSync(ac_d_xHx->getName(), dum); assert(success);
          std::cout << "L2 Norm of gradient is: " << dum << std::endl;
        }
      } 
    }
  }
  

  // setting up and calling the optimizer in ../src/exatn/..
  exatn::TensorNetworkOptimizer::resetDebugLevel(1);
  exatn::TensorNetworkOptimizer optimizer(ham,ket_ansatz_,1e-4);
  optimizer.resetLearningRate(0.9);
  optimizer.resetDebugLevel(2);
  bool converged = optimizer.optimize();
  success = exatn::sync(); assert(success);
  if(converged){
   std::cout << "Optimization succeeded!" << std::endl;
  }else{
   std::cout << "Optimization failed!" << std::endl; assert(false);
  }

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
  std::cout << "Completed apending ordering projectors " << std::endl;
}


void Simulation::constructEnergyFunctional(){
  
  /*
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
  */

}

void Simulation::constructEnergyDerivatives(){
  /*
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  std::unordered_set<std::string> tensor_names;
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        auto res = tensor_names.emplace(tensor.getName()); 
        if (res.second){
          auto g_xHx_tensor = std::make_shared<exatn::Tensor>("GradientTensor_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto g_xx_tensor = std::make_shared<exatn::Tensor>("GradientAuxTensor_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto h_xHx_rank0_tensor = exatn::makeSharedTensor("HessianTensor_"+tensor.getName());
          auto h_xx_rank0_tensor = exatn::makeSharedTensor("HessianAuxTensor_"+tensor.getName());
          auto created = exatn::createTensor("GradientTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
		  created = exatn::createTensor("GradientAuxTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          created = exatn::createTensor("HessianTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          created = exatn::createTensor("HessianAuxTensor_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          exatn::TensorExpansion tmp1d(*functional_,tensor.getName(),true);
          exatn::TensorExpansion tmp2d(*norm_,tensor.getName(),true);
          exatn::TensorExpansion tmp3d(*functional_,tensor.getTensor(),g_xHx_tensor);
          exatn::TensorExpansion tmp4d(*norm_,tensor.getTensor(),g_xx_tensor);
          auto gradient_expansion = std::make_shared<exatn::TensorExpansion>(tmp1d);
          auto gradient_aux_expansion = std::make_shared<exatn::TensorExpansion>(tmp2d);
          auto hessian_expansion = std::make_shared<exatn::TensorExpansion>(tmp3d);
          auto hessian_aux_expansion = std::make_shared<exatn::TensorExpansion>(tmp4d);
          derivatives_.push_back(std::make_tuple(tensor.getName(), gradient_expansion, gradient_aux_expansion, hessian_expansion, hessian_aux_expansion, g_xHx_tensor, g_xx_tensor, h_xHx_rank0_tensor, h_xx_rank0_tensor));

        }
      } 
    }
  }
  */
}

void Simulation::initWavefunctionAnsatz(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        //auto initialized = exatn::initTensorRnd(tensor.getName()); assert(initialized);
        auto initialized = exatn::initTensorFile(tensor.getName(),"h4_tensor.txt"); assert(initialized);
        std::cout << "Initializing tensor " << tensor.getName() << "..." << std::endl;
      } 
    }
  }
  std::cout << " Done initializing all optimizable tensors " << std::endl;
 
  
  // evaluate norm of wavefunction
  auto created = exatn::createTensorSync("AC_Norm", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
  auto ac_norm = exatn::getTensor("AC_Norm");
  auto initialized = exatn::initTensor(ac_norm->getName(), 0.0); assert(initialized);
  auto evaluated = exatn::evaluateSync(*l2_norm_squared_,ac_norm); assert(evaluated);
  
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
        // number of tensors in network less the number of ordering projectors
        int num_optimizable = network->network->getNumTensors() - (num_particles_-1);
        std::cout << "Number of optimizable tensors: " << num_optimizable << std::endl;
        double factor = pow(sqrt(1.0/val_norm), 1.0/double(num_optimizable));
       // double factor = pow(sqrt(1.0/val_norm), 1.0/double(1.0));
        auto scaled = exatn::scaleTensor(tensor.getName(), factor); assert(scaled);
      } 
    }
  }

  // reevaluate norm functional
  created = exatn::createTensorSync("AC_Norm_Post", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
  auto ac_norm_post = exatn::getTensor("AC_Norm_Post");
  initialized = exatn::initTensor(ac_norm_post->getName(), 0.0); assert(initialized);
  evaluated = exatn::evaluateSync(*l2_norm_squared_,ac_norm_post); assert(evaluated);
  success = exatn::printTensorSync(ac_norm_post->getName()); assert(success);

  auto destroyed = exatn::destroyTensorSync(ac_norm->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(ac_norm_post->getName()); assert(destroyed);
  std::cout << "Wavefunction ansatz has been initialized and normalized ....." << std::endl;
  
}


double Simulation::evaluateEnergyFunctional(){
  double energy = 0.0;
  /*
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

  energy = val_xHx/val_norm;
  std::cout << " (E, Norm): " << energy << ", " << val_norm << std::endl;

  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        int num_optimizable = network->network->getNumTensors() - (num_particles_-1);
        std::cout << "Number of optimizable tensors: " << num_optimizable << std::endl;
        double factor = pow(sqrt(1.0/val_norm), 1.0/double(num_optimizable));
        auto scaled = exatn::scaleTensor(tensor.getName(), factor); assert(scaled);
      } 
    }
  }
  // store value
  state_energies_.clear();
  state_energies_.push_back(energy);

  // destroy accumulator tensor
  auto destroyed = exatn::destroyTensorSync(ac_xHx->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(ac_norm->getName()); assert(destroyed);
  */
  return energy;
}

void Simulation::evaluateEnergyDerivatives(){
  /*
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  //  derivative: <Psi|H|Psi>
  for (unsigned int i = 0; i < derivatives_.size(); i++){
    auto tensor_name = std::get<0>(derivatives_[i]);
    auto tensor = exatn::getTensor(tensor_name);
    const auto tensor_shape = tensor->getShape();
    const exatn::TensorSignature tensor_signature = tensor->getSignature();
    unsigned int rank = tensor->getRank();
    auto ac_gradient_tensor = exatn::getTensor("GradientTensor_" + tensor_name);
    auto ac_gradient_aux_tensor = exatn::getTensor("GradientAuxTensor_" + tensor_name);
    auto ac_hessian_tensor = exatn::getTensor("HessianTensor_" + tensor_name);
    auto ac_hessian_aux_tensor = exatn::getTensor("HessianAuxTensor_" + tensor_name);
    auto gradient_expansion = std::get<1>(derivatives_[i]);
    auto gradient_aux_expansion = std::get<2>(derivatives_[i]);
    auto hessian_expansion = std::get<3>(derivatives_[i]);
    auto hessian_aux_expansion = std::get<4>(derivatives_[i]);
    auto initialized = exatn::initTensorSync(ac_gradient_tensor->getName(),0.0); assert(initialized);
    initialized = exatn::initTensorSync(ac_gradient_aux_tensor->getName(),0.0); assert(initialized);
    initialized = exatn::initTensorSync(ac_hessian_tensor->getName(),0.0); assert(initialized);
    initialized = exatn::initTensorSync(ac_hessian_aux_tensor->getName(),0.0); assert(initialized);
    auto evaluated = exatn::evaluateSync(*gradient_expansion, ac_gradient_tensor); assert(evaluated);
    evaluated = exatn::evaluateSync(*gradient_aux_expansion, ac_gradient_aux_tensor); assert(evaluated);
    //evaluated = exatn::evaluateSync(*hessian_expansion, ac_hessian_tensor); assert(evaluated);
    //evaluated = exatn::evaluateSync(*hessian_aux_expansion, ac_hessian_aux_tensor); assert(evaluated);
    // compute norm and get e0
    auto created = exatn::createTensorSync("AC_Norm",TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
    auto ac_norm = exatn::getTensor("AC_Norm");
    initialized = exatn::initTensor(ac_norm->getName(), 0.0); assert(initialized);
    evaluated = exatn::evaluateSync(*norm_,ac_norm); assert(evaluated);
    auto local_copy = exatn::getLocalTensor(ac_norm->getName()); assert(local_copy);
    const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
    auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
    double val_norm = *body_ptr;
    body_ptr = nullptr;

    auto grad = std::make_shared<exatn::Tensor>("Grad_"+tensor_name, tensor_shape, tensor_signature);
    created = exatn::createTensorSync(grad,tensor->getElementType()); assert(created);
    initialized = exatn::initTensorSync(grad->getName(),0.0); assert(initialized);
    std::string add_pattern;
    auto generated = exatn::generate_addition_pattern(rank,add_pattern,true,grad->getName(), ac_gradient_tensor->getName()); assert(generated);
    auto factor = 1.0/val_norm; 
    auto added = exatn::addTensors(add_pattern, factor); assert(added);
    add_pattern.clear();
    generated = exatn::generate_addition_pattern(rank,add_pattern,true,grad->getName(), ac_gradient_aux_tensor->getName()); assert(generated);
    factor = state_energies_[0]/val_norm;
    added = exatn::addTensors(add_pattern, -factor); assert(added);
    add_pattern.clear();
    
    initialized = exatn::initTensorSync(ac_gradient_tensor->getName(),0.0); assert(initialized);
    generated = exatn::generate_addition_pattern(rank,add_pattern,true,ac_gradient_tensor->getName(), grad->getName()); assert(generated);
    factor = 1.0;
    added = exatn::addTensors(add_pattern, factor); assert(added);
    add_pattern.clear();
    

    //auto success = exatn::printTensorSync(ac_gradient_tensor->getName()); assert(success);
    std::cout << " Optimizable tensor name is " << tensor_name  << std::endl;
    environments_.emplace_back(Environment{exatn::getTensor(tensor_name),
                 ac_gradient_tensor,
                 *gradient_expansion,
                 *gradient_aux_expansion
                 });
   
    std::cout << "Done evaluating derivative ...." << std::endl;   
    auto destroyed = exatn::destroyTensorSync(grad->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(ac_norm->getName()); assert(destroyed);
  }
  */
}
  
void Simulation::updateWavefunctionAnsatzTensors(){
  /*
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  int maxiter = 1;
  // for every optimizable tensoer
  for (auto & environment: environments_){
    const std::string tensor_name = environment.tensor->getName();
    auto tensor = exatn::getTensor(tensor_name);
    const std::string gradient_name = environment.gradient->getName();
    const exatn::TensorShape tensor_shape = environment.tensor->getShape();
    const exatn::TensorSignature tensor_signature = environment.tensor->getSignature();
    unsigned int rank = environment.tensor->getRank();
    // query gradient tensor
    double maxAbsElement = 0.0;
    auto success = exatn::computeMaxAbsSync(gradient_name, maxAbsElement); assert(success);
    std::cout << "Max. absolute value in gradient tensor w.r.t. " << tensor_name << " is " << maxAbsElement << std::endl;
    //success = exatn::printTensorSync(gradient_name); assert(success);
    if (maxAbsElement < convergence_thresh_) continue;
    
    // microiterations
    if (iter >= 10){
      maxiter = 5;
    }
    for ( int microiteration = 0; microiteration < maxiter; microiteration++){
      // update tensor factor
      auto ac_gradient_tensor = exatn::getTensor("GradientTensor_" + tensor_name);
      auto gradient_expansion = environment.gradient_expansion;
      auto initialized = exatn::initTensorSync(ac_gradient_tensor->getName(),0.0); assert(initialized);
      auto evaluated = exatn::evaluateSync(gradient_expansion, ac_gradient_tensor); assert(evaluated);
      auto ac_gradient_aux_tensor = exatn::getTensor("GradientAuxTensor_" + tensor_name);
      auto gradient_aux_expansion = environment.gradient_aux_expansion;
      initialized = exatn::initTensorSync(ac_gradient_aux_tensor->getName(),0.0); assert(initialized);
      evaluated = exatn::evaluateSync(gradient_aux_expansion, ac_gradient_aux_tensor); assert(evaluated);
      // compute norm and get e0
      auto created = exatn::createTensorSync("AC_Norm",TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
      auto ac_norm = exatn::getTensor("AC_Norm");
      initialized = exatn::initTensor(ac_norm->getName(), 0.0); assert(initialized);
      evaluated = exatn::evaluateSync(*norm_,ac_norm); assert(evaluated);
      auto local_copy = exatn::getLocalTensor(ac_norm->getName()); assert(local_copy);
      const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
      auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
      double val_norm = *body_ptr;
      body_ptr = nullptr;
    
      auto grad = std::make_shared<exatn::Tensor>("Grad_"+tensor_name, tensor_shape, tensor_signature);
      created = exatn::createTensorSync(grad,tensor->getElementType()); assert(created);
      initialized = exatn::initTensorSync(grad->getName(),0.0); assert(initialized);
      std::string add_pattern;
      auto generated = exatn::generate_addition_pattern(rank,add_pattern,true,grad->getName(), ac_gradient_tensor->getName()); assert(generated);
      auto factor = 1.0/val_norm; 
      auto added = exatn::addTensors(add_pattern, factor); assert(added);
      add_pattern.clear();
      generated = exatn::generate_addition_pattern(rank,add_pattern,true,grad->getName(), ac_gradient_aux_tensor->getName()); assert(generated);
      factor = state_energies_[0]/(val_norm * val_norm);
      added = exatn::addTensors(add_pattern, -factor); assert(added);
      add_pattern.clear();
      
      initialized = exatn::initTensorSync(ac_gradient_tensor->getName(),0.0); assert(initialized);

      generated = exatn::generate_addition_pattern(rank,add_pattern,true,ac_gradient_tensor->getName(), grad->getName()); assert(generated);
      factor = 1.0;
      added = exatn::addTensors(add_pattern, factor); assert(added);
      add_pattern.clear();
      auto success = exatn::computeMaxAbsSync(grad->getName(), maxAbsElement); assert(success);
      std::cout << "Max. absolute value in gradient tensor: " << maxAbsElement << std::endl;
      auto destroyed = exatn::destroyTensorSync(grad->getName()); assert(destroyed);
      if (maxAbsElement < convergence_thresh_) continue;
      */
      /*
      auto prior_tensor = std::make_shared<exatn::Tensor>("PriorTensor_"+tensor_name,tensor_shape,tensor_signature);
      auto created = exatn::createTensorSync(prior_tensor,tensor->getElementType()); assert(created);
      //std::string add_pattern;
      auto initialized = exatn::initTensor(prior_tensor->getName(), 0.0); assert(initialized);
      std::string add_pattern;
      auto generated = exatn::generate_addition_pattern(rank,add_pattern,true,prior_tensor->getName(), tensor_name); assert(generated);
      std::cout << add_pattern << std::endl;
      auto added = exatn::addTensors(add_pattern,1.0); assert(added);
      add_pattern.clear();
    
      initialized = exatn::initTensor(tensor_name, 0.0); assert(initialized);
      generated = exatn::generate_addition_pattern(rank,add_pattern,true,tensor_name, prior_tensor->getName()); assert(generated);
      std::cout << add_pattern << std::endl;
      added = exatn::addTensors(add_pattern,1.0); assert(added);
      add_pattern.clear();
    
      generated = exatn::generate_addition_pattern(rank,add_pattern,true,tensor->getName(), gradient_name); assert(generated);
      std::cout << add_pattern << std::endl;
      auto lr = 0.75;
      added = exatn::addTensors(add_pattern, -lr); assert(added);
      add_pattern.clear();

      //initialized = exatn::initTensor(ac_norm->getName(), 0.0); assert(initialized);
      //evaluated = exatn::evaluateSync(*norm_,ac_norm); assert(evaluated);
      //local_copy = exatn::getLocalTensor(ac_norm->getName()); assert(local_copy);
      //access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
      //val_norm = *body_ptr;
      //body_ptr = nullptr;
      // int num_optimizable = 4;
      //  std::cout << "Number of optimizable tensors: " << num_optimizable << std::endl;
      //  factor = pow(sqrt(1.0/val_norm), 1.0/double(num_optimizable));
      //  auto scaled = exatn::scaleTensor(tensor->getName(), factor); assert(scaled);

      std::cout << " Done updating tensor "  << tensor_name << std::endl;
      auto destroyed = exatn::destroyTensorSync(prior_tensor->getName()); assert(destroyed);
      //destroyed = exatn::destroyTensorSync(ac_norm->getName()); assert(destroyed);
   // }
  } 
  environments_.clear();
  */
}
 
} //namespace castn
