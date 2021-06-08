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

  auto created = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(created);
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
  double energy_pre  = 0.;
  double energy_post  = 0.;
  macroIter = 0;
  do {
    double val_norm = 0.0;
    energy_pre = energy_post;
    evaluateEnergyFunctional();
    energy_post = state_energies_[0];
    evaluateEnergyDerivatives();
    // update each optimizable tensor in tensor network expansion
    updateWavefunctionAnsatzTensors(); 
    std::cout << std::setw(7) << std::right <<  " macroIter: " <<
          std::setw(7) << std::right << macroIter <<
          std::setw(12) << std::fixed << state_energies_[0] <<
          std::setw(12) << std::fixed << energy_pre-energy_post <<
          std::setw(12) << std::fixed << std::endl;

   macroIter++; 
    
  } while (fabs(energy_pre-energy_post) > 1e-10);

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
  
  exatn::TensorExpansion xSx(*bra_ansatz_,*ket_ansatz_);
  norm_square_ = std::make_shared<exatn::TensorExpansion>(xSx);
  norm_square_->printIt();
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
          auto g_xHx_tensor = std::make_shared<exatn::Tensor>("_g_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto g_xx_tensor = std::make_shared<exatn::Tensor>("_g_aux_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto h_xHx_tensor = std::make_shared<exatn::Tensor>("_h_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto h_xx_tensor = std::make_shared<exatn::Tensor>("_h_aux_"+tensor.getName(),tensor.getShape(), tensor.getSignature());
          auto created = exatn::createTensor("_g_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
		  created = exatn::createTensor("_g_aux_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          created = exatn::createTensor("_h_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          created = exatn::createTensor("_h_aux_"+tensor.getName(),TENS_ELEM_TYPE,tensor.getShape()); assert(created);
          exatn::TensorExpansion g(*functional_,tensor.getName(),true);
          exatn::TensorExpansion g_aux(*norm_square_,tensor.getName(),true);
          exatn::TensorExpansion h(*functional_,tensor.getTensor(),g_xHx_tensor);
          exatn::TensorExpansion h_aux(*norm_square_,tensor.getTensor(),g_xx_tensor);
          auto gradient_expansion = std::make_shared<exatn::TensorExpansion>(g);
          auto gradient_aux_expansion = std::make_shared<exatn::TensorExpansion>(g_aux);
          auto hessian_expansion = std::make_shared<exatn::TensorExpansion>(h);
          auto hessian_aux_expansion = std::make_shared<exatn::TensorExpansion>(h_aux);
          derivatives_.push_back(std::make_tuple(tensor.getName(), gradient_expansion, gradient_aux_expansion, hessian_expansion, hessian_aux_expansion, g_xHx_tensor, g_xx_tensor, h_xHx_tensor, h_xx_tensor));
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
        auto initialized = exatn::initTensorRnd(tensor.getName()); assert(initialized);
        //auto initialized = exatn::initTensorFile(tensor.getName(),"h4_tensor.txt"); assert(initialized);
      } 
    }
  }
  std::cout << " Done initializing all optimizable tensors " << std::endl;
 
  // evaluate norm of wavefunction
  auto created = exatn::createTensorSync("_norm_square_", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
  auto norm_square_tensor = exatn::getTensor("_norm_square_");
  auto initialized = exatn::initTensor(norm_square_tensor->getName(), 0.0); assert(initialized);
  auto evaluated = exatn::evaluateSync(*norm_square_,norm_square_tensor); assert(evaluated);
  
  auto local_copy = exatn::getLocalTensor(norm_square_tensor->getName()); assert(local_copy);
  const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
  auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_norm_square = *body_ptr;
  body_ptr = nullptr;
  std::cout << "Norm before renormalization is: " << sqrt(val_norm_square) << std::endl; 
  //auto success = exatn::printTensorSync(norm_square_tensor->getName()); assert(success);

  // scale optimizable tensors so that overall norm is unity
  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        // number of tensors in network less the number of ordering projectors
        int num_optimizable = network->network->getNumTensors() - (num_particles_-1);
        double factor = pow(sqrt(1.0/val_norm_square), 1.0/double(num_optimizable));
        auto scaled = exatn::scaleTensor(tensor.getName(), factor); assert(scaled);
      } 
    }
  }

  // reevaluate norm functional
  created = exatn::createTensorSync("_norm_square_post", TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
  auto norm_square_tensor_post = exatn::getTensor("_norm_square_post");
  initialized = exatn::initTensor(norm_square_tensor_post->getName(), 0.0); assert(initialized);
  evaluated = exatn::evaluateSync(*norm_square_, norm_square_tensor_post); assert(evaluated);
  //success = exatn::printTensorSync(norm_square_tensor_post->getName()); assert(success);

  // destroy tensors
  auto destroyed = exatn::destroyTensorSync(norm_square_tensor->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(norm_square_tensor_post->getName()); assert(destroyed);
  std::cout << "Wavefunction ansatz has been initialized and normalized ....." << std::endl;
}

double Simulation::evaluateEnergyFunctional(){
  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;
  const auto TENSOR_SHAPE = exatn::TensorShape{};
  auto created = exatn::createTensorSync("_norm_square",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
  auto norm_square_tensor = exatn::getTensor("_norm_square");
  auto initialized = exatn::initTensor(norm_square_tensor->getName(), 0.0); assert(initialized);
  // evaluate norm square
  auto evaluated = exatn::evaluateSync(*norm_square_,norm_square_tensor); assert(evaluated);
  auto local_copy = exatn::getLocalTensor(norm_square_tensor->getName()); assert(local_copy);
  const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
  auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_norm_square = *body_ptr;
  body_ptr = nullptr;
  
  // create accumulator tensor for the closed tensor expansion
  created = exatn::createTensorSync("_functional_tensor",TENS_ELEM_TYPE, TENSOR_SHAPE); assert(created);
  auto functional_tensor = exatn::getTensor("_functional_tensor");
  initialized = exatn::initTensor(functional_tensor->getName(), 0.0); assert(initialized);
  evaluated = exatn::evaluateSync(*functional_,functional_tensor); assert(evaluated);
  // get value 
  local_copy = exatn::getLocalTensor(functional_tensor->getName()); assert(local_copy);
  access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
  double val_functional = *body_ptr;
  
  double energy = 0.0;
  energy = val_functional/val_norm_square;
  std::cout << " (E, Norm): " << energy << ", " << val_norm_square << std::endl;

  for (auto network = ket_ansatz_->cbegin(); network != ket_ansatz_->cend(); ++network){
    for ( auto tensor_conn = network->network->begin(); tensor_conn != network->network->end(); ++tensor_conn){
      const auto & tensor = tensor_conn->second;
      if (tensor.isOptimizable()){
        int num_optimizable = network->network->getNumTensors() - (num_particles_-1);
        double factor = pow(sqrt(1.0/val_norm_square), 1.0/double(num_optimizable));
        auto scaled = exatn::scaleTensor(tensor.getName(), factor); assert(scaled);
      } 
    }
  }
  // store value
  state_energies_.clear();
  state_energies_.push_back(energy);

  // destroy accumulator tensor
  auto destroyed = exatn::destroyTensorSync(functional_tensor->getName()); assert(destroyed);
  destroyed = exatn::destroyTensorSync(norm_square_tensor->getName()); assert(destroyed);
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
	//auto ac_gradient_tensor = std::make_shared<exatn::Tensor>("GradientTensor_"+tensor_name, tensor_shape, tensor_signature);
    auto gradient_tensor = exatn::getTensor("_g_" + tensor_name);
    auto gradient_aux_tensor = exatn::getTensor("_g_aux_" + tensor_name);
    auto hessian_tensor = exatn::getTensor("_h_" + tensor_name);
    auto hessian_aux_tensor = exatn::getTensor("_h_aux_" + tensor_name);
    auto gradient_expansion = std::get<1>(derivatives_[i]);
    auto gradient_aux_expansion = std::get<2>(derivatives_[i]);
    auto hessian_expansion = std::get<3>(derivatives_[i]);
    auto hessian_aux_expansion = std::get<4>(derivatives_[i]);
    auto initialized = exatn::initTensorSync(gradient_tensor->getName(),0.0); assert(initialized);
    initialized = exatn::initTensorSync(gradient_aux_tensor->getName(),0.0); assert(initialized);
    initialized = exatn::initTensorSync(hessian_tensor->getName(),0.0); assert(initialized);
    initialized = exatn::initTensorSync(hessian_aux_tensor->getName(),0.0); assert(initialized);
    auto evaluated = exatn::evaluateSync(*gradient_expansion, gradient_tensor); assert(evaluated);
    evaluated = exatn::evaluateSync(*gradient_aux_expansion, gradient_aux_tensor); assert(evaluated);
    //evaluated = exatn::evaluateSync(*hessian_expansion, hessian_tensor); assert(evaluated);
    //evaluated = exatn::evaluateSync(*hessian_aux_expansion, hessian_aux_tensor); assert(evaluated);
    // compute norm square and energy
    auto created = exatn::createTensorSync("_norm_square",TENS_ELEM_TYPE, exatn::TensorShape{}); assert(created);
    auto norm_square_tensor = exatn::getTensor("_norm_square");
    initialized = exatn::initTensor(norm_square_tensor->getName(), 0.0); assert(initialized);
    evaluated = exatn::evaluateSync(*norm_square_,norm_square_tensor); assert(evaluated);
    auto local_copy = exatn::getLocalTensor(norm_square_tensor->getName()); assert(local_copy);
    const  exatn::TensorDataType<TENS_ELEM_TYPE>::value * body_ptr;
    auto access_granted = local_copy->getDataAccessHostConst(&body_ptr); assert(access_granted);
    double val_norm_square = *body_ptr;
    body_ptr = nullptr;

    auto grad = std::make_shared<exatn::Tensor>("Grad_"+tensor_name, tensor_shape, tensor_signature);
    created = exatn::createTensorSync(grad,tensor->getElementType()); assert(created);
    initialized = exatn::initTensorSync(grad->getName(),0.0); assert(initialized);
    std::string add_pattern;
    auto generated = exatn::generate_addition_pattern(rank,add_pattern,true,grad->getName(), gradient_tensor->getName()); assert(generated);
    auto factor = 1.0; 
    auto added = exatn::addTensors(add_pattern, factor); assert(added);
    add_pattern.clear();
    generated = exatn::generate_addition_pattern(rank,add_pattern,true,grad->getName(), gradient_aux_tensor->getName()); assert(generated);
    factor = -1.0 * state_energies_[0];
    added = exatn::addTensors(add_pattern, factor); assert(added);
    add_pattern.clear();
    
    initialized = exatn::initTensorSync(gradient_tensor->getName(),0.0); assert(initialized);
    generated = exatn::generate_addition_pattern(rank,add_pattern,true,gradient_tensor->getName(), grad->getName()); assert(generated);
    factor = 1.0;
    added = exatn::addTensors(add_pattern, factor); assert(added);
    add_pattern.clear();
    
    //auto success = exatn::printTensorSync(ac_gradient_tensor->getName()); assert(success);
    std::cout << " Optimizable tensor name is " << tensor_name  << std::endl;
    environments_.emplace_back(Environment{exatn::getTensor(tensor_name),
                 gradient_tensor,
                 *gradient_expansion,
                 *gradient_aux_expansion
                 });
   
    std::cout << "Done evaluating derivative ...." << std::endl;   
    auto destroyed = exatn::destroyTensorSync(grad->getName()); assert(destroyed);
    destroyed = exatn::destroyTensorSync(norm_square_tensor->getName()); assert(destroyed);
  }
}
  
void Simulation::updateWavefunctionAnsatzTensors(){
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
    std::cout << "Max. absolute value in gradient tensor " << gradient_name << " w.r.t. " << tensor_name << " is " << maxAbsElement << std::endl;
    //success = exatn::printTensorSync(gradient_name); assert(success);
    if (maxAbsElement < convergence_thresh_) continue;
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
    std::cout << " Done updating tensor "  << tensor_name << std::endl;
    auto destroyed = exatn::destroyTensorSync(prior_tensor->getName()); assert(destroyed);
  }
  environments_.clear();
}
 
} //namespace castn
