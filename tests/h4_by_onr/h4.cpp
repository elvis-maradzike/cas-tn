#ifdef MPI_ENABLED
#include "mpi.h"
#endif
#include "exatn.hpp"
#include "quantum.hpp"
#include "talshxx.hpp"
#include <iomanip>
using namespace std::chrono;

int main(int argc, char** argv){

  exatn::ParamConf exatn_parameters;
  exatn_parameters.setParameter("host_memory_buffer_size",48L*1024L*1024L*1024L);

#ifdef MPI_ENABLED
  int thread_provided;
  int mpi_error = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_provided);
  assert(mpi_error == MPI_SUCCESS);
  assert(thread_provided == MPI_THREAD_MULTIPLE);
  exatn::initialize(exatn::MPICommProxy(MPI_COMM_WORLD), exatn_parameters, "lazy-dag-executor");
#else
  exatn::initialize(exatn_parameters, "lazy-dag-executor");
#endif

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
  const int num_spin_sites = 8;
  const int bond_dim_lim = 16;
  const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_spin_sites/2)),bond_dim_lim);
  const int arity = 2;
  const std::string tn_type = "MPS";
  const unsigned int num_states = 1;
  const double accuracy = 1e-4;
  bool success = true;
  bool root = (exatn::getProcessRank() == 0);

  exatn::initialize();
  {
  // define input tensor to make target:
  auto target_tensor = exatn::makeSharedTensor("TargetTensor",std::vector<int>(num_spin_sites,2));
  auto success = exatn::createTensor(target_tensor, TENS_ELEM_TYPE);
  auto target_net = exatn::makeSharedTensorNetwork("TargetNet");
  target_net->appendTensor(1,target_tensor,{});
  target_net->markOptimizableAllTensors();
  auto target = exatn::makeSharedTensorExpansion();
  target->appendComponent(target_net,{1.0,0.0});
  target->rename("TargetExpansion");
  //target->printIt();
  target->markOptimizableAllTensors();

  // configure the tensor network builder:
  auto tn_builder = exatn::getTensorNetworkBuilder(tn_type);
  if(tn_type == "MPS"){
   success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
  }else if(tn_type == "TTN"){
   success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
   success = tn_builder->setParameter("arity",arity); assert(success);
  }else{
   assert(false);
  }
  
  // output tensor
  auto approximant_tensor = exatn::makeSharedTensor("TensorSpace",std::vector<int>(num_spin_sites,2));
  success = exatn::createTensor(approximant_tensor, TENS_ELEM_TYPE);
  //approximant_tensor->printIt();

  // tensor network 
  auto approximant_net = exatn::makeSharedTensorNetwork("ApproximantNetwork",approximant_tensor,*tn_builder,false);
  approximant_net->markOptimizableAllTensors();

  // tensor network expansion
  auto approximant = exatn::makeSharedTensorExpansion();
  approximant->appendComponent(approximant_net,{1.0,0.0});
  approximant->rename("Approximant");
  
  // read in Hamiltonian
  auto hamiltonian = exatn::quantum::readSpinHamiltonian("MyHamiltonian","h_spin_representation.txt",TENS_ELEM_TYPE, "OpenFermion");

  // create and initialize tensor network 
  if(root) std::cout << "Creating and initializing tensor network vector tensors ... " << std::endl;
  success = exatn::createTensorsSync(*approximant_net,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*approximant_net); assert(success);
  success = exatn::initTensorFile("TargetTensor", "optimized_tensor.txt"); assert(success);
  if(root) std::cout << "Ok" << std::endl;

  // run reconstruction
  approximant->conjugate();
  exatn::TensorNetworkReconstructor reconstructor(target,approximant,1e-7);
  reconstructor.resetDebugLevel(2,0);
  success = exatn::sync(); assert(success);
  double residual_norm, fidelity;
  bool reconstructed = reconstructor.reconstruct(&residual_norm,&fidelity,true);
  success = exatn::sync(); assert(success);
    if(reconstructed){
      std::cout << "Reconstruction succeeded: Residual norm = " << residual_norm
              << "; Fidelity = " << fidelity << std::endl;
    }else{
      std::cout << "Reconstruction failed!" << std::endl; assert(false);
    }


    approximant->conjugate();
   
    // ground state:
    if(root) std::cout << "Ground state search for the original Hamiltonian:" << std::endl;
    exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
    exatn::TensorNetworkOptimizer optimizer(hamiltonian,approximant,accuracy);
    optimizer.enableParallelization(true);
    success = exatn::sync(); assert(success);
    bool converged = optimizer.optimize(num_states);
    success = exatn::sync(); assert(success);
    if(converged){
      if(root){
        std::cout << "Search succeeded:" << std::endl;
        for(unsigned int root_id = 0; root_id < num_states; ++root_id){
          std::cout << "Expectation value " << root_id << " = "
                 << optimizer.getExpectationValue(root_id) << std::endl;
        }
      }
    }else{
      if(root) std::cout << "Search failed!" << std::endl;
      assert(false);
    }
  }

  exatn::finalize();

#ifdef MPI_ENABLED
  mpi_error = MPI_Finalize();
  assert(mpi_error == MPI_SUCCESS);
#endif
  
  return 0;

}
