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

  exatn::initialize();
  {

    // define tensor expanstion explicitly
    const unsigned int num_states = 1;
    const double accuracy = 1e-4;
    const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
    auto c = exatn::makeSharedTensor("C", exatn::TensorShape{2,2,2,2,2,2,2,2});
    auto success = exatn::createTensor(c, TENS_ELEM_TYPE);
    auto net = exatn::makeSharedTensorNetwork("Net");
    net->appendTensor(1,c,{});
    auto ansatz = exatn::makeSharedTensorExpansion();
    ansatz->appendComponent(net,{1.0,0.0});
    ansatz->rename("Ansatz");
    ansatz->printIt();
    ansatz->markOptimizableAllTensors();
    
    bool root = (exatn::getProcessRank() == 0);
    
    //Read Hamiltonian in spin representation:
    auto hamiltonian = exatn::quantum::readSpinHamiltonian("MyHamiltonian","h_spin_representation.txt",TENS_ELEM_TYPE, "OpenFermion");
    hamiltonian->printIt();
    
    // Create and initialize tensor network vector tensors:
    if(root) std::cout << "Creating and initializing tensor network vector tensors ... ";
    success = exatn::createTensorsSync(*net,TENS_ELEM_TYPE); assert(success);
    success = exatn::initTensorsRndSync(*net); assert(success);
    if(root) std::cout << "Ok" << std::endl;

    //Ground state:
    if(root) std::cout << "Ground and excited states search for the original Hamiltonian:" << std::endl;
    exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
    net->markOptimizableAllTensors();
    success = exatn::initTensorsRndSync(*ansatz); assert(success);
    exatn::TensorNetworkOptimizer optimizer(hamiltonian,ansatz,accuracy);
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
      success = exatn::printTensorFileSync("C","ansatz_optimized.txt"); assert(success);
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

