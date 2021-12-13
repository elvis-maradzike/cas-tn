#ifdef MPI_ENABLED
#include "mpi.h"
#endif
#include "exatn.hpp"
#include "quantum.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../../src/simulation.hpp"
using namespace std::chrono;
using namespace castn;

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
  const int bond_dim_lim = 4;
  const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_spin_sites/2)),bond_dim_lim);
  const int arity = 2;
  const std::string tn_type = "MPS";
  const unsigned int num_states = 3;
  const double accuracy = 1e-4;

  bool success = true;
  bool root = (exatn::getProcessRank() == 0);

  exatn::initialize();

  {

    //Read the MCVQE Hamiltonian in spin representation:
    auto hamiltonian0 = exatn::quantum::readSpinHamiltonian("MyHamiltonian","h_spin_representation.txt",TENS_ELEM_TYPE, "OpenFermion");

    hamiltonian0->printIt();
    
    //Configure the tensor network builder:
    auto tn_builder = exatn::getTensorNetworkBuilder(tn_type);
    if(tn_type == "MPS"){
     success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
    }else if(tn_type == "TTN"){
     success = tn_builder->setParameter("max_bond_dim",max_bond_dim); assert(success);
     success = tn_builder->setParameter("arity",arity); assert(success);
    }else{
     assert(false);
    }

    //Build tensor network vectors:
    auto ket_tensor = exatn::makeSharedTensor("TensorSpace",std::vector<int>(num_spin_sites,2));
    auto vec_net0 = exatn::makeSharedTensorNetwork("VectorNet1",ket_tensor,*tn_builder,false);
    vec_net0->markOptimizableAllTensors();
    auto vec_tns0 = exatn::makeSharedTensorExpansion("VectorTNS1",vec_net0,std::complex<double>{1.0,0.0});
    auto vec_net1 = exatn::makeSharedTensorNetwork("VectorNet2",ket_tensor,*tn_builder,false);
    vec_net1->markOptimizableAllTensors();
    auto vec_tns1 = exatn::makeSharedTensorExpansion("VectorTNS2",vec_net1,std::complex<double>{1.0,0.0});
    auto vec_net2 = exatn::makeSharedTensorNetwork("VectorNet3",ket_tensor,*tn_builder,false);
    vec_net2->markOptimizableAllTensors();
    auto vec_tns2 = exatn::makeSharedTensorExpansion("VectorTNS3",vec_net2,std::complex<double>{1.0,0.0});

    // Create and initialize tensor network vector tensors:
    if(root) std::cout << "Creating and initializing tensor network vector tensors ... ";
    success = exatn::createTensorsSync(*vec_net0,TENS_ELEM_TYPE); assert(success);
    success = exatn::initTensorsRndSync(*vec_net0); assert(success);
    success = exatn::createTensorsSync(*vec_net1,TENS_ELEM_TYPE); assert(success);
    success = exatn::initTensorsRndSync(*vec_net1); assert(success);
    success = exatn::createTensorsSync(*vec_net2,TENS_ELEM_TYPE); assert(success);
    success = exatn::initTensorsRndSync(*vec_net2); assert(success);
    if(root) std::cout << "Ok" << std::endl;

    //Ground and three excited states in one call:
    if(root) std::cout << "Ground and excited states search for the original Hamiltonian:" << std::endl;
    exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
    vec_net0->markOptimizableAllTensors();
    success = exatn::initTensorsRndSync(*vec_tns0); assert(success);
    exatn::TensorNetworkOptimizer optimizer3(hamiltonian0,vec_tns0,accuracy);
    optimizer3.enableParallelization(true);
    success = exatn::sync(); assert(success);
    bool converged = optimizer3.optimize(num_states);
    success = exatn::sync(); assert(success);
    if(converged){
      if(root){
        std::cout << "Search succeeded:" << std::endl;
        for(unsigned int root_id = 0; root_id < num_states; ++root_id){
          std::cout << "Expectation value " << root_id << " = "
                 << optimizer3.getExpectationValue(root_id) << std::endl;
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
