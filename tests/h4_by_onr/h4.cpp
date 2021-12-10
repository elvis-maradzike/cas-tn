#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../../src/simulation.hpp"
#include "quantum.hpp"
using namespace std::chrono;
using namespace castn;


int main(int argc, char** argv){

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::COMPLEX64;
  const int num_spin_sites = 8;
  const int bond_dim_lim = 4;
  const int max_bond_dim = std::min(static_cast<int>(std::pow(2,num_spin_sites/2)),bond_dim_lim);
  const int arity = 2;
  const std::string tn_type = "TTN";
  const unsigned int num_states = 3;
  const double accuracy = 1e-4;

  bool success = true;
  bool root = (exatn::getProcessRank() == 0);

  exatn::initialize();

  {

    //Read the MCVQE Hamiltonian in spin representation:
    auto hamiltonian0 = exatn::quantum::readSpinHamiltonian("MCVQEHamiltonian","h_transformed.txt",TENS_ELEM_TYPE,"QCWare");

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

  return 0;

}
