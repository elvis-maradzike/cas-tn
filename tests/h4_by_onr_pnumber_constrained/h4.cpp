#ifdef MPI_ENABLED
#include "mpi.h"
#endif
#include "exatn.hpp"
#include "quantum.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../../src/particle_ansatz.hpp"

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
  
  exatn::resetLoggingLevel(2,2);
  
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
  // active orbitals, active particles, core orbitals, total orbitals, total particles
  std::size_t num_active_orbitals = 8, num_active_particles = 4, 
  num_core_orbitals = 0, num_total_orbitals = 8, num_total_particles = 4;

  exatn::initialize();
  {
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
  auto input_tensor = exatn::makeSharedTensor("TensorSpace",std::vector<int>(num_spin_sites,2));
  success = exatn::createTensor(input_tensor, TENS_ELEM_TYPE);
  //input_tensor->printIt();

  // tensor network 
  auto input_net = exatn::makeSharedTensorNetwork("InputNetwork",input_tensor,*tn_builder,false);
  input_net->markOptimizableAllTensors();

  // tensor network expansion
  auto ansatz = exatn::makeSharedTensorExpansion();
  ansatz->appendComponent(input_net,{1.0,0.0});
  ansatz->rename("KetAnsatz");
  
  // read in Hamiltonian
  auto hamiltonian = exatn::quantum::readSpinHamiltonian("MyHamiltonian","hamiltonian.txt",TENS_ELEM_TYPE, "OpenFermion");

  auto particle_number = exatn::quantum::readSpinHamiltonian("MyParticleNumber","particle_number.txt",TENS_ELEM_TYPE, "OpenFermion");

  // create and initialize tensor network 
  success = exatn::createTensorsSync(*input_net,TENS_ELEM_TYPE); assert(success);
  success = exatn::initTensorsRndSync(*input_net); assert(success);

  double convergence_thresh = castn::ParticleAnsatz::DEFAULT_CONVERGENCE_THRESH;
  //SpinSiteAnsatz optimizer(num_active_orbitals,num_active_particles,num_core_orbitals,num_total_orbitals,num_total_particles);

  double lambda = 0.005;
  double factor = double(num_total_particles) * double(num_total_particles);

  // define
  auto number_op_tensor = exatn::makeSharedTensor("NumberOpTensor",exatn::TensorShape{2,2});
  auto prod_number_op_tensors = exatn::makeSharedTensor("ProdNumberOpTensors",exatn::TensorShape{2,2,2,2});
  auto constant_n = exatn::makeSharedTensor("ConstantN",exatn::TensorShape{});

  // create 
  success = exatn::createTensor(number_op_tensor,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(prod_number_op_tensors,TENS_ELEM_TYPE); assert(success);
  success = exatn::createTensor(constant_n,TENS_ELEM_TYPE); assert(success);

  // initialize 
  success = exatn::initTensorData("NumberOpTensor",std::vector<std::complex<double>>{
                                           {0.0,0.0}, {0.0,0.0},
                                           {0.0,0.0}, {1.0,0.0}}); assert(success);
  success = exatn::initTensorData("ProdNumberOpTensors",std::vector<std::complex<double>>{
                                           {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0},
                                           {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0},
                                           {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {0.0,0.0},
                                           {0.0,0.0}, {0.0,0.0}, {0.0,0.0}, {1.0,0.0}}); assert(success);
  success = exatn::initTensor("ConstantN",std::complex<double>{factor,0.0}); assert(success);

  // operator 
  auto modifiedHamiltonian = exatn::makeSharedTensorOperator("ModifiedHamiltonian");

  factor = -2.0 * lambda * double(num_total_particles);
  for ( unsigned int i = 0; i < num_total_orbitals; i++){
    success = hamiltonian->appendComponent(number_op_tensor,{{i,0}},{{i,1}},{factor,0.0}); assert(success);
  }

  factor = lambda;
  for ( unsigned int i = 0; i < num_total_particles; i++){
    for ( unsigned int j = i+1; j < num_total_particles; j++){
      success = hamiltonian->appendComponent(prod_number_op_tensors,{{i,0},{j,1}},{{i,2},{j,3}},{factor,0.0}); assert(success);
    }
  }

  success = hamiltonian->appendComponent(constant_n,{},{},{factor,0.0}); assert(success);
  modifiedHamiltonian->printIt();


  SpinSiteAnsatz optimizer(num_total_orbitals, num_total_particles);
  optimizer.resetWaveFunctionAnsatz(ansatz);
  optimizer.resetHamiltonianOperator(hamiltonian);
  optimizer.resetConstraintOperator(modifiedHamiltonian);
  optimizer.optimize(1,convergence_thresh);

   /**
      // ground state:
    if(root) std::cout << "Ground state search for the original Hamiltonian:" << std::endl;
    exatn::TensorNetworkOptimizer::resetDebugLevel(1,0);
    exatn::TensorNetworkOptimizer optimizer(hamiltonian,ansatz,accuracy);
    optimizer.enableParallelization(true);
    success = exatn::sync(); assert(success);
    bool converged = optimizer.optimize(num_states);
    success = exatn::sync(); assert(success);
    if(converged){
        std::cout << "Search succeeded:" << std::endl;
          std::cout << "Expectation value " << 0 << " = "
                 << optimizer.getExpectationValue(0) << std::endl;
    }
    **/ 

  }
  
  exatn::finalize();

#ifdef MPI_ENABLED
  mpi_error = MPI_Finalize();
  assert(mpi_error == MPI_SUCCESS);
#endif
  
  return 0;

}
