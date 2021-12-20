#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../../src/particle_number_representation.hpp"

using namespace std::chrono;
using namespace castn;

int main(int argc, char** argv){

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  // active orbitals, active particles, core orbitals, total orbitals, total particles
  std::size_t nao = 8, nap = 4, nco = 0, nto = 8, ntp = 4;

  // dimension extents of output tensor
  const exatn::DimExtent np = nto, nq = nto, nr = nto, ns = nto;

  const auto max_bond_dim = 64;

  exatn::initialize();

  {
    // declare output tensor
    auto output_tensor = std::make_shared<exatn::Tensor>("Z0", exatn::TensorShape{np,nq,nr,ns});

    // builder for mps
    auto builder = exatn::getTensorNetworkBuilder("MPS");
    auto success = builder->setParameter("max_bond_dim", max_bond_dim); assert(success);
    auto ansatz_net = exatn::makeSharedTensorNetwork("AnsatzNetwork", output_tensor, *builder);
    ansatz_net->printIt();

    //Allocate/initialize tensors in the tensor network ansatz:
    for(auto tens_conn = ansatz_net->begin(); tens_conn != ansatz_net->end(); ++tens_conn){
      if(tens_conn->first != 0){
        success = exatn::createTensor(tens_conn->second.getTensor(),TENS_ELEM_TYPE); assert(success);
        success = exatn::initTensorRnd(tens_conn->second.getName()); assert(success);
      }
    }

    // declare tensors for one and two-electron integrals
    auto h1 = exatn::makeSharedTensor("H1", exatn::TensorShape{nto,nto});
    auto h2 = exatn::makeSharedTensor("H2", exatn::TensorShape{nto,nto,nto,nto});

    // create tensors for one and two-electron integrals
    success = exatn::createTensorSync(h1,TENS_ELEM_TYPE); assert(success);
    success = exatn::createTensorSync(h2,TENS_ELEM_TYPE); assert(success);

    // initialize tensors for one and two-electron integrals
    success = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(success);
    success = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(success);

    // create vector of tensors, containing one and two-electron integrals
    std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;
    hamiltonian.push_back(h2);
    hamiltonian.push_back(h1);

    // declare and create tensor expansion (wavefunction ansatz)
    std::shared_ptr<exatn::TensorExpansion> ansatz;
    ansatz = std::make_shared<exatn::TensorExpansion>();
    ansatz->appendComponent(ansatz_net,{1.0,0.0});

    double convergence_thresh = castn::ParticleNumberRepresentation::DEFAULT_CONVERGENCE_THRESH;

    // declare object from Simulation class
    ParticleNumberRepresentation optimizer(nao, nap, nco, nto, ntp);
    optimizer.resetWaveFunctionAnsatz(ansatz);
    optimizer.resetHamiltonian(hamiltonian);
    optimizer.optimize(1,convergence_thresh);

  }

  exatn::finalize();

  return 0;

}
