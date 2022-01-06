#include "exatn.hpp"
#include "talshxx.hpp"
#include <iomanip>
#include "../../src/particle_ansatz.hpp"

using namespace std::chrono;
using namespace castn;

int main(int argc, char** argv){

  const auto TENS_ELEM_TYPE = exatn::TensorElementType::REAL64;

  // active orbitals, active particles, core orbitals, total orbitals, total particles
  std::size_t nao = 16, nap = 8, nco = 0, nto = 16, ntp = 8;

  // output tensor dimension extents
  const exatn::DimExtent o1 = nto, o2 = nto, o3 = nto, o4 = nto;
  const exatn::DimExtent o5 = nto, o6 = nto, o7 = nto, o8 = nto;

  const auto max_bond_dim = std::min(int(std::pow(nto,(ntp/2))),4);

  exatn::initialize();

  {
    // declare output tensor
    auto output_tensor = std::make_shared<exatn::Tensor>("Z0", exatn::TensorShape{o1,o2,o3,o4,o5,o6,o7,o8});

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

    // initializing tensors for one and two-electron integrals
    success = exatn::initTensorFile(h1->getName(),"oei.txt"); assert(success);
    success = exatn::initTensorFile(h2->getName(),"tei.txt"); assert(success);

    // create hamiltonian operator from one and two-electron integrals
    std::vector<std::shared_ptr<exatn::Tensor> > hamiltonian;
    hamiltonian.push_back(h2);
    hamiltonian.push_back(h1);

    std::shared_ptr<exatn::TensorExpansion> ansatz;
    ansatz = std::make_shared<exatn::TensorExpansion>();
    ansatz->appendComponent(ansatz_net,{1.0,0.0});

    double convergence_thresh = castn::ParticleNumberRepresentation::DEFAULT_CONVERGENCE_THRESH;

    // declare object from Simulation class
    ParticleAnsatz optimizer(nao, nap, nco, nto, ntp);
    optimizer.resetWaveFunctionAnsatz(ansatz);
    optimizer.resetHamiltonian(hamiltonian);
    optimizer.optimize(1,convergence_thresh);

  }

  exatn::finalize();

  return 0;

}
