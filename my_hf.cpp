//
// Created by Kshitij Surjuse on 9/13/22.
//


#include <iostream>
#include <string>
#include <vector>
#include <fstream>

//Eigen
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <libint2.hpp>
#if !LIBINT2_CONSTEXPR_STATICS
#  include <libint2/statics_definition.h>
#endif

#include <btas/btas.h>
#include <btas/tensor.h>
using namespace btas;

using std::cout;
using std::string;
using std::endl;
using libint2::Shell;
using libint2::Atom;
using libint2::BasisSet;
using libint2::Engine;
using libint2::Operator;
using std::vector;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> DiagonalMatrix;

struct inp_params{
    string scf = "RHF";
    string method = "HF";
    string basis= "STO-3G";
    double scf_convergence = 1e-10;
    int do_diis = 1;
    int spin_mult = 1;
    int charge = 0;
    string unit = "A";
};

struct scf_results{
    bool is_rhf=1;
    int nelec, nbasis,no,nv , nalpha,nbeta;
    Matrix C,F,Calpha,Cbeta,Falpha,Fbeta;
    double scf_energy;
};

vector <Atom>  read_geometry(string filename , string unit);
inp_params read_input(string inp_file);
double enuc_calc(vector<Atom> atoms);
size_t nbasis(const std::vector<libint2::Shell>& shells);
scf_results RHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc);
scf_results UHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc);
Tensor<double> make_ao_ints(const std::vector<libint2::Shell>& shells);
double mp2_energy(Tensor <double> eri,scf_results& SCF);



int main(int argc, char* argv[]) {
    // INPUT ARGUMENTS
    string xyzfile = argv[1];
    const auto inpfile = argv[2];
    // INPUT PARAMETERS
    inp_params inpParams = read_input(inpfile);
    cout << "method: " << inpParams.method << endl;
    cout << "basis: " <<inpParams.basis << endl;
    cout << "scf_convergence: " <<inpParams.scf_convergence << endl;

    const std::vector<libint2::Atom> atoms= read_geometry(xyzfile , inpParams.unit);
    // Print Geometry
    cout<< "molecular geometry :" << "\n";
    for (auto n : atoms){
        cout<<n.atomic_number << std::setprecision(12) << " \t "
            << n.x << "\t " << n.y<< "\t " << n.z<< "\n";
    }

    // no. of electrons
    int nelec=0;
    for(int i=0; i< atoms.size();i++){nelec+= atoms[i].atomic_number;}
    nelec -= inpParams.charge;
    cout << "No. of electrons = " << nelec << endl;

    // nuclear repulsion energy
    auto enuc = enuc_calc(atoms);
    cout << "Nuclear Repulsion energy = \t" <<enuc <<
    "  a.u."<< std:: setprecision(15)<< "\n";

    //  Construct basisset
    BasisSet obs(inpParams.basis, atoms);
    auto Nbasis = nbasis(obs.shells());
    cout <<"Number of basis functions: "
    << Nbasis << std::setprecision(8) << "\n";


    // SCF PROCEDURES

    scf_results SCF;

    //RHF
    if(inpParams.scf == "RHF"){
        SCF = RHF(obs,atoms,Nbasis,nelec,inpParams ,enuc);
    }

    //UHF
    if(inpParams.scf == "UHF"){
        SCF = UHF(obs,atoms,Nbasis,nelec,inpParams ,enuc);
    }

    if(inpParams.method == "MP2") {
        auto eri = make_ao_ints(obs.shells());
        double emp2 = mp2_energy(eri,SCF);
        cout<< std::setprecision(15) << "MP2 energy: "<< emp2<<endl;
    }
}


