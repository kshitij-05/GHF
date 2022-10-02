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

struct inp_params{
    string scf = "RHF";
    string method = "HF";
    string basis= "STO-3G";
    double scf_convergence = 1e-10;
    int do_diis = 1;
    int spin_mult = 1;
    int charge = 0;
};

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> DiagonalMatrix;

std::vector<Atom> read_geometry(string filename);
inp_params read_input(string inp_file);
double enuc_calc(vector<Atom> atoms);
size_t nbasis(const std::vector<libint2::Shell>& shells);
vector<Matrix> RHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc);
void UHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc);
Tensor<double> make_ao_ints(const std::vector<libint2::Shell>& shells);
double mp2_energy(Tensor <double> eri, Matrix coeffs, Matrix F , int no, int nv);
double mp2_rhf(Tensor<double> eri, Matrix coeffs, Matrix F, int no, int nv);
Tensor<double> make_ao_ints_simple(const std::vector<libint2::Shell>& shells);


int main(int argc, char* argv[]) {
    // INPUT ARGUMENTS
    string xyzfile = argv[1];
    const auto inpfile = argv[2];
    const std::vector<libint2::Atom> atoms= read_geometry(xyzfile);
    // Print Geometry
    cout<< "molecular geometry :" << "\n";
    for (auto n : atoms){
        cout<<n.atomic_number << std::setprecision(12) << " \t "
        << n.x << "\t " << n.y<< "\t " << n.z<< "\n";
    }
    // INPUT PARAMETERS
    inp_params inpParams = read_input(inpfile);
    cout << "method: " << inpParams.method << endl;
    cout << "basis: " <<inpParams.basis << endl;
    cout << "scf_convergence: " <<inpParams.scf_convergence << endl;
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

    //RHF
    vector<Matrix> scf_results;
    if(inpParams.scf == "RHF"){
        scf_results = RHF(obs,atoms,Nbasis,nelec,inpParams ,enuc);
    }

    //UHF
    if(inpParams.scf == "UHF"){
        UHF(obs,atoms,Nbasis,nelec,inpParams ,enuc);
    }

    if(inpParams.method == "MP2") {
        auto eri = make_ao_ints_simple(obs.shells());
        int no = nelec/2;
        int nv = Nbasis-no;
        cout << "number of occupied orbitals: " << no << endl;
        cout << "number of virtual  orbitals: "<< Nbasis-no << endl;
        double emp2 = mp2_rhf(eri,scf_results[1],scf_results[0],no, nv);
        cout<< std::setprecision(15) << "MP2 energy: "<< emp2<<endl;
    }
}


