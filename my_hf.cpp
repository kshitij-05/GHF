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


std::vector<Atom> read_geometry(string filename);
std::vector<string> read_input(string inp_file);
double enuc_calc(vector<Atom> atoms);
size_t nbasis(const std::vector<libint2::Shell>& shells);
Matrix compute_1body_ints(const std::vector<libint2::Shell>& shells,libint2::Operator obtype,const std::vector<Atom>& atoms);
Matrix make_density(Matrix C ,const int nocc);
double scf_energy(Matrix D,Matrix H, Matrix F);
Matrix make_fock(const std::vector<libint2::Shell>& shells,const Matrix& D);

int main(int argc, char* argv[]) {
    string xyzfile = argv[1];
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    const auto inpfile = argv[2];
    const std::vector<libint2::Atom> atoms= read_geometry(xyzfile);
    // Print Geometry
    cout<< "molecular geometry :" << "\n";
    for (auto n : atoms){
        cout<<n.atomic_number << std::setprecision(12) << " \t " << n.x << "\t " << n.y<< "\t " << n.z<< "\n";
    }
    std::vector <string> inplines = read_input(inpfile);
    string method, basis_name;
    method = inplines[0];
    basis_name = inplines[1];

    // np. of electrons
    int nelec=0;
    for (int i=0; i< atoms.size();i++){nelec+= atoms[i].atomic_number;}
    const auto nocc = nelec / 2;
    cout << "No. of electrons = " << nelec << endl;

    // nuclear repulsion energy
    auto enuc = enuc_calc(atoms);
    cout << "Nuclear Repulsion energy = \t" <<enuc << "  a.u."<< std:: setprecision(15)<< "\n";

    //  Construct basisset
    BasisSet obs(basis_name, atoms);
    auto Nbasis = nbasis(obs.shells());
    cout <<"Number of basis functions: " << Nbasis << std::setprecision(8) << "\n";

    // 1 body intergrals
    libint2::initialize();
    auto S = compute_1body_ints(obs.shells(), Operator::overlap , atoms);
    auto T = compute_1body_ints(obs.shells(), Operator::kinetic , atoms);
    auto V = compute_1body_ints(obs.shells(), Operator::nuclear , atoms);
    Matrix H = T+V;

    // initial guess fock = S^1/2.T * H * S^1/2
    Eigen::SelfAdjointEigenSolver<Matrix> ovlp_inv_sqrt(S);
    auto S_inv = ovlp_inv_sqrt.operatorInverseSqrt();
    Matrix init_guess_fock;
    init_guess_fock.noalias() = S_inv.transpose()*H*S_inv;
    //cout << init_guess_fock << std::setprecision(8)<< endl;

    //Build Initial Guess Density
    Eigen::SelfAdjointEigenSolver<Matrix> init_fock_diag(init_guess_fock);
    Matrix C = S_inv*init_fock_diag.eigenvectors();
    Matrix D = make_density(C ,nocc);
    //cout << D << std::setprecision(8)<< endl;
    double init_energy = scf_energy(D,H,init_guess_fock);

    //SCF LOOP
    double hf_energy = 0.0;
    bool do_diis = 1;
    int set_lenght = 6;
    vector<Matrix> error_set;
    vector<Matrix> fock_set;
    for (int i=0;i<100; i++){
        Matrix Fock =  H + make_fock(obs.shells(),D);

        //DIIS
        /*if(do_diis){
            int Nerr = error_set.size();
            if(i>0){
                Matrix  error = Fock*D*S - S*D*Fock;
                if(Nerr < set_lenght){
                    error_set.push_back(error);
                    fock_set.push_back(Fock);
                }
                else if(Nerr>=set_lenght){
                    error_set.erase(error_set.begin());
                    fock_set.erase(fock_set.begin());
                    error_set.push_back(error);
                    fock_set.push_back(Fock);
                }
            }
            if(Nerr>=2){
                Matrix Bmat = Eigen::MatrixXd::Zero(Nerr+1,Nerr+1);
                Matrix zerovec = Eigen::MatrixXd::Zero(0,Nerr+1);
                zerovec(0,Nerr)=-1.0;
                for(auto a=0; a<Nerr;a++){
                    for(auto b=0; b<a+1; b++){
                        Bmat(a,b) = Bmat(b,a) = (error_set[a]*error_set[b]).trace();
                        Bmat(Nerr,a) = Bmat(a,Nerr) = -1.0;
                    }
                }
                if(Bmat.determinant() !=0.0){
                    Matrix coeff = Bmat.fullPivHouseholderQr().solve(zerovec);
                    Fock -=Fock;
                    for(auto a =0;a<coeff.cols()-1;a++){
                        Fock +=coeff(a)*fock_set[a];
                    }
                }
                else if(Bmat.determinant()==0.0){
                    do_diis = 0;
                    cout << "Bmat is singular" << endl;
                }
            }
        }*/

        Eigen::SelfAdjointEigenSolver<Matrix> fock_diag(S_inv.transpose()*Fock*S_inv);
        C = S_inv*fock_diag.eigenvectors();
        D = make_density(C,nocc);
        double new_energy = scf_energy(D,H,Fock);
        double delta_e = abs(new_energy - hf_energy);
        hf_energy = new_energy;
        cout << std::setprecision(12)<< "iter no :\t" << i+1 << "\t" << "Energy :\t" << hf_energy + enuc << "\t" << "Delta_E :\t" << delta_e << endl;
        if (delta_e <=1e-12){
            cout << "Converged SCF energy :" << hf_energy +enuc<< "  a.u." << endl;
            cout << fock_diag.eigenvalues() << endl;
            break;
        }
    }
}


