//
// Created by Kshitij Surjuse on 9/20/22.
//
//Eigen
#include <Eigen/Cholesky>
#include <Eigen/Dense>
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

Matrix compute_1body_ints(const std::vector<libint2::Shell>&
shells,libint2::Operator obtype,const std::vector<Atom>& atoms);
Matrix make_fock(const std::vector<libint2::Shell>& shells,const Matrix& D);
Matrix make_fock_uhf(const std::vector<libint2::Shell>& shells,
                     const Matrix& Dt, const Matrix& Dspin);
Matrix compute_2body_fock_simple(const std::vector<libint2::Shell>& shells,
                                 const Matrix& D);


Matrix make_density(Matrix C ,const int nocc){

    auto nrows = C.rows();
    auto ncols = C.cols();
    Matrix Density = Eigen::MatrixXd::Zero(nrows,ncols);
    for(size_t i =0;i<nrows;i++){
        for(size_t j = 0; j< ncols; j++ ){
            for(size_t  m=0; m < nocc; m++){
                Density(i,j) += C(i,m)*C(j,m);
            }
        }
    }
    return Density;
}

double scf_energy(Matrix D,Matrix H, Matrix F){
    double energy = 0.0;
    auto nrows = D.rows();
    auto ncols = D.cols();
    for( size_t i = 0; i<nrows ; i++){
        for(size_t j=0; j < ncols; j++){
            energy +=  D(j,i) * (H(i,j)+F(i,j));
        }
    }
    return energy;
}


double uhf_energy(Matrix Dt, Matrix H , Matrix Dalpha, Matrix Falpha, Matrix Dbeta, Matrix Fbeta){
    double uhf_energy = 0.0;
    auto nrows = Dt.rows();
    auto ncols = Dt.cols();
    for(auto i=0; i<nrows;i++){
        for(auto j=0;j<ncols;j++){
            uhf_energy += Dt(j,i)*H(i,j) + Dalpha(j,i)*Falpha(i,j) + Dbeta(j,i)*Fbeta(i,j);
        }
    }
    return 0.5*uhf_energy;
}

Matrix dot_prod(Matrix A, Matrix B){
    Matrix C = Eigen::MatrixXd::Zero(A.rows(),B.cols());
    for (auto i=0; i < A.rows() ;i++){
        for(auto j=0; j < B.cols() ; j++){
            for(auto k=0; k< A.cols(); k++){
                C(i,j) += A(i,k)*B(k,j);
            }
        }
    }
    return C;
}

struct inp_params{
    string scf = "RHF";
    string method = "HF";
    string basis= "STO-3G";
    double scf_convergence = 1e-10;
    int do_diis = 1;
    int spin_mult = 1;
    int charge = 0;
};





void RHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc){
    const auto nocc = nelec / 2;
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

    //Build Initial Guess Density
    Eigen::SelfAdjointEigenSolver<Matrix> init_fock_diag(init_guess_fock);
    Matrix C = S_inv*init_fock_diag.eigenvectors();
    Matrix D = make_density(C ,nocc);

    cout  << "Initial SCF energy: " << scf_energy(D,H,init_guess_fock)<< endl;
    //SCF LOOP
    double hf_energy = 0.0;
    int set_lenght = 8;
    vector<Matrix> error_set;
    vector<Matrix> fock_set;
    for (int i=0;i<100; i++){
        Matrix Fock =  H + make_fock(obs.shells(),D);
        //DIIS
        if(inpParams.do_diis==1){
            int Nerr = error_set.size();
            if(i>0){
                Matrix  error = dot_prod(Fock, dot_prod(D,S))
                                - dot_prod(S,dot_prod(D,Fock));
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
                Matrix zerovec = Eigen::MatrixXd::Zero(Nerr +1,1);
                zerovec(Nerr,0)=-1.0;
                for(auto a=0;a<Nerr;a++){
                    for(auto b=0;b<a+1;b++){
                        Matrix temp = dot_prod(error_set[a].transpose(),error_set[b]);
                        Bmat(a,b) = Bmat(b,a) = temp.trace();
                        Bmat(Nerr,a) = Bmat(a,Nerr) = -1.0;
                    }
                }
                if(Bmat.determinant() !=0.0){
                    Eigen::FullPivHouseholderQR<Matrix> solver(Bmat);
                    solver.setThreshold(1e-15);
                    Matrix coeff = solver.solve(zerovec);
                    Fock -=Fock;
                    for(int a=0;a<Nerr;a++){
                        Fock += coeff(a)*fock_set[a];
                    }
                }
                else if(Bmat.determinant()==0.0){
                    inpParams.do_diis = 0;
                    cout << "Bmat is singular" << endl;
                }
            }
        }
        Eigen::SelfAdjointEigenSolver<Matrix> fock_diag(S_inv.transpose()*Fock*S_inv);
        C = S_inv*fock_diag.eigenvectors();
        D = make_density(C,nocc);
        double new_energy = scf_energy(D,H,Fock);
        double delta_e = abs(new_energy - hf_energy);
        hf_energy = new_energy;
        cout << std::setprecision(15)<< "iter no :\t" << i+1 << "\t" << "Energy :\t"
             << hf_energy + enuc << "\t" << "Delta_E :\t" << delta_e << endl;
        if (delta_e <=inpParams.scf_convergence){
            cout << "Converged SCF energy :" << hf_energy +enuc<< "  a.u." << endl;
            auto mo_energy = fock_diag.eigenvalues();
            cout << "Eigenvalues of Fock Matrix:"<< endl;
            for(int a=0;a< Nbasis;a++){
                cout << a+1 << "\t" << mo_energy(a)<< endl;
            }
            break;
        }
    }
}

void UHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc) {
    // 1 body intergrals
    libint2::initialize();
    auto S = compute_1body_ints(obs.shells(), Operator::overlap, atoms);
    auto T = compute_1body_ints(obs.shells(), Operator::kinetic, atoms);
    auto V = compute_1body_ints(obs.shells(), Operator::nuclear, atoms);
    Matrix H = T + V;

    // initial guess fock = S^1/2.T * H * S^1/2
    Eigen::SelfAdjointEigenSolver<Matrix> ovlp_inv_sqrt(S);
    auto S_inv = ovlp_inv_sqrt.operatorInverseSqrt();
    Matrix init_guess_fock;
    init_guess_fock.noalias() = dot_prod(S_inv.transpose(),dot_prod(H , S_inv));

    //Build Initial Guess Density
    Eigen::SelfAdjointEigenSolver<Matrix> init_fock_diag(init_guess_fock);
    Matrix C = S_inv * init_fock_diag.eigenvectors();
    int nbeta = (nelec - inpParams.spin_mult + 1) / 2;
    int nalpha = nbeta + (inpParams.spin_mult - 1);
    cout << "Number of alpha electrons: " << nalpha << endl;
    cout << "Number of beta electrons:  " << nbeta << endl;
    Matrix Dalpha = make_density(C, nalpha);
    Matrix Dbeta = make_density(C, nbeta);
    Matrix Dt = Dalpha + Dbeta;
    cout << std::setprecision(15) << "initial energy :" << uhf_energy(Dt,H,Dalpha,init_guess_fock,Dbeta,init_guess_fock) << endl;

    double hf_energy = 0.0;
    for (int i = 0; i < 100; i++) {
        Matrix Falpha = H + make_fock_uhf(obs.shells(), Dt, Dalpha);
        Matrix Fbeta = H + make_fock_uhf(obs.shells(), Dt, Dbeta);
        Eigen::SelfAdjointEigenSolver<Matrix> falpha_diag(S_inv.transpose()*Falpha*S_inv);
        Matrix Calpha = S_inv*falpha_diag.eigenvectors();
        Dalpha = make_density(Calpha,nalpha);

        Eigen::SelfAdjointEigenSolver<Matrix> fbeta_diag(S_inv.transpose()*Fbeta*S_inv);
        Matrix Cbeta = S_inv*fbeta_diag.eigenvectors();
        Dbeta = make_density(Cbeta,nalpha);
        Dt = Dbeta+ Dalpha;
        double new_energy = uhf_energy(Dt,H,Dalpha,Falpha,Dbeta, Fbeta);
        double delta_e = abs(new_energy - hf_energy);
        hf_energy = new_energy;
        cout << std::setprecision(15)<< "iter no :\t" << i+1 << "\t" << "Energy :\t"
             << hf_energy + enuc << "\t" << "Delta_E :\t" << delta_e << endl;
        if (delta_e <=inpParams.scf_convergence){
            cout << "Converged SCF energy :" << hf_energy +enuc<< "  a.u." << endl;
            break;
        }
    }
}

