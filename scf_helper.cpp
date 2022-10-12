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

Matrix make_fock_uhf_complex(const std::vector<libint2::Shell>& shells,
                     const Matrix& Dt, const Matrix& Dspin);

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

double uhf_energy(Matrix Dt, Matrix H , Matrix Dalpha,
                  Matrix Falpha, Matrix Dbeta, Matrix Fbeta){
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
    int singles = 1;
    int spin_mult = 1;
    int charge = 0;
    string unit = "A";
};

Matrix make_X(Matrix S){
    int n = S.rows();
    Eigen::SelfAdjointEigenSolver<Matrix> S_diag(S);
    auto eigenvec  = S_diag.eigenvectors();
    auto eigenvalues = S_diag.eigenvalues();
    Matrix half_eigen = Matrix::Zero(n,n);
    for(auto i=0;i<n;i++){
        half_eigen(i,i) = pow(eigenvalues(i),-0.5);
    }
    Matrix temp  = dot_prod(eigenvec, half_eigen);
    Matrix X = dot_prod(temp,eigenvec.transpose());
    return X;
}

struct scf_results{
    bool is_rhf=1;
    int nelec,nbasis,no,nv , nalpha,nbeta;
    Matrix C,F,Calpha,Cbeta,Falpha,Fbeta;
    double scf_energy;
};

Matrix guess_density(int no, int nbasis){
    Matrix gD = Matrix::Zero(nbasis,nbasis);
    for( auto i=0;i<no;i++){
        gD(i,i) =1.0;
    }
    return gD;
}

scf_results RHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc){
    scf_results results;
    const auto nocc = nelec / 2;
    results.nelec = nelec;
    results.nbasis = Nbasis;
    results.no = nelec;
    results.nv = 2*Nbasis-nelec;
    // 1 body intergrals
    libint2::initialize();
    auto S = compute_1body_ints(obs.shells(), Operator::overlap , atoms);
    auto T = compute_1body_ints(obs.shells(), Operator::kinetic , atoms);
    auto V = compute_1body_ints(obs.shells(), Operator::nuclear , atoms);
    Matrix H = T+V;

    // initial guess fock = S^1/2.T * H * S^1/2
    auto S_inv = make_X(S);
    Matrix init_guess_fock;
    init_guess_fock = dot_prod(S_inv.transpose(), dot_prod(H,S_inv));

    //Build Initial Guess Density
    Eigen::SelfAdjointEigenSolver<Matrix> init_fock_diag(init_guess_fock);
    Matrix C = dot_prod(S_inv,init_fock_diag.eigenvectors());
    Matrix D = guess_density(nocc,Nbasis);
    cout  << std::setprecision(12)<< "Initial SCF energy: " << scf_energy(D,H,H)<< endl;
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

        Matrix diag = dot_prod(S_inv.transpose(), dot_prod(Fock,S_inv));
        Eigen::SelfAdjointEigenSolver<Matrix> fock_diag(diag);
        C = dot_prod(S_inv,fock_diag.eigenvectors());
        D = make_density(C,nocc);
        double new_energy = scf_energy(D,H,Fock);
        double delta_e = abs(new_energy - hf_energy);
        hf_energy = new_energy;
        if(i==0){cout << "iter      " << "Energy (a.u.)\t\t"<< "Delta_E :\t"<< endl;}
        cout << std::setprecision(13)<< i+1 << "\t"
             << hf_energy + enuc << "\t\t"  << delta_e << endl;
        auto mo_energy = fock_diag.eigenvalues();
        if (delta_e <=inpParams.scf_convergence){
            cout << "Converged SCF energy :" << hf_energy +enuc<< "  a.u." << endl;
            cout << "Eigenvalues of Fock Matrix:"<< endl;
            for(int a=0;a< Nbasis;a++){
                cout << a+1 << "\t" << mo_energy(a)<< endl;
            }
            Matrix F  = Matrix::Zero(Nbasis,Nbasis);
            for(auto i=0;i<Nbasis;i++){
                F(i,i) = mo_energy(i);
            }
            results.scf_energy = hf_energy+enuc;
            results.F = F;
            results.C = C;
            break;
        }
    }
    return results;
}

scf_results UHF(BasisSet obs, vector<libint2::Atom> atoms,int Nbasis, int nelec, inp_params inpParams , double enuc) {
    scf_results results;
    const auto nocc = nelec;
    results.is_rhf = 0;
    results.nelec = nelec;
    results.nbasis = Nbasis;
    results.no = nelec;
    results.nv = 2*Nbasis-nelec;
    // 1 body intergrals
    libint2::initialize();
    auto S = compute_1body_ints(obs.shells(), Operator::overlap, atoms);
    auto T = compute_1body_ints(obs.shells(), Operator::kinetic, atoms);
    auto V = compute_1body_ints(obs.shells(), Operator::nuclear, atoms);
    Matrix H = T + V;

    // initial guess fock = S^1/2.T * H * S^1/2
    auto S_inv = make_X(S);
    Matrix init_guess_fock;
    init_guess_fock = dot_prod(S_inv.transpose(), dot_prod(H,S_inv));

    //Build Initial Guess Density
    Eigen::SelfAdjointEigenSolver<Matrix> init_fock_diag(init_guess_fock);
    Matrix C = dot_prod(S_inv,init_fock_diag.eigenvectors());
    int nbeta = (nelec - inpParams.spin_mult + 1) / 2;
    int nalpha = nbeta + (inpParams.spin_mult - 1);
    results.nalpha = nalpha;
    results.nbeta = nbeta;
    cout << "Number of alpha electrons: " << nalpha << endl;
    cout << "Number of beta electrons:  " << nbeta << endl;
    Matrix Dalpha = guess_density(nalpha,Nbasis);
    Matrix Dbeta = guess_density(nbeta,Nbasis);
    Matrix Dt = Dalpha + Dbeta;
    cout << std::setprecision(12) << "initial energy :" << uhf_energy(Dt,H,Dalpha,H,Dbeta,H) << endl;

    double hf_energy = 0.0;
    int set_lenght = 6;
    vector<Matrix> era,erb,fsa,fsb;

    for (int i = 0; i < 200; i++) {
        Matrix Falpha = H + make_fock_uhf(obs.shells(), Dt, Dalpha);
        Matrix Fbeta = H + make_fock_uhf(obs.shells(), Dt, Dbeta);

        if(inpParams.do_diis==1){
            int Nerr = era.size();
            if(i>0){
                Matrix  aerr = dot_prod(Falpha, dot_prod(Dalpha,S))
                                - dot_prod(S,dot_prod(Dalpha,Falpha));
                Matrix  berr = dot_prod(Fbeta, dot_prod(Dbeta,S))
                               - dot_prod(S,dot_prod(Dbeta,Fbeta));
                if(Nerr < set_lenght){
                    era.push_back(aerr);
                    erb.push_back(berr);
                    fsa.push_back(Falpha);
                    fsb.push_back(Fbeta);
                }
                else if(Nerr>=set_lenght){
                    era.erase(era.begin());
                    erb.erase(erb.begin());
                    fsa.erase(fsa.begin());
                    fsb.erase(fsb.begin());
                    era.push_back(aerr);
                    erb.push_back(berr);
                    fsa.push_back(Falpha);
                    fsb.push_back(Fbeta);
                }
            }
            if(Nerr>=2){
                Matrix Bmata = Matrix::Zero(Nerr+1,Nerr+1);
                Matrix zeroveca = Matrix::Zero(Nerr +1,1);
                Matrix Bmatb = Matrix::Zero(Nerr+1,Nerr+1);
                Matrix zerovecb = Matrix::Zero(Nerr +1,1);
                zeroveca(Nerr,0)=-1.0;
                zerovecb(Nerr,0)=-1.0;
                for(auto a=0;a<Nerr;a++){
                    for(auto b=0;b<a+1;b++){
                        Matrix tempa = dot_prod(era[a].transpose(),era[b]);
                        Matrix tempb = dot_prod(erb[a].transpose(),erb[b]);
                        Bmata(a,b) = Bmata(b,a) = tempa.trace();
                        Bmatb(a,b) = Bmatb(b,a) = tempb.trace();
                        Bmata(Nerr,a) = Bmata(a,Nerr) = -1.0;
                        Bmatb(Nerr,a) = Bmatb(a,Nerr) = -1.0;
                    }
                }

                if(Bmata.determinant() !=0.0 && Bmatb.determinant() !=0.0) {
                    Eigen::FullPivHouseholderQR<Matrix> solvera(Bmata);
                    Eigen::FullPivHouseholderQR<Matrix> solverb(Bmatb);
                    solvera.setThreshold(1e-15);
                    solverb.setThreshold(1e-15);
                    Matrix coeffa = solvera.solve(zeroveca);
                    Matrix coeffb = solverb.solve(zerovecb);
                    Falpha -=Falpha;
                    Fbeta -=Fbeta;
                    for(int a=0;a<Nerr;a++){
                        Falpha += coeffa(a)*fsa[a];
                        Fbeta += coeffb(a)*fsb[a];
                    }
                }
                else if(Bmata.determinant()==0.0){
                    inpParams.do_diis = 0;
                    cout << "Bmat is singular" << endl;
                }
            }
        }

        Matrix tempa = dot_prod(S_inv.transpose(), dot_prod(Falpha,S_inv));
        Eigen::SelfAdjointEigenSolver<Matrix> falpha_diag(tempa);
        Matrix Calpha = dot_prod(S_inv,falpha_diag.eigenvectors());
        Dalpha = make_density(Calpha,nalpha);
        Matrix tempb = dot_prod(S_inv.transpose(), dot_prod(Fbeta,S_inv));
        Eigen::SelfAdjointEigenSolver<Matrix> fbeta_diag(tempb);
        Matrix Cbeta = dot_prod(S_inv,fbeta_diag.eigenvectors());
        Dbeta = make_density(Cbeta,nbeta);
        Dt = Dbeta+ Dalpha;
        double new_energy = uhf_energy(Dt,H,Dalpha,Falpha,Dbeta, Fbeta);
        double delta_e = abs(new_energy - hf_energy);
        hf_energy = new_energy;
        if(i==0){cout << "iter      " << "Energy (a.u.)\t\t"<< "Delta_E :\t"<< endl;}
        cout << std::setprecision(13)<< i+1 << "\t"
             << hf_energy + enuc << "\t\t"  << delta_e << endl;
        auto MOEa = falpha_diag.eigenvalues();
        auto MOEb = fbeta_diag.eigenvalues();
        if (delta_e <=inpParams.scf_convergence){
            cout << std::setprecision(15)<< "Converged SCF energy :" << hf_energy +enuc<< "  a.u." << endl << endl;
            Matrix Fa  = Matrix::Zero(Nbasis,Nbasis);
            Matrix Fb  = Matrix::Zero(Nbasis,Nbasis);
            Matrix Moe = Matrix::Zero(Nbasis*2,Nbasis*2);
            cout << "index  \t" << "Alpha eigvals\t" << "\t" << "Beta eigvals\t" << endl;
            for(auto i=0;i<Nbasis;i++){
                Fa(i,i) = MOEa(i);
                Fb(i,i) = MOEb(i);
                cout << i << "\t"<< MOEa(i) << "\t"<< MOEb(i)<< endl;
            }
            for(auto i=0;i<2*Nbasis;i++){
                Moe(i,i)=(i%2==0)*MOEa(i/2) + (i%2==1)*MOEb(i/2);
            }
            results.F = Moe;
            results.scf_energy = hf_energy+enuc;
            results.Falpha = Fa;
            results.Fbeta = Fb;
            results.Calpha = Calpha;
            results.Cbeta = Cbeta;
            break;
        }
    }
    return results;
}

