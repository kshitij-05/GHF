//
// Created by Kshitij Surjuse on 9/20/22.
//
//Eigen
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> DiagonalMatrix;

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
            energy += D(i,j) * (H(i,j)+F(i,j));
        }
    }
    return energy;
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