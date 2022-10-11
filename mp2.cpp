//
// Created by Kshitij Surjuse on 10/1/22.
//
#include <iostream>
#include <string>
#include <vector>
//Libint
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
using libint2::Engine;
using libint2::Operator;
using libint2::BasisSet;
using std::vector;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> DiagonalMatrix;
using real_t = libint2::scalar_type;
struct scf_results{
    bool is_rhf=1;
    int nelec, nbasis,no,nv , nalpha,nbeta;
    Matrix C,F,Calpha,Cbeta,Falpha,Fbeta;
    double scf_energy;
};
size_t nbasis(const std::vector<libint2::Shell>& shells);
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);
Tensor<double> make_ao_ints(const std::vector<libint2::Shell>& shells);
Tensor<double> aotomo(const Tensor<double>& eri ,const Matrix& coeffs1 ,
                      const Matrix& coeffs2 ,const Matrix& coeffs3 ,const Matrix& coeffs4);
Tensor<double> get_int(const Tensor<double>& soints , int nocc, int nvir ,string int_type);
Tensor<double> denom(const Tensor<double>& F,int no, int nv);
Tensor<double> space2spin(const Tensor<double>& Maa,const Tensor<double>& Mbb ,const Tensor<double>& Mab);


Tensor<double> fockspace2spin(const Matrix& Fock){
    auto nbasis = Fock.rows();
    Tensor<double> fock_spin (nbasis*2,nbasis*2);
    for(auto i=0;i<nbasis*2;i++){
        fock_spin(i,i) = Fock(i/2,i/2);
    }
    return fock_spin;
}

struct mp2_results{
    Tensor<double> t2_mp2;
    double mp2_energy = 0.0;
};

mp2_results mp2_energy(const Tensor<double>& oovv , const Tensor<double>& eijab){
    double emp2 = 0.0;
    mp2_results  results;
    auto no_so = oovv.extent(0);
    auto nv_so = oovv.extent(2);

    Tensor<double> t2(no_so,no_so,nv_so,nv_so);
    for(auto i=0;i<no_so; i++){
        for(auto j=0;j<no_so;j++){
            for(auto a=0;a<nv_so;a++){
                for(auto b=0;b<nv_so;b++){
                    t2(i,j,a,b) = oovv(i,j,a,b)
                            /eijab(i,j,a,b);
                    emp2+= 0.25*(oovv(i,j,a,b)*oovv(i,j,a,b))
                           /eijab(i,j,a,b);
                }
            }
        }
    }
    results.mp2_energy = emp2;
    results.t2_mp2 = t2;
    return results;
}

mp2_results mp2(Tensor<double> eri,scf_results& SCF){
    mp2_results results;
    double emp2=0.0;
    if(SCF.is_rhf){
        int no = SCF.no;
        int nv = SCF.nv;
        Tensor<double> moints = aotomo(eri,SCF.C,SCF.C,SCF.C,SCF.C);
        Tensor<double> soints = space2spin(moints,moints,moints);
        Tensor<double> oovv = get_int(soints,no,nv, "oovv");
        Tensor<double> moe = fockspace2spin(SCF.F);
        Tensor<double> eijab = denom(moe,no,nv);
        results=  mp2_energy(oovv,eijab);
    }

    else if(SCF.is_rhf == 0){
        Tensor<double> Malpha_alpha = aotomo(eri,SCF.Calpha,SCF.Calpha,SCF.Calpha,SCF.Calpha);
        Tensor<double> Mbeta_beta   = aotomo(eri,SCF.Cbeta,SCF.Cbeta,SCF.Cbeta,SCF.Cbeta);
        Tensor<double> Malpha_beta  = aotomo(eri,SCF.Calpha,SCF.Calpha,SCF.Cbeta,SCF.Cbeta);
        Tensor<double> soints = space2spin(Malpha_alpha,Mbeta_beta,Malpha_beta);
        cout << "no. of occupied: "<< SCF.no << endl;
        cout << "no. of virtuals: "<< SCF.nv << endl;
        Tensor<double> oovv = get_int(soints,SCF.no,SCF.nv,"oovv");
        Tensor<double> Moe (SCF.nbasis*2,SCF.nbasis*2);
        for(auto i=0;i<SCF.nbasis*2;i++){
            if(i%2==0){Moe(i,i)=SCF.Falpha(i/2,i/2);}
            else if(i%2==1){Moe(i,i)=SCF.Fbeta(i/2,i/2);}
        }
        Tensor<double> eijab = denom(Moe,SCF.no,SCF.nv);
        results = mp2_energy(oovv,eijab);
    }
    return results;
}
