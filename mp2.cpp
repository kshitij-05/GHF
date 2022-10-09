//
// Created by Kshitij Surjuse on 10/1/22.
//
#include <iostream>
#include <string>
#include <vector>
//Libint
#include <libint2/util/intpart_iter.h>
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

size_t nbasis(const std::vector<libint2::Shell>& shells);
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);
using real_t = libint2::scalar_type;

struct scf_results{
    bool is_rhf=1;
    int nelec, nbasis,no,nv , nalpha,nbeta;
    Matrix C,F,Calpha,Cbeta,Falpha,Fbeta;
    double scf_energy;
};

Tensor<double> make_ao_ints(const std::vector<libint2::Shell>& shells) {
    // construct the 2-electron repulsion integrals engine
    size_t Nbasis = nbasis(shells);
    Tensor<double> ERI (Nbasis,Nbasis,Nbasis,Nbasis);
    Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
    auto shell2bf = map_shell_to_basis_function(shells);
    const auto& buf = engine.results();
    for(auto s1=0; s1!=shells.size(); ++s1) {
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = shells[s1].size();   // number of basis functions in this shell
        for(auto s2=0; s2<=s1; ++s2) {
            auto bf2_first = shell2bf[s2];
            auto n2 = shells[s2].size();
            for(auto s3=0; s3<=s1; ++s3) {
                auto bf3_first = shell2bf[s3];
                auto n3 = shells[s3].size();
                const auto s4_max = (s1 == s3) ? s2 : s3;
                for(auto s4=0; s4<=s4_max; ++s4) {
                    auto bf4_first = shell2bf[s4];
                    auto n4 = shells[s4].size();
                    engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
                    const auto* buf_1234 = buf[0];
                    if (buf_1234 == nullptr)
                        continue;
                    for(auto f1=0, f1234=0; f1!=n1; ++f1) {
                        const auto bf1 = f1 + bf1_first;
                        for(auto f2=0; f2!=n2; ++f2) {
                            const auto bf2 = f2 + bf2_first;
                            for(auto f3=0; f3!=n3; ++f3) {
                                const auto bf3 = f3 + bf3_first;
                                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                                    const auto bf4 = f4 + bf4_first;
                                    const auto value = buf_1234[f1234];
                                    //cout << bf1 << " "<< bf2 << " "<< bf3 << " "<< bf4 << " "<< value << endl;
                                    ERI(bf1,bf2,bf3,bf4) = value;
                                    ERI(bf1,bf2,bf4,bf3) = value;
                                    ERI(bf2,bf1,bf3,bf4) = value;
                                    ERI(bf2,bf1,bf4,bf3) = value;
                                    ERI(bf3,bf4,bf1,bf2) = value;
                                    ERI(bf3,bf4,bf2,bf1) = value;
                                    ERI(bf4,bf3,bf1,bf2) = value;
                                    ERI(bf4,bf3,bf2,bf1) = value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return ERI;
}

Tensor<double> aotomo(const Tensor<double>& eri ,const Matrix& coeffs1 ,
                      const Matrix& coeffs2 ,const Matrix& coeffs3 ,const Matrix& coeffs4){
    const auto nbasis = eri.extent(0);
    Tensor<double> C1(nbasis,nbasis);
    Tensor<double> C2(nbasis,nbasis);
    Tensor<double> C3(nbasis,nbasis);
    Tensor<double> C4(nbasis,nbasis);
    for(auto i=0;i<nbasis;i++){
        for(auto j=0; j<nbasis;j++){
            C1(i,j)= coeffs1(i,j);
            C2(i,j)= coeffs2(i,j);
            C3(i,j)= coeffs3(i,j);
            C4(i,j)= coeffs4(i,j);
        }
    }
    Tensor<double> temp1(nbasis,nbasis,nbasis,nbasis);
    Tensor<double> moints(nbasis,nbasis,nbasis,nbasis);
    const char p ='p';
    const char q ='q';
    const char r ='r';
    const char s ='s';
    const char i ='i';
    const char j ='j';
    const char k ='k';
    const char l ='l';

    contract(1.0, eri, {p,q,r,s}, C1, {s,l}, 1.0, temp1, {p,q,r,l});
    contract(1.0, temp1, {p,q,r,l}, C2, {r,k}, 1.0, moints, {p,q,k,l});
    temp1 -=temp1;
    contract(1.0,C3,{q,j} , moints, {p,q,k,l},1.0,temp1,{p,j,k,l});
    moints -=moints;
    contract(1.0,C4,{p,i} , temp1, {p,j,k,l},1.0,moints,{i,j,k,l});
    return moints;
}

Tensor<double> space2spin(const Tensor<double>& moints){
    const auto n2 = 2*moints.extent(0);
    Tensor<double> soints (n2,n2,n2,n2);
    for(auto p=0;p<n2;p++){
        for(auto q=0;q<n2;q++){
            for(auto r=0;r<n2;r++){
                for(auto s=0;s<n2;s++){
                    const auto value1 = moints(p/2,r/2,q/2,s/2) * (p%2 == r%2) * (q%2 == s%2);
                    const auto value2 = moints(p/2,s/2,q/2,r/2) * (p%2 == s%2) * (q%2 == r%2);
                    soints(p,q,r,s) = value1 - value2;
                }
            }
        }
    }
    return soints;
}

Tensor<double> fockspace2spin(const Matrix& Fock){
    auto nbasis = Fock.rows();
    Tensor<double> fock_spin (nbasis*2,nbasis*2);
    for(auto i=0;i<nbasis*2;i++){
            fock_spin(i,i) = Fock(i/2,i/2);
    }
    return fock_spin;
}

Tensor<double> get_oovv(const Tensor<double>& soints , int nocc, int nvir){
    Tensor<double> oovv (nocc,nocc,nvir,nvir);
    for(auto i=0;i<nocc; i++){
        for(auto j=0;j<nocc;j++){
            for(auto a=0;a<nvir;a++){
                for(auto b=0;b<nvir;b++){
                    oovv(i,j,a,b) = soints(i,j,nocc+a,nocc+b);
                }
            }
        }
    }
    return oovv;
}

Tensor<double> denom(const Tensor<double>& F,int no, int nv){
    Tensor<double> eijab(no,no,nv,nv);
    for(auto i=0;i<no; i++){
        for(auto j=0;j<no;j++){
            for(auto a=0;a<nv;a++){
                for(auto b=0;b<nv;b++){
                    eijab(i,j,a,b) += F(i,i)+F(j,j)-F(no+a,no+a)-F(no+b,no+b);
                }
            }
        }
    }
    return  eijab;
}

Tensor<double> space2spin_uhf(const Tensor<double>& Maa,const Tensor<double>& Mbb ,const Tensor<double>& Mab){
    const auto n = Maa.extent(0);
    const auto n2 = 2*n;
    Tensor<double> soints (n2,n2,n2,n2);
    for(auto i=0;i<n2;i++){
        for(auto j=0;j<n2;j++){
            for(auto k=0;k<n2;k++){
                for(auto l=0;l<n2;l++){
                    if(i%2==0 && j%2==0 && k%2==0 && l%2==0){
                        soints(i,j,k,l) = Maa(i/2,k/2,j/2,l/2)
                                - Maa(j/2,k/2,i/2,l/2);
                    }
                    else if(i%2==1 && j%2==1 && k%2==1 && l%2==1){
                        soints(i,j,k,l) = Mbb(i/2,k/2,j/2,l/2)
                                - Mbb(j/2,k/2,i/2,l/2);
                    }
                    else if(i%2==0 && j%2==1 && k%2==0 && l%2==1){
                        soints(i,j,k,l) = Mab(i/2,k/2,j/2,l/2);
                    }
                    else if(i%2==1 && j%2==0 && k%2==1 && l%2==0) {
                        soints(i,j,k,l) = Mab(i/2,k/2,j/2,l/2);
                    }
                    else if(i%2==1 && j%2==0 && k%2==0 && l%2==1){
                        soints(i,j,k,l) = - Mab(j/2,k/2,i/2,l/2);
                    }
                    else if(i%2==0 && j%2==1 && k%2==1 && l%2==0){
                        soints(i,j,k,l) = - Mab(j/2,k/2,i/2,l/2);
                    }
                }
            }
        }
    }
    return soints;
}

double mp2_energy(Tensor<double> eri,scf_results& SCF){
    double emp2=0.0;
    if(SCF.is_rhf){
        int no = SCF.no;
        int nv = SCF.nv;
        Tensor<double> moints = aotomo(eri,SCF.C,SCF.C,SCF.C,SCF.C);
        Tensor<double> soints = space2spin(moints);
        Tensor<double> oovv = get_oovv(soints,no*2,nv*2);
        Tensor<double> moe = fockspace2spin(SCF.F);
        Tensor<double> eijab = denom(moe,no*2,nv*2);
        int no_so = no*2;
        int nv_so = nv*2;
        for(auto i=0;i<no_so; i++){
            for(auto j=0;j<no_so;j++){
                for(auto a=0;a<nv_so;a++){
                    for(auto b=0;b<nv_so;b++){
                        emp2+= 0.25*(oovv(i,j,a,b)*oovv(i,j,a,b))
                               /eijab(i,j,a,b);
                    }
                }
            }
        }
    }

    else if(SCF.is_rhf == 0){
        Tensor<double> Malpha_alpha = aotomo(eri,SCF.Calpha,SCF.Calpha,SCF.Calpha,SCF.Calpha);
        Tensor<double> Mbeta_beta   = aotomo(eri,SCF.Cbeta,SCF.Cbeta,SCF.Cbeta,SCF.Cbeta);
        Tensor<double> Malpha_beta  = aotomo(eri,SCF.Calpha,SCF.Calpha,SCF.Cbeta,SCF.Cbeta);
        Tensor<double> soints = space2spin_uhf(Malpha_alpha,Mbeta_beta,Malpha_beta);
        //Tensor<double> anti_symm_ints = anti_sym(soints);
        cout << "no. of occupied: "<< SCF.no << endl;
        cout << "no. of virtuals: "<< SCF.nv << endl;
        Tensor<double> oovv = get_oovv(soints,SCF.no,SCF.nv);
        Tensor<double> Moe (SCF.nbasis*2,SCF.nbasis*2);
        for(auto i=0;i<SCF.nbasis*2;i++){
            if(i%2==0){Moe(i,i)=SCF.Falpha(i/2,i/2);}
            else if(i%2==1){Moe(i,i)=SCF.Fbeta(i/2,i/2);}
        }
        Tensor<double> eijab = denom(Moe,SCF.no,SCF.nv);
        int no_so = SCF.no;
        int nv_so = SCF.nv;
        for(auto i=0;i<no_so; i++){
            for(auto j=0;j<no_so;j++){
                for(auto a=0;a<nv_so;a++){
                    for(auto b=0;b<nv_so;b++){
                        emp2+= 0.25*(oovv(i,j,a,b)*oovv(i,j,a,b))/eijab(i,j,a,b);
                    }
                }
            }
        }
    }
    return emp2;
}
