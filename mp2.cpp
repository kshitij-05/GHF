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


Tensor<double> make_ao_ints_simple(const std::vector<libint2::Shell>& shells){
    size_t Nbasis = nbasis(shells);
    Tensor<double> ERI (Nbasis,Nbasis,Nbasis,Nbasis);
    Engine engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
    auto shell2bf = map_shell_to_basis_function(shells);
    // buf[0] points to the target shell set after every call  to engine.compute()
    const auto& buf = engine.results();
    // loop over shell pairs of the Fock matrix, {s1,s2}
    // Fock matrix is symmetric, but skipping it here for simplicity (see compute_2body_fock)
    for(auto s1=0; s1!=shells.size(); ++s1) {
        auto bf1_first = shell2bf[s1]; // first basis function in this shell
        auto n1 = shells[s1].size();
        for(auto s2=0; s2!=shells.size(); ++s2) {
            auto bf2_first = shell2bf[s2];
            auto n2 = shells[s2].size();
            // loop over shell pairs of the density matrix, {s3,s4}
            // again symmetry is not used for simplicity
            for(auto s3=0; s3!=shells.size(); ++s3) {
                auto bf3_first = shell2bf[s3];
                auto n3 = shells[s3].size();
                for(auto s4=0; s4!=shells.size(); ++s4) {
                    auto bf4_first = shell2bf[s4];
                    auto n4 = shells[s4].size();
                    // Coulomb contribution to the Fock matrix is from {s1,s2,s3,s4} integrals
                    engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
                    const auto* buf_1234 = buf[0];
                    if (buf_1234 == nullptr)
                        continue; // if all integrals screened out, skip to next quartet
                    // we don't have an analog of Eigen for tensors (yet ... see github.com/BTAS/BTAS, under development)
                    // hence some manual labor here:
                    // 1) loop over every integral in the shell set (= nested loops over basis functions in each shell)
                    // and 2) add contribution from each integral
                    for(auto f1=0, f1234=0; f1!=n1; ++f1) {
                        const auto bf1 = f1 + bf1_first;
                        for(auto f2=0; f2!=n2; ++f2) {
                            const auto bf2 = f2 + bf2_first;
                            for(auto f3=0; f3!=n3; ++f3) {
                                const auto bf3 = f3 + bf3_first;
                                for(auto f4=0; f4!=n4; ++f4, ++f1234) {
                                    const auto bf4 = f4 + bf4_first;
                                    ERI(bf1,bf2,bf3,bf4) = buf_1234[f1234];
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


Tensor<double> aotomo_simple(Tensor<double> eri ,Matrix coeffs){
    int nbasis = eri.extent(0);
    Tensor<double> C(nbasis,nbasis);
    for(auto i=0;i<nbasis;i++){
        for(auto j=0; j<nbasis;j++){
            C(i,j)= coeffs(i,j);
        }
    }
    Tensor<double> temp1(nbasis,nbasis,nbasis,nbasis);
    Tensor<double> temp2(nbasis,nbasis,nbasis,nbasis);
    Tensor<double> temp3(nbasis,nbasis,nbasis,nbasis);
    Tensor<double> moints(nbasis,nbasis,nbasis,nbasis);
    for(auto mu=0;mu<nbasis;mu++){
        for(auto i=0;i<nbasis;i++){
            for(auto x=0;x<nbasis;x++){
                for(auto y=0;y<nbasis;y++){
                    for(auto z=0;z<nbasis;z++){
                        temp1(mu,x,y,z) +=C(i,mu)*eri(i,x,y,z);
                    }
                }
            }
        }
        for(auto nu=0;nu<nbasis;nu++){
            for(auto j=0;j<nbasis;j++){
                for(auto w=0;w<nbasis;w++){
                    for(auto v=0;v<nbasis;v++){
                        temp2(mu,nu,w,v)+=C(j,nu)*temp1(mu,j,w,v);
                    }
                }
            }
            for(auto lam=0;lam<nbasis;lam++){
                for(auto k=0;k<nbasis;k++){
                    for(auto u=0;u<nbasis;u++){
                        temp3(mu,nu,lam,u) += C(k,lam)*temp2(mu,nu,k,u);
                    }
                }
                for(auto sig=0;sig<nbasis;sig++){
                    for(auto l=0;l<nbasis;l++){
                        moints(mu,nu,lam,sig)+=C(l,sig)*temp3(mu,nu,lam,l);
                    }
                }
            }
        }
    }
    return moints;
}



Tensor<double> aotomo(Tensor<double> eri ,Matrix coeffs){
    int nbasis = eri.extent(0);

    Tensor<double> C(nbasis,nbasis);
    for(auto i=0;i<nbasis;i++){
        for(auto j=0; j<nbasis;j++){
            C(i,j)= coeffs(i,j);
        }
    }
    Tensor<double> temp1(nbasis,nbasis,nbasis,nbasis);
    Tensor<double> temp2(nbasis,nbasis,nbasis,nbasis);
    Tensor<double> temp3(nbasis,nbasis,nbasis,nbasis);
    Tensor<double> moints(nbasis,nbasis,nbasis,nbasis);
    const char p ='p';
    const char q ='q';
    const char r ='r';
    const char s ='s';
    const char i ='i';
    const char j ='j';
    const char k ='k';
    const char l ='l';

    contract(1.0, eri, {p,q,r,s}, C, {s,l}, 1.0, temp1, {p,q,r,l});
    contract(1.0, temp1, {p,q,r,l}, C, {r,k}, 1.0, temp2, {p,q,k,l});
    contract(1.0,C,{q,j} , temp2, {p,q,k,l},1.0,temp3,{p,j,k,l});
    contract(1.0,C,{p,i} , temp3, {p,j,k,l},1.0,moints,{i,j,k,l});
    return moints;
}


Tensor<double> space2spin(Tensor<double> moints){
    int n2 = 2*moints.extent(0);
    Tensor<double> soints (n2,n2,n2,n2);
    for(auto p=0;p<n2;p++){
        for(auto q=0;q<n2;q++){
            for(auto r=0;r<n2;r++){
                for(auto s=0;s<n2;s++){
                    soints(p,q,r,s) = moints(p/2,r/2,q/2,s/2)- moints(p/2,r/2,s/2,q/2);
                }
            }
        }
    }

    return soints;
}

Tensor<double> fockspace2spin(Matrix Fock){
    int nbasis = Fock.rows();
    Tensor<double> fock_spin (nbasis*2,nbasis*2);
    for(auto i=0;i<nbasis*2;i++){
            fock_spin(i,i) = Fock(i/2,i/2);
    }
    return fock_spin;
}

Tensor<double> get_oovv(Tensor<double> moints , int nocc, int nvir){
    Tensor<double> oovv (nocc*2,nocc*2,nvir*2,nvir*2);
    Tensor<double> soints = space2spin(moints);
    for(auto i=0;i<nocc*2; i++){
        for(auto j=0;j<nocc*2;j++){
            for(auto a=0;a<nvir*2;a++){
                for(auto b=0;b<nvir*2;b++){
                    oovv(i,j,a,b) = soints(i,j,nocc+a,nocc+b);
                }
            }
        }
    }
    return oovv;
}


Tensor<double> denom(Tensor<double> F,int no, int nv){
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


double mp2_rhf(Tensor<double> eri, Matrix coeffs, Matrix F, int no, int nv){
    double emp2 = 0.0;
    Tensor<double> moints = aotomo_simple(eri,coeffs);
    for(auto i=0;i<no;i++){
        for(auto j=0;j<no;j++){
            for(auto a=0;a<nv;a++){
                for(auto b=0;b<nv;b++){
                    const auto v1 = moints(i,no+a,j,no+b)*
                            (2.0*moints(i,no+a,j,no+b) - moints(i,no+b,j,no+a));
                    const auto v2 =( F(i,i)+F(j,j)-F(no+a,no+a)-F(no+b,no+b) );
                    emp2 += v1/v2;
                }
            }
        }
    }

    return emp2;
}


double mp2_energy(Tensor <double> eri, Matrix coeffs, Matrix F , int no, int nv){
    Tensor<double> moints = aotomo(eri,coeffs);
    Tensor<double> oovv = get_oovv(moints,no,nv);
    Tensor<double> moe = fockspace2spin(F);
    Tensor<double> eijab = denom(moe,no*2,nv*2);
    double emp2=0.0;
    for(auto i=0;i<no*2; i++){
        for(auto j=0;j<no*2;j++){
            for(auto a=0;a<nv*2;a++){
                for(auto b=0;b<nv*2;b++){
                    cout << oovv(i,j,a,b)<< endl;
                    //emp2+= 0.25*(oovv(i,j,a,b)*oovv(i,j,a,b))/eijab(i,j,a,b);
                }
            }
        }
    }
    return emp2;
}


