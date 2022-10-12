//
// Created by Kshitij Surjuse on 10/10/22.
//
#include <string>
#include <iostream>
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
size_t nbasis(const std::vector<libint2::Shell>& shells);
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells);
using real_t = libint2::scalar_type;

struct scf_results{
    bool is_rhf=1;
    int nelec, nbasis,no,nv , nalpha,nbeta;
    Matrix C,F,Calpha,Cbeta,Falpha,Fbeta;
    double scf_energy;
};
struct INTEGRALS{
    Tensor<double> vvvv;
    Tensor<double> vvvo;
    Tensor<double> oovv;
    Tensor<double> ovvo;
    Tensor<double> ovov;
    Tensor<double> ooov;
    Tensor<double> ovoo;
    Tensor<double> oooo;
    Tensor<double> F;
    Tensor<double> Fii;
    Tensor<double> Faa;
    Tensor<double> Fia;
    Tensor<double> eia;
    Tensor<double> eijab;
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

Tensor<double> get_int(const Tensor<double>& soints , int nocc, int nvir ,string int_type){
    auto nt = soints.extent(0);
    int n1 = (int_type[0] == 'o')*nocc + (int_type[0]== 'v')*nvir;
    int n2 = (int_type[1] == 'o')*nocc + (int_type[1]== 'v')*nvir;
    int n3 = (int_type[2] == 'o')*nocc + (int_type[2]== 'v')*nvir;
    int n4 = (int_type[3] == 'o')*nocc + (int_type[3]== 'v')*nvir;
    int i1 = (int_type[0] == 'o')*0 + (int_type[0]== 'v')*nocc;
    int i2 = (int_type[1] == 'o')*0 + (int_type[1]== 'v')*nocc;
    int i3 = (int_type[2] == 'o')*0 + (int_type[2]== 'v')*nocc;
    int i4 = (int_type[3] == 'o')*0 + (int_type[3]== 'v')*nocc;
    Tensor<double> slice_int (n1,n2,n3,n4);
    for(auto i=0;i<n1; i++){
        for(auto j=0;j<n2;j++){
            for(auto a=0;a<n3;a++){
                for(auto b=0;b<n4;b++){
                    slice_int(i,j,a,b) = soints(i1+i,i2+j,i3+a,i4+b);
                }
            }
        }
    }
    return slice_int;
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

Tensor<double> Dia(const Tensor<double>& Fii,const Tensor<double>& Faa){
    auto no = Fii.extent(0);
    auto nv = Faa.extent(0);
    Tensor<double> eia(no,nv);
    for (auto i=0; i<no;i++){
        for (auto a=0;a<nv;a++){
            eia(i,a)= Fii(i,i)-Faa(a,a);
        }
    }
    return eia;
}

Tensor<double> Dijab(const Tensor<double>& Fii,const Tensor<double>& Faa){
    auto no = Fii.extent(0);
    auto nv = Faa.extent(0);
    Tensor<double> eijab(no,no,nv,nv);
    for(auto i=0;i<no; i++){
        for(auto j=0;j<no;j++){
            for(auto a=0;a<nv;a++){
                for(auto b=0;b<nv;b++){
                    eijab(i,j,a,b) += Fii(i,i)+Fii(j,j)-Faa(a,a)-Faa(b,b);
                }
            }
        }
    }
    return eijab;
}

Tensor<double> space2spin(const Tensor<double>& Maa,
                              const Tensor<double>& Mbb ,const Tensor<double>& Mab){
    const auto n = Maa.extent(0);
    const auto n2 = 2*n;
    Tensor<double> soints (n2,n2,n2,n2);
    for(auto i=0;i<n2;i++){
        for(auto j=0;j<n2;j++){
            for(auto k=0;k<n2;k++){
                for(auto l=0;l<n2;l++){
                    if(i%2==0 && j%2==0 && k%2==0 && l%2==0){
                        soints(i,j,k,l) = Maa(i/2,k/2,j/2,l/2)   // <ij||kl>
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

Tensor<double> copy_fock(const Matrix& F1,const Matrix& F2,  int n1 , int n2){
    Tensor<double> Fout(n2-n1,n2-n1);
    for(auto i=0;i<n2-n1;i++){
        if(n1%2==0){
            Fout(i,i) = (i%2==0)*F1(n1/2+i/2,n1/2+i/2) + (i%2==1)*F2(n1/2+i/2,n1/2+i/2);
        }
        else if(n1%2==1){
            Fout(i,i) = (i%2==0)*F1(n1/2+1+1+i/2,n1/2+1+i/2) + (i%2==1)*F2(n1/2+1+i/2,n1/2+1+i/2);
        }
    }
    return Fout;
}


INTEGRALS make_ints(const Tensor<double>& eri,const scf_results& SCF){
    INTEGRALS integrals;
    Tensor<double> soints;
    if(SCF.is_rhf){
        Tensor<double> moints = aotomo(eri,SCF.C,SCF.C,SCF.C,SCF.C);
        soints = space2spin(moints,moints,moints);
        integrals.F = copy_fock(SCF.F , SCF.F,0,2*SCF.nbasis);
        integrals.Fii = copy_fock(SCF.F , SCF.F,0,SCF.no);
        integrals.Faa = copy_fock(SCF.F , SCF.F,SCF.no,2*SCF.nbasis);
        Tensor<double> Fia (SCF.no,SCF.nv);
        for(auto i=0;i<SCF.no;i++){
            for(auto a=0;a<SCF.nv;a++){
                Fia(i,a) = integrals.F(i,SCF.no+a);
            }
        }
        integrals.Fia = Fia;
    }
    else if(SCF.is_rhf == 0){
        Tensor<double> Malpha_alpha = aotomo(eri,SCF.Calpha,SCF.Calpha,SCF.Calpha,SCF.Calpha);
        Tensor<double> Mbeta_beta   = aotomo(eri,SCF.Cbeta,SCF.Cbeta,SCF.Cbeta,SCF.Cbeta);
        Tensor<double> Malpha_beta  = aotomo(eri,SCF.Calpha,SCF.Calpha,SCF.Cbeta,SCF.Cbeta);
        soints = space2spin(Malpha_alpha,Mbeta_beta,Malpha_beta);
        integrals.F = copy_fock(SCF.Falpha,SCF.Fbeta,0,2*SCF.nbasis);
        integrals.Fii = copy_fock(SCF.Falpha,SCF.Fbeta,0,SCF.no);
        Tensor<double> Faa(SCF.nv,SCF.nv);
        for(auto i=0;i<SCF.nv;i++){
            Faa(i,i) = SCF.F(SCF.no+i,SCF.no+i);
        }
        integrals.Faa = Faa;
        Tensor<double> Fia (SCF.no,SCF.nv);
        for(auto i=0;i<SCF.no;i++){
            for(auto a=0;a<SCF.nv;a++){
                Fia(i,a) = integrals.F(i,SCF.no+a);
            }
        }
        integrals.Fia = Fia;
    }
    integrals.vvvv = get_int(soints,SCF.no,SCF.nv,"vvvv");
    integrals.vvvo = get_int(soints,SCF.no,SCF.nv,"vvvo");
    integrals.oovv = get_int(soints,SCF.no,SCF.nv,"oovv");
    integrals.ooov = get_int(soints,SCF.no,SCF.nv,"ooov");
    integrals.oooo = get_int(soints,SCF.no,SCF.nv,"oooo");
    integrals.ovvo = get_int(soints,SCF.no,SCF.nv,"ovvo");
    integrals.ovov = get_int(soints,SCF.no,SCF.nv,"ovov");
    return integrals;
}

