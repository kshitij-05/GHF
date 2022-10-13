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
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> DiagonalMatrix;


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

struct scf_results{
    bool is_rhf=1;
    int nelec, nbasis,no,nv , nalpha,nbeta;
    Matrix C,F,Calpha,Cbeta,Falpha,Fbeta;
    double scf_energy;
};
struct mp2_results{
    Tensor<double> t2_mp2;
    double mp2_energy;
};

struct INTEGRALS{
    Tensor<double> vvvv;
    Tensor<double> vvvo;
    Tensor<double> ovvv;
    Tensor<double> vovv;
    Tensor<double> oovv;
    Tensor<double> ovvo;
    Tensor<double> ovov;
    Tensor<double> ooov;
    Tensor<double> oovo;
    Tensor<double> ovoo;
    Tensor<double> oooo;
    Tensor<double> F;
    Tensor<double> Fii;
    Tensor<double> Faa;
    Tensor<double> Fia;
    Tensor<double> eia;
    Tensor<double> eijab;
};

INTEGRALS make_ints(const Tensor<double>& eri,const scf_results& SCF);
Tensor<double> Dia(const Tensor<double>& Fii,const Tensor<double>& Faa);
Tensor<double> Dijab(const Tensor<double>& Fii,const Tensor<double>& Faa);

struct intermidiates{
    Tensor <double> Fae, Fmi, Fme, Wmnij, Wmbej,Wabef;
};

Tensor<double> tau2(double X, const Tensor<double>& ts , const Tensor<double>& td){
    auto no = ts.extent(0);
    auto nv = ts.extent(1);
    Tensor<double> TAU2(no,no,nv,nv);
    for(auto i=0;i<no;i++){
        for(auto j=0;j<no;j++){
            for(auto a=0;a<nv;a++){
                for(auto b=0;b<nv;b++){
                    TAU2(i,j,a,b) +=X*ts(i,a)*ts(j,b) - X*ts(i,b)*ts(j,a)+td(i,j,a,b);
                }
            }
        }
    }
    return TAU2;
}

Tensor <double> t1_sqr(const Tensor<double>& ts) {
    auto no = ts.extent(0);
    auto nv = ts.extent(1);
    Tensor<double> TAU(no, nv, no, nv);
    for (auto i = 0; i < no; i++) {
        for (auto j = 0; j < no; j++) {
            for (auto a = 0; a < nv; a++) {
                for (auto b = 0; b < nv; b++) {
                    TAU(i, a, j, b) +=  ts(i, a) * ts(j, b);
                }
            }
        }
    }
    return TAU;
}



intermidiates update_imds(const Tensor<double>& ts,const Tensor<double>& td,const INTEGRALS& integrals){
    auto no = ts.extent(0);
    auto nv = ts.extent(1);
    intermidiates imds;
    const char m='m';
    const char n='n';
    const char e='e';
    const char f='f';
    const char a='a';
    const char b='b';
    const char i='i';
    const char j='j';

    Tensor<double> Fae(nv,nv);
    contract(-0.5,integrals.Fia,{m,e}, ts,{m,a},1.0,Fae,{a,e});
    contract(1.0,ts,{m,f},integrals.ovvv,{m,a,f,e},1.0,Fae,{a,e});
    contract(-0.5, tau2(0.5,ts,td),{m,n,a,f},integrals.oovv,{m,n,e,f},1.0,Fae,{a,e});
    for(auto c=0;c<nv;c++){
        for(auto d=0;d<nv;d++){
            Fae(c,d) += (1-(c==d))*integrals.Faa(c,d);
        }
    }
    imds.Fae = Fae;
    Tensor<double> Fmi(no,no);
    contract(0.5,ts,{i,e},integrals.Fia,{m,e},1.0,Fmi,{m,i});
    contract(1.0,ts,{n,e},integrals.ooov,{m,n,i,e},1.0,Fmi,{m,i});
    contract(0.5, tau2(0.5,ts,td),{i,n,e,f},integrals.oovv,{m,n,e,f},1.0,Fmi,{m,i});
    for(auto k=0;k<no;k++){
        for(auto l=0;l<no;l++){
            Fmi(k,l) += (1-(k==l))*integrals.Fii(k,l);
        }
    }
    imds.Fmi = Fmi;
    Tensor<double> Fme(no,nv);
    contract(1.0,ts,{n,f},integrals.oovv,{m,n,e,f},1.0,Fme,{m,e});
    Fme += integrals.Fia;
    imds.Fme = Fme;
    Tensor<double> Wmnij(no,no,no,no);
    contract(1.0,ts,{j,e},integrals.ooov,{m,n,i,e},1.0,Wmnij,{m,n,i,j});
    contract(-1.0,ts,{i,e},integrals.ooov,{m,n,j,e},1.0,Wmnij,{m,n,i,j});
    contract(0.25, tau2(1.0,ts,td),{i,j,e,f}, integrals.oovv,{m,n,e,f},1.0,Wmnij,{m,n,i,j});
    Wmnij += integrals.oooo;
    imds.Wmnij = Wmnij;
    Tensor<double> Wmbej(no,nv,nv,no);
    Wmbej += integrals.ovvo;
    contract(1.0,ts,{j,f},integrals.ovvv,{m,b,e,f},1.0,Wmbej,{m,b,e,j});
    contract(-1.0,ts,{n,b},integrals.oovo,{m,n,e,j},1.0,Wmbej,{m,b,e,j});
    contract(-0.5,td,{j,n,f,b},integrals.oovv,{m,n,e,f},1.0,Wmbej, {m,b,e,j});
    contract(-1.0, t1_sqr(ts),{j,f,n,b},integrals.oovv,{m,n,e,f},1.0,Wmbej, {m,b,e,j});
    imds.Wmbej = Wmbej;

    Tensor<double> Wabef(nv,nv,nv,nv);
    Wabef += integrals.vvvv;
    contract(-1.0,ts,{m,b},integrals.vovv,{a,m,e,f},1.0,Wabef,{a,b,e,f});
    contract(1.0,ts,{m,a},integrals.vovv,{b,m,e,f},1.0,Wabef,{a,b,e,f});
    contract(0.25, tau2(1.0,ts,td),{m,n,a,b},integrals.oovv,{m,n,e,f},1.0,Wabef,{a,b,e,f});

    imds.Wabef = Wabef;

    return imds;
}

Tensor<double> make_t1(const Tensor<double>& ts,const Tensor<double>& td,
                       const INTEGRALS& integrals,const intermidiates& imds,const Tensor<double>& eia){
    auto no = ts.extent(0);
    auto nv = ts.extent(1);
    const char m='m';
    const char n='n';
    const char e='e';
    const char f='f';
    const char a='a';
    const char i='i';
    Tensor<double> tsnew(no,nv);
    tsnew += integrals.Fia;    //+Fia
    contract(1.0,ts,{i,e},imds.Fae,{a,e},1.0,tsnew,{i,a});                  //-T(i,e)*F(a,e)
    contract(-1.0,ts,{m,a},imds.Fmi,{m,i},1.0,tsnew,{i,a});                 //+T(a,m)*F(m,e)
    contract(1.0,td,{i,m,a,e},imds.Fme,{m,e},1.0,tsnew,{i,a});              //+T(i,m,a,e)*F(m,e)
    contract(-1.0,ts,{n,f},integrals.ovov,{n,a,i,f},1.0,tsnew,{i,a});       //-T(n,f)*<na||if>
    contract(-0.5,td,{i,m,e,f},integrals.ovvv,{m,a,e,f},1.0,tsnew,{i,a});   //-1/2T(i,m,e,f)*<ma||ef>
    contract(-0.5,td,{m,n,a,e},integrals.oovo,{n,m,e,i},1.0,tsnew,{i,a});   //-1/2T(m,n,a,e)*<nm||ei>
    Tensor<double> tsout (no,nv);
    for(auto k=0;k<no;k++){
        for(auto d=0;d<nv;d++){
            tsout(k,d) += tsnew(k,d)/eia(k,d);
        }
    }
    return tsout;
}

Tensor<double> make_t2(const Tensor<double>& ts, const Tensor<double>& td, const intermidiates& imds,
                       const INTEGRALS& integrals,const Tensor<double>& eijab){
    auto no = ts.extent(0);
    auto nv = ts.extent(1);
    const char m='m';
    const char n='n';
    const char e='e';
    const char f='f';
    const char a='a';
    const char b='b';
    const char i='i';
    const char j='j';
    Tensor<double> tdnew(no,no,nv,nv);

    tdnew += integrals.oovv;  //+<ij||ab>

    contract(1.0,td,{i,j,a,e},imds.Fae,{b,e},1.0,tdnew,{i,j,a,b});
    contract(-1.0,td,{i,j,b,e},imds.Fae,{a,e},1.0,tdnew,{i,j,a,b}); // P(ab) T(ijae) Fbe

    Tensor<double> temp1(nv,nv);
    contract(1.0,ts,{m,b},imds.Fme,{m,e},1.0,temp1,{b,e});
    contract(-0.5,td,{i,j,a,e},temp1,{b,e},1.0,tdnew,{i,j,a,b});
    temp1 -=temp1;
    contract(1.0,ts,{m,a},imds.Fme,{m,e},1.0,temp1,{a,e});
    contract(0.5,td,{i,j,b,e},temp1,{a,e},1.0,tdnew,{i,j,a,b});  //  -1/2 P(ab) T(mb) Fme

    contract(-1.0,td,{i,m,a,b},imds.Fmi,{m,j},1.0, tdnew,{i,j,a,b});
    contract(1.0,td,{j,m,a,b},imds.Fmi,{m,i},1.0,tdnew,{i,j,a,b});  //-P(ij) * T(imab) Fmj

    Tensor<double> temp2 (no,no);
    contract(1.0,ts,{j,e},imds.Fme,{m,e},1.0,temp2,{j,m});
    contract(-0.5,td,{i,m,a,b},temp2,{j,m},1.0,tdnew,{i,j,a,b});
    temp2 -=temp2;
    contract(1.0,ts,{i,e},imds.Fme,{m,e},1.0,temp2,{i,m});
    contract(0.5,td,{j,m,a,b},temp2,{i,m},1.0,tdnew,{i,j,a,b});   //-P(ij) * Tji * Fme

    contract(0.5, tau2(1.0,ts,td),{m,n,a,b},imds.Wmnij,{m,n,i,j},1.0,tdnew,{i,j,a,b});     // 1/2  Tau(mnab)  Wmnij

    contract(1.0,ts,{i,e},integrals.vvvo,{a,b,e,j},1.0,tdnew,{i,j,a,b});
    contract(-1.0,ts,{j,e},integrals.vvvo,{a,b,e,i},1.0,tdnew,{i,j,a,b});  //P(ij)  Tie  <ab||ej>

    contract(-1.0,ts,{m,a},integrals.ovoo,{m,b,i,j},1.0,tdnew,{i,j,a,b});
    contract(1.0,ts,{m,b},integrals.ovoo,{m,a,i,j},1.0,tdnew,{i,j,a,b});  // P(ab) Tma  <mb||ij>

    contract(1.0,td,{i,m,a,e},imds.Wmbej,{m,b,e,j},1.0,tdnew,{i,j,a,b});
    contract(-1.0,td,{j,m,a,e},imds.Wmbej,{m,b,e,i},1.0,tdnew,{i,j,a,b});
    contract(-1.0,td,{i,m,b,e},imds.Wmbej,{m,a,e,j},1.0,tdnew,{i,j,a,b});
    contract(1.0,td,{j,m,b,e},imds.Wmbej,{m,a,e,i},1.0,tdnew,{i,j,a,b});  //P(ij)P(ab)  Timae  Wmbej

    contract(1.0, t1_sqr(ts),{i,e,m,a},integrals.ovov,{m,b,j,e},1.0,tdnew,{i,j,a,b});
    contract(-1.0, t1_sqr(ts),{j,e,m,a},integrals.ovov,{m,b,i,e},1.0,tdnew,{i,j,a,b});
    contract(-1.0, t1_sqr(ts),{i,e,m,b},integrals.ovov,{m,a,j,e},1.0,tdnew,{i,j,a,b});
    contract(1.0, t1_sqr(ts),{j,e,m,b},integrals.ovov,{m,a,i,e},1.0,tdnew,{i,j,a,b});    //P(ij)P(ab)  Tie Tma <mb||ej>

    contract(0.5, tau2(1.0,ts,td),{i,j,e,f} ,imds.Wabef,{a,b,e,f},1.0,tdnew,{i,j,a,b}); //1/2 Tau(ijef)  Wabef

    Tensor<double> tdout (no,no,nv,nv);
    for(auto k=0;k<no;k++){
        for(auto l=0;l<no;l++){
            for(auto c=0;c<nv;c++){
                for(auto d=0;d<nv;d++){
                    tdout(k,l,c,d) = tdnew(k,l,c,d)/eijab(k,l,c,d);
                }
            }
        }
    }
    return tdout;
}

double ccsd_energy(const Tensor<double>& ts, const Tensor<double>& td, const Tensor<double>& fia,const Tensor<double>& oovv){
    double ecc= 0.0;
    auto no = ts.extent(0);
    auto nv = ts.extent(1);
    for(auto i=0;i<no;i++){
        for(auto a=0;a<nv;a++){
            ecc+=fia(i,a)*ts(i,a);
            for(auto j=0;j<no;j++){
                for(auto b=0;b<nv;b++){
                    ecc+=0.25*oovv(i,j,a,b)*td(i,j,a,b) +
                            0.5*oovv(i,j,a,b)*ts(i,a)*ts(j,b);
                }
            }
        }
    }
    return ecc;
}



double ccsd(const inp_params&  inpParams, const Tensor<double>& eri, const scf_results& SCF,const mp2_results&  MP2){
    double ecc = 0.0;
    int no = SCF.no;
    int nv = SCF.nv;
    INTEGRALS integrals = make_ints(eri,SCF);
    Tensor<double> eia = Dia(integrals.Fii,integrals.Faa);
    Tensor<double> eijab = Dijab(integrals.Fii,integrals.Faa);
    Tensor<double> ts(no,nv);
    Tensor<double> td = MP2.t2_mp2;
    for (int i=0;i<200;i++){
        if(i==0){cout << "iter      " << "Ecc (a.u.)\t\t"<< "Delta_E :\t"<< endl;}
        double new_ecc = ccsd_energy(ts,td,integrals.Fia,integrals.oovv);
        double del_ecc = abs(new_ecc - ecc);
        ecc = new_ecc;
        cout << i+1 << "\t" << ecc << "\t" << del_ecc << endl;
        if(del_ecc < inpParams.scf_convergence){
            cout << endl;
            break;
        }
        intermidiates imds = update_imds(ts,td,integrals);
        if(inpParams.singles==1){
            ts = make_t1(ts,td,integrals,imds,eia);
        }
        td = make_t2(ts,td,imds,integrals,eijab);
    }
    return ecc;
}