//
// Created by Kshitij Surjuse on 9/19/22.
//
//Libint

#include <iostream>
#include <string>
#include <vector>
#include <libint2.hpp>
#if !LIBINT2_CONSTEXPR_STATICS
#  include <libint2/statics_definition.h>
#endif

using std::string;
using std::vector;
using libint2::Atom;

const vector<string> spliter(const string& s, const char& c)
{
    string buff{""};
    vector<string> v;
    for(auto n:s)
    {
        if(n != c) buff+=n; else
        if(n == c && buff != "") { v.push_back(buff); buff = ""; }
    }
    if(buff != "") v.push_back(buff);
    return v;
}

int atomic_no_sym(string sym){
    int Z=0;
    if (sym=="H"){Z=1;}
    if (sym=="He"){Z=2;}
    if (sym=="Li"){Z=3;}
    if (sym=="Be"){Z=4;}
    if (sym=="B"){Z=5;}
    if (sym=="C"){Z=6;}
    if (sym=="N"){Z=7;}
    if (sym=="O"){Z=8;}
    if (sym=="F"){Z=9;}
    if (sym=="Ne"){Z=10;}
    if (sym=="Na"){Z=11;}
    if (sym=="Mg"){Z=12;}
    if (sym=="Al"){Z=13;}
    if (sym=="Si"){Z=14;}
    if (sym=="P"){Z=15;}
    if (sym=="S"){Z=16;}
    if (sym=="Cl"){Z=17;}
    if (sym=="Ar"){Z=18;}
    if (sym=="K"){Z=19;}
    if (sym=="Ca"){Z=20;}

    return Z;
}


Atom line_read(vector<string> line){
    int i =0;
    Atom atom;
    atom.atomic_number = atomic_no_sym(line[0]);
    atom.x = 1.8897259885789*std::stod(line[1]);
    atom.y = 1.8897259885789*std::stod(line[2]);
    atom.z = 1.8897259885789*std::stod(line[3]);
    return atom;
}


vector <Atom>  read_geometry(string filename){
    std::ifstream xyz (filename);
    int number_of_atoms =0;
    vector<Atom> atoms;
    Atom one_atom;
    if (xyz.is_open()){
        int m=0;
        Atom atom;
        string myline;
        while(xyz){
            std::getline (xyz, myline);
            if(m==0){
                number_of_atoms = std::stof(myline);
            }
            else if(m>1 && m<number_of_atoms+2){
                vector<string> line {spliter(myline, ' ')};
                atom = line_read(line);
                atoms.push_back(atom);
            }
            m++;
        }
    }
    return atoms;
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


inp_params read_input(string inp_file){
    std::ifstream inp (inp_file);
    inp_params inpParams;
    if(inp.is_open()){
        string line;
        while(inp){
            std::getline (inp,line);
            vector<string> vals {spliter(line, ' ')};
            if(vals[0] == "method"){
                inpParams.method = vals[1];
            }
            if(vals[0] == "basis"){
                inpParams.basis = vals[1];
            }
            if(vals[0] == "scf_convergence"){
                inpParams.scf_convergence = std::stod(vals[1]);
            }
            if(vals[0] == "do_diis"){
                inpParams.do_diis = std::stoi(vals[1]);
            }
            if(vals[0] == "spin_mult"){
                inpParams.spin_mult = std::stoi(vals[1]);
            }
            if(vals[0] == "charge"){
                inpParams.charge = std::stoi(vals[1]);
            }
            if(vals[0]=="scf"){
                inpParams.scf = vals[1];
            }
        }
    }
    return inpParams;
}


double enuc_calc(vector<Atom> atoms){
    double enuc = 0.0;
    for(int i=0; i < atoms.size();i++){
        for(int j=0;j<i;j++){
            auto e2 = (1.0*atoms[i].atomic_number) * (1.0*atoms[j].atomic_number);
            auto rij = sqrt(pow(atoms[i].x - atoms[j].x , 2) +
                            pow(atoms[i].y-atoms[j].y , 2) +
                            pow(atoms[i].z-atoms[j].z , 2)) ;
            enuc+= e2/rij;
        }
    }
    return enuc;
}

