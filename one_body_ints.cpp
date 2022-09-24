//
// Created by Kshitij Surjuse on 9/20/22.
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

size_t nbasis(const std::vector<libint2::Shell>& shells) {
    size_t n = 0;
    for (const auto& shell: shells)
        n += shell.size();
    return n;
}
std::vector<size_t> map_shell_to_basis_function(const std::vector<libint2::Shell>& shells) {
    std::vector<size_t> result;
    result.reserve(shells.size());
    size_t n = 0;
    for (auto shell: shells) {
        result.push_back(n);
        n += shell.size();
    }
    return result;
}

using real_t = libint2::scalar_type;
Matrix compute_1body_ints(const std::vector<libint2::Shell>& shells,libint2::Operator obtype, const std::vector<Atom>& atoms)
{
    const auto n = nbasis(shells);
    Matrix result(n,n);
    // construct the overlap integrals engine
    Engine engine(obtype, max_nprim(shells), max_l(shells), 0);
    // nuclear attraction ints engine needs to know where the charges sit ...
    // the nuclei are charges in this case; in QM/MM there will also be classical charges
    if (obtype == Operator::nuclear) {
        std::vector<std::pair<real_t,std::array<real_t,3>>> q;
        for(const auto& atom : atoms) {
            q.push_back( {static_cast<real_t>(atom.atomic_number), {{atom.x, atom.y, atom.z}}} );
        }
        engine.set_params(q);
    }
    auto shell2bf = map_shell_to_basis_function(shells);
    // buf[0] points to the target shell set after every call  to engine.compute()
    const auto& buf = engine.results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over Hermitian operators: (1|2) = (2|1)
    for(auto s1=0; s1!=shells.size(); ++s1) {

        auto bf1 = shell2bf[s1]; // first basis function in this shell
        auto n1 = shells[s1].size();

        for(auto s2=0; s2<=s1; ++s2) {

            auto bf2 = shell2bf[s2];
            auto n2 = shells[s2].size();

            // compute shell pair
            engine.compute(shells[s1], shells[s2]);

            // "map" buffer to a const Eigen Matrix, and copy it to the corresponding blocks of the result
            Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
            result.block(bf1, bf2, n1, n2) = buf_mat;
            if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1} block, note the transpose!
                result.block(bf2, bf1, n2, n1) = buf_mat.transpose();

        }
    }

    return result;
}
