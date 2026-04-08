using PEPSKit
using PEPSKit: NestedTensor
using TensorKit

# qp_PEPS and qp_Env
Vspace = Vect[fℤ₂](0 => 2, 1 => 2)
Pspace = Vect[fℤ₂](0 => 2, 1 => 2)
A = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
B = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
qp_A = PEPSKit.InfiniteQPPEPS(A, B);

Espace = Vect[fℤ₂](0 => 8, 1 => 8)
env0 = CTMRGEnv(randn, ComplexF64, A, Espace);
qp_env0 = PEPSKit.qp_CTMRGEnv(env0);

AN = InfiniteSquareNetwork(A);
qp_AN = InfiniteSquareNetwork(qp_A); 

# test ctmrg_iteration
trunc = truncerror(atol = 1e-10) & truncrank(16)
qp_ctm_alg = PEPSKit.SequentialQPCTMRG(; maxiter=100, tol=1e-6, trunc=trunc)
# qp_env1, info = PEPSKit.ctmrg_iteration(qp_AN, qp_env0, qp_ctm_alg);
qp_env_f, info = leading_boundary(qp_env0, qp_AN, qp_ctm_alg; conv_level=1);

# Hubbard model Hamiltonian at half-filling
t, U = 1, 6
ham = hubbard_model(ComplexF64, Trivial, Trivial, InfiniteSquare(2, 2); t, U, mu=U / 2)

qp_env = qp_env_f;
# gs_e = expectation_value(A, ham, env)
gs_e = PEPSKit.qp_expectation_value(qp_A, ham, qp_env; E_order=1, N_order=1)
es_e = PEPSKit.qp_expectation_value(qp_A, ham, qp_env; E_order=4, N_order=1)
es_e = PEPSKit.qp_energy(qp_A, qp_env, ham)



