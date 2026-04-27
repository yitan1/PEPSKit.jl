using PEPSKit
using TensorKit
using Random

# ising model
g = 3.1
e = -1.6417 * 2
mˣ = 0.91

D = 2
chi = 16
gradtol = 1.0e-3

# initialize states
ham = transverse_field_ising(InfiniteSquare(); g)
Random.seed!(2928528935)
peps0 = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
# env0 = CTMRGEnv(peps0, ComplexSpace(chi))
env0 = PEPSKit.init_env(peps0)
env1, = leading_boundary(env0, peps0, SimultaneousCTMRG(; maxiter=50, tol=1e-6, trunc=truncerror(atol=1e-10) & truncrank(16)))

# find fixedpoint
gs, env_gs, E, = fixedpoint(ham, peps0, env1; tol = gradtol)

cost_function(gs, env_gs, ham)
network_value(gs, env_gs)

# compute basis
Bs = PEPSKit.tangent_basis(env_gs, gs)
ham_shift = PEPSKit.shift_hamiltonian(gs, env_gs, ham)


B11 = Bs[1, 1]
# siz = size(Bs)
# Nij = collect(dims(Bs[i, j])[2] for i in 1:siz[1], j in 1:siz[2])
# N = sum(Nij)

N = dims(B11)[2]
effH = zeros(ComplexF64, N, N)
effN = zeros(ComplexF64, N, N)
B = TensorMap(Array(reshape(B11[:, 1], dims(gs[1]))), space(gs[1]))
es = InfinitePEPS([B;;])

# qp_peps0 = InfiniteQPPEPS(gs, B)
qp_env0 = PEPSKit.qp_CTMRGEnv(env_gs);
res = PEPSKit.run_es(gs, es, qp_env0, ham_shift; pre_ctm_alg = pre_qp_ctm_alg)

trunc = truncerror(atol = 1e-10) & truncrank(16)
it = 30
pre_qp_ctm_alg = PEPSKit.SequentialQPCTMRG(; miniter = it, maxiter=it, tol=1e-6, trunc=trunc)

qp_env1, _ = leading_boundary(qp_env0, qp_peps0, pre_qp_ctm_alg; conv_level=1)
qp_ctm_alg = PEPSKit.SequentialQPCTMRG(; maxiter=4, tol=1e-6, trunc=trunc)
ctm_alg = PEPSKit.SequentialCTMRG(; maxiter=4, tol=1e-6, trunc=trunc)


exci_alg = PEPSKit.InfiniteQPExcitation(qp_ctm_alg)
PEPSKit.excitation(ham, exci_alg, A, env0; pre_ctm_alg=qp_ctm_alg)






# Hubbard model
t, U = 1, 6
ham = hubbard_model(ComplexF64, Trivial, Trivial, InfiniteSquare(2, 2); t, U, mu=U / 2)

Vspace = Vect[fℤ₂](0 => 2, 1 => 2)
Pspace = Vect[fℤ₂](0 => 2, 1 => 2)
A0 = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
Espace = Vect[fℤ₂](0 => 8, 1 => 8)
env0 = CTMRGEnv(randn, ComplexF64, A0, Espace);

# compute ground state
boundary_alg = (; tol = 1.0e-8, alg = :simultaneous, verbosity = 2, trunc = (; alg = :fixedspace))
gradient_alg = (; tol = 1.0e-6, maxiter = 10, alg = :eigsolver, iterscheme = :diffgauge)
optimizer_alg = (; tol = 1.0e-4, alg = :lbfgs, verbosity = 3, maxiter = 25, ls_maxiter = 2, ls_maxfg = 2)
reuse_env = true

env1, = leading_boundary(env0, A0; boundary_alg...)
gs, env_gs, E, info = fixedpoint(
    ham, A0, env1; boundary_alg, gradient_alg, optimizer_alg, reuse_env
)
