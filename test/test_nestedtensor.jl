using PEPSKit
using PEPSKit: NestedTensor
using TensorKit

# qp_PEPS and qp_Env
Vspace = Vect[fℤ₂](0 => 2, 1 => 2)
Pspace = Vect[fℤ₂](0 => 2, 1 => 2)
A = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
B = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
qp_A = PEPSKit.qp_InfinitePEPS(A, B);

Espace = Vect[fℤ₂](0 => 8, 1 => 8)
env0 = CTMRGEnv(randn, ComplexF64, A, Espace);
qp_env0 = PEPSKit.qp_CTMRGEnv(env0);

AN = InfiniteSquareNetwork(A);
qp_AN = InfiniteSquareNetwork(qp_A);

# test qp -> gs
typeof(PEPSKit.gs_Network(qp_AN))
typeof(PEPSKit.gs_CTMRGEnv(qp_env0))

# test network value
using BenchmarkTools
@btime PEPSKit._contract_site((1, 1), AN, env0)
@btime PEPSKit._contract_site((1, 1), qp_AN, qp_env0)
PEPSKit._contract_corners((1, 1), qp_env0)
Ns = norm(qp_A, qp_env0)

PEPSKit.network_value(qp_AN, qp_env0)

# test ctmrg_iteration
trunc = truncbelow(1e-10) & truncdim(16)
qp_ctm_alg = SequentialQPCTMRG(; maxiter=3, tol=1e-6, trscheme=trunc)
qp_env1, info = PEPSKit.ctmrg_iteration(qp_AN, qp_env0, qp_ctm_alg);
qp_env_f, info = leading_boundary(qp_env0, qp_AN, qp_ctm_alg; conv_level=4);

# Hubbard model Hamiltonian at half-filling
t, U = 1, 6
ham = hubbard_model(ComplexF64, Trivial, Trivial, InfiniteSquare(2, 2); t, U, mu=U / 2)
expectation_value(A, ham, env0)
PEPSKit.qp_expectation_value(qp_A, ham, qp_env0)
PEPSKit.qp_energy(qp_A, qp_env0, ham)

exci_alg = PEPSKit.InfiniteQPExcitation(qp_ctm_alg)
PEPSKit.excitation(ham, exci_alg, A, env0; pre_ctm_alg=qp_ctm_alg)



function test_alloc()
    t1 = rand(ComplexF64, ℂ^2, ℂ^2)
    T = TO.tensoralloc(
        NestedTensor{TensorMap{ComplexF64}}, TO.tensorstructure(t1), Val(false)
    )
    @assert TO.tensorstructure(T) == TO.tensorstructure(t1)
    @assert isa(T, NestedTensor)
    @assert isa(T[1], TensorMap{ComplexF64})
end

function test_c()
    t1 = rand(ℂ^10 * ℂ^5 * ℂ^3)
    T1 = NestedTensor([t1, t1, t1, t1])

    t2 = rand((ℂ^3)', ℂ^5 * ℂ^8)
    T2 = NestedTensor([t2, t2, t2, t2])

    t3 = rand(ℂ^8, (ℂ^10))
    T3 = NestedTensor([t3, t3, t3, t3])

    @time @tensor C1[:] := t1[-1 2 1] * t2[1 2 3] * t3[3 -2]
    @time @tensor C2[:] := t1[-1 2 1] * T2[1 2 3] * T3[3 -2]

    @time @tensor res = T1[4 2 1] * T2[1 2 3] * T3[3 4]
    return println("res = ", res)
end

test_c()


# AD test for nestedtensor
function test_ad(t1, t2, t3)
    # t1 = rand(ℂ^10 * ℂ^5 * ℂ^3)
    T1 = NestedTensor([t1, t1, t1, t1])

    T2 = NestedTensor([t2, t2, t2, t2])

    T3 = NestedTensor([t3, t3, t3, t3])

    # @time @tensor C1[:] := t1[-1 2 1] * t2[1 2 3] * t3[3 -2]
    # @time @tensor C2[:] := t1[-1 2 1] * T2[1 2 3] * T3[3 -2]

    @tensor res = T1[4 2 1] * T2[1 2 3] * T3[3 4]
    # TensorKit.tensorcontract!
    # println("res = ", res)
    return res[1]
end

using Zygote
t1 = rand(ℂ^10 * ℂ^5 * ℂ^3)
t2 = rand((ℂ^3)', ℂ^5 * ℂ^8)
t3 = rand(ℂ^8, (ℂ^10))
test_ad(t1, t2, t3)
gradient(t -> test_ad(t, t2, t3), t1)

using TensorOperations
t1 = rand(ℂ^5, ℂ^5)
t2 = rand(ℂ^5, ℂ^5)
T1 = NestedTensor([t1, t1, t1, t1])
function f1(t1)
    T = NestedTensor([t1, t2, t2, t2])
    res = tensortrace(T, ((), ()), ((1,), (2,)), false)
    # @tensor res = T[1 1]
    # println("res = ", res)
    return tensorscalar(res)[1]
end
f1(t1)
withgradient(f1, t1)

