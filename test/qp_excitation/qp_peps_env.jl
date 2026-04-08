using TensorKit
using VectorInterface
using PEPSKit
using PEPSKit: NestedTensor

# qp_PEPS
Vspace = Vect[fℤ₂](0 => 2, 1 => 2)
Pspace = Vect[fℤ₂](0 => 2, 1 => 2)
A = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
B = InfinitePEPS(rand, ComplexF64, Pspace, Vspace; unitcell=(2, 2));
qp_A = PEPSKit.InfiniteQPPEPS(A, B);

bra = 1
ket = 2
r, c = 1, 1

@assert qp_A[r, c][1] == A[r, c]
@assert qp_A[r, c][2] == B[r, c]
@assert qp_A[r, c][3] == VectorInterface.zerovector(A[r, c])
@assert qp_A[r, c][4] == VectorInterface.zerovector(A[r, c])

qp_Ad = PEPSKit.bra(qp_A);
@assert qp_Ad[r, c][1] == A[r, c]
@assert qp_Ad[r, c][2] == VectorInterface.zerovector(A[r, c])
@assert qp_Ad[r, c][3] == B[r, c]
@assert qp_Ad[r, c][4] == VectorInterface.zerovector(A[r, c])

AN = InfiniteSquareNetwork(A); 
qp_AN = InfiniteSquareNetwork(qp_A); 
AN[r, c][bra] isa TensorMap
qp_AN[r, c][bra] isa NestedTensor
@assert qp_AN[r, c][bra][1] == A[r, c]
@assert qp_AN[r, c][bra][2] == B[r, c]
@assert qp_AN[r, c][bra][3] == VectorInterface.zerovector(A[r, c])
@assert qp_AN[r, c][bra][4] == VectorInterface.zerovector(A[r, c])

@assert qp_AN[r, c][ket][1] == A[r, c]
@assert qp_AN[r, c][ket][2] == VectorInterface.zerovector(A[r, c])
@assert qp_AN[r, c][ket][3] == B[r, c]
@assert qp_AN[r, c][ket][4] == VectorInterface.zerovector(A[r, c])


# qp_Env
Espace = Vect[fℤ₂](0 => 8, 1 => 8)
env0 = CTMRGEnv(randn, ComplexF64, A, Espace);
qp_env0 = PEPSKit.qp_CTMRGEnv(env0);

# 2. qp_peps -> gs_peps
# PEPSKit.gs_Network(qp_AN) == AN
# PEPSKit.gs_CTMRGEnv(qp_env0) == 

# interface with original code
# Networkvalue

gs_netval = PEPSKit.network_value(qp_AN, qp_env0)
qp_netval = PEPSKit.qp_network_value(qp_AN, qp_env0)
@assert isapprox(gs_netval, qp_netval[1])

# Norm


# Energy
t, U = 1, 6
ham = hubbard_model(ComplexF64, Trivial, Trivial, InfiniteSquare(2, 2); t, U, mu=U / 2)
e0 = PEPSKit.expectation_value(A, ham, env0)
qp_e0 = PEPSKit.qp_expectation_value(qp_A, ham, qp_env0; E_order=4) 


PEPSKit.gs_norm1x1(AN, env0)
PEPSKit.gs_norm1x1(qp_AN, qp_env0)

PEPSKit.gs_env_bra(env0, A)

PEPSKit.tangent_basis(env0, A)