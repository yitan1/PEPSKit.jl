using LinearAlgebra

using JLD2
using MatrixAlgebraKit
using PEPSKit
using TensorKit
using Zygote: withgradient

# Step 3: load the cached basis and build a small effective excitation problem.

cache_dir = joinpath(@__DIR__, "cache")
basis_file = joinpath(cache_dir, "basis.jld2")
isfile(basis_file) || error("Missing cache file: run qp_basis.jl first")

data = load(basis_file)
ham_shift = data["ham_shift"]
gs = data["gs"]
env_gs = data["env_gs"]
Bs = data["Bs"]

χ = 16
num_basis = 4      # set to `N` below to build the full matrix
pre_rg_steps = 30
ad_rg_steps = 4

trunc = truncerror(; atol=1.0e-10) & truncrank(χ)

Bbasis = Bs[1, 1]
N = dims(Bbasis)[2]
M = min(num_basis, N)

effH = zeros(ComplexF64, M, M)
effN = zeros(ComplexF64, M, M)
qp_es = zeros(Float64, M)
qp_nrms = zeros(ComplexF64, M)

qp_ctm_alg = PEPSKit.SequentialQPCTMRG(;
    miniter=pre_rg_steps,
    maxiter=pre_rg_steps,
    tol=1.0e-6,
    trunc,
)
qp_ad_alg = PEPSKit.SequentialQPCTMRG(;
    miniter=ad_rg_steps,
    maxiter=ad_rg_steps,
    tol=1.0e-6,
    trunc,
)

for i in 1:M
    println("\n===== basis $i / $M =====")

    B_tensor = TensorMap(Array(reshape(Bbasis[:, i], dims(gs[1]))), space(gs[1]))
    B = InfinitePEPS([B_tensor;;])
    qp_peps = PEPSKit.InfiniteQPPEPS(gs, B)

    println("Running QP CTMRG pre-steps")
    qp_env0 = PEPSKit.qp_CTMRGEnv(env_gs)
    qp_env, = leading_boundary(qp_env0, qp_peps, qp_ctm_alg; conv_level=4)

    qp_e = PEPSKit.qp_energy(qp_peps, qp_env, ham_shift)
    qp_nrm = PEPSKit.qp_norm(qp_peps, qp_env)
    qp_es[i] = real(qp_e)
    qp_nrms[i] = qp_nrm

    println("Computing gH with AD through a few QP RG steps")
    _, gsH = withgradient(B) do x
        qp_peps_x = PEPSKit.InfiniteQPPEPS(gs, x)
        qp_env_x, = leading_boundary(qp_env, qp_peps_x, qp_ad_alg; conv_level=4)
        PEPSKit.qp_energy(qp_peps_x, qp_env_x, ham_shift)
    end
    gH = only(gsH)

    println("Computing gN directly from the norm environment")
    env_bra = PEPSKit._contract_env_bra((1, 1), PEPSKit.bra(qp_peps), qp_env)
    nAA = PEPSKit._contract_env_bra_ket(env_bra[1], qp_peps[1, 1][1])
    gN_tensor = env_bra[3] / nAA
    gN = InfinitePEPS([gN_tensor;;])

    projH = vec(convert(Array, Bbasis' * PEPSKit.to_vec(gH[1])))
    projN = vec(convert(Array, Bbasis' * PEPSKit.to_vec(gN[1])))
    effH[:, i] .= projH[1:M]
    effN[:, i] .= projN[1:M]

    @show qp_e qp_nrm
    @show effH[i, i] / effN[i, i]
    @show qp_e / qp_nrm
end

H = (effH + effH') / 2
Nmat = (effN + effN') / 2

norm_eig = eigen(Hermitian(Nmat))
keep = findall(>(maximum(real(norm_eig.values)) * 1.0e-10), real(norm_eig.values))
P = norm_eig.vectors[:, keep]

spectrum = eigen(Hermitian(P' * H * P), Hermitian(P' * Nmat * P))

out = joinpath(cache_dir, "spectrum.jld2")
jldsave(out; effH, effN, H, Nmat, spectrum, qp_es, qp_nrms)

println("\n===== excitation spectrum =====")
@show real(spectrum.values)
@show qp_es ./ real.(qp_nrms)
println("Saved spectrum cache:")
println(out)
