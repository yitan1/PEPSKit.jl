using LinearAlgebra

using JLD2
using MatrixAlgebraKit
using PEPSKit
using TensorKit
using ChainRulesCore: ignore_derivatives
using Zygote: withgradient

# Step 3: load the cached basis and build a small effective excitation problem.

cache_dir = joinpath(@__DIR__, "cache")
basis_file = get(ENV, "BASIS_FILE", joinpath(cache_dir, "basis.jld2"))
isfile(basis_file) || error("Missing cache file: run qp_basis.jl first")

data = load(basis_file)
ham_shift = data["ham_shift"]
gs = data["gs"]
env_gs = data["env_gs"]
Bs = data["Bs"]

χ = 16
num_basis_env = get(ENV, "NUM_BASIS", "")
pre_rg_steps = parse(Int, get(ENV, "PRE_RG_STEPS", "30"))
ad_rg_steps = parse(Int, get(ENV, "AD_RG_STEPS", "4"))

trunc = truncerror(; atol=1.0e-10) & truncrank(χ)

Bbasis = Bs[1, 1]
N = dims(Bbasis)[2]
M = isempty(num_basis_env) ? N : min(parse(Int, num_basis_env), N)

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

    local B = PEPSKit._basis_column_peps(Bbasis, i, gs)
    qp_peps = PEPSKit.InfiniteQPPEPS(gs, B)

    println("Running QP CTMRG pre-steps")
    qp_env0 = PEPSKit.qp_CTMRGEnv(env_gs)
    qp_env, = leading_boundary(qp_env0, qp_peps, qp_ctm_alg; conv_level=4)

    println("Computing gH with AD through a few QP RG steps")
    qp_env_ad_ref = Ref{Any}()
    _, gsH = withgradient(B) do x
        qp_peps_x = PEPSKit.InfiniteQPPEPS(gs, x)
        qp_env_x, = leading_boundary(qp_env, qp_peps_x, qp_ad_alg; conv_level=4)
        ignore_derivatives() do
            qp_env_ad_ref[] = qp_env_x
        end
        PEPSKit.qp_energy(qp_peps_x, qp_env_x, ham_shift)
    end
    gH = only(gsH)

    qp_env_ad = qp_env_ad_ref[]
    qp_e = PEPSKit.qp_energy(qp_peps, qp_env_ad, ham_shift)
    qp_nrm = PEPSKit.qp_norm(qp_peps, qp_env_ad)
    qp_es[i] = real(qp_e)
    qp_nrms[i] = qp_nrm

    println("Computing gN directly from the norm environment")
    env_ket = PEPSKit._contract_env_ket((1, 1), PEPSKit.ket(qp_peps), qp_env_ad)
    qp_bra = PEPSKit.bra(qp_peps)
    nAA = PEPSKit._contract_env_ket_bra(env_ket[1], qp_bra[1, 1][1])
    gN_tensor = env_ket[2] / nAA

    projH = vec(convert(Array, Bbasis' * PEPSKit.to_vec(gH[1])))
    projN = vec(convert(Array, Bbasis' * PEPSKit.to_vec(gN_tensor)))
    effH[:, i] .= projH[1:M] ./ 2
    effN[:, i] .= projN[1:M]

    @show qp_e qp_nrm
    @show effH[i, i] / effN[i, i]
    @show qp_e / qp_nrm
end

H = (effH + effH') / 2
Nmat = (effN + effN') / 2

norm_eig = eigen(Hermitian(Nmat))
norm_values = real(norm_eig.values)
norm_cutoff = maximum(norm_values) * 1.0e-10
keep = findall(>(norm_cutoff), norm_values)

println("\n===== effective matrix diagnostics =====")
@show norm(effH - effH') / norm(effH)
@show norm(effN - effN') / norm(effN)
@show diag(effH) ./ diag(effN)
@show qp_es ./ real.(qp_nrms)

H_asym = abs.(effH - effH')
N_asym = abs.(effN - effN')
H_pairs = [(i, j, H_asym[i, j], abs(effH[i, j]), abs(effH[j, i])) for i in 1:M for j in (i + 1):M]
N_pairs = [(i, j, N_asym[i, j], abs(effN[i, j]), abs(effN[j, i])) for i in 1:M for j in (i + 1):M]
sort!(H_pairs; by=x -> x[3], rev=true)
sort!(N_pairs; by=x -> x[3], rev=true)
if !isempty(H_pairs)
    @show H_pairs[1:min(length(H_pairs), 10)]
end
if !isempty(N_pairs)
    @show N_pairs[1:min(length(N_pairs), 10)]
end

@show norm_values
@show norm_cutoff
@show keep
@show length(keep)
if !isempty(keep)
    kept_norm_values = norm_values[keep]
    @show minimum(kept_norm_values)
    @show maximum(kept_norm_values)
    @show maximum(kept_norm_values) / minimum(kept_norm_values)
end

P = norm_eig.vectors[:, keep]

spectrum = eigen(Hermitian(P' * H * P), Hermitian(P' * Nmat * P))

out = get(ENV, "SPECTRUM_OUT", joinpath(cache_dir, "spectrum.jld2"))
jldsave(out; effH, effN, H, Nmat, spectrum, qp_es, qp_nrms)

println("\n===== excitation spectrum =====")
@show real(spectrum.values)
@show qp_es ./ real.(qp_nrms)
println("Saved spectrum cache:")
println(out)
