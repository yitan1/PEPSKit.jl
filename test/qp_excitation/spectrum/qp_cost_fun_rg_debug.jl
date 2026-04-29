using JLD2
using MatrixAlgebraKit
using PEPSKit
using TensorKit
using ChainRulesCore: ignore_derivatives
using Zygote: withgradient

cache_dir = joinpath(@__DIR__, "cache")
basis_file = get(ENV, "BASIS_FILE", joinpath(cache_dir, "basis.jld2"))
isfile(basis_file) || error("Missing cache file: run qp_basis.jl first")

data = load(basis_file)
ham_shift = data["ham_shift"]
gs = data["gs"]
env_gs = data["env_gs"]
Bs = data["Bs"]

basis_index = parse(Int, get(ENV, "BASIS_INDEX", "1"))
chi = 16
pre_rg_steps = 30
rg_steps = 4
run_cold_start = parse(Bool, get(ENV, "RUN_COLD_START", "true"))
run_warm_start = parse(Bool, get(ENV, "RUN_WARM_START", "true"))

Bbasis = Bs[1, 1]
N = dims(Bbasis)[2]
basis_index <= N || error("basis_index=$basis_index exceeds basis dimension $N")

function basis_peps(Bbasis, i, gs)
    B_tensor = TensorMap(Array(reshape(Bbasis[:, i], dims(gs[1]))), space(gs[1]))
    return InfinitePEPS([B_tensor;;])
end

function project_to_basis(Bbasis, tensor)
    return vec(convert(Array, Bbasis' * PEPSKit.to_vec(tensor)))
end

function norm_column(qp_peps, qp_env, Bbasis)
    env_ket = PEPSKit._contract_env_ket((1, 1), PEPSKit.ket(qp_peps), qp_env)
    qp_bra = PEPSKit.bra(qp_peps)
    nAA = PEPSKit._contract_env_ket_bra(env_ket[1], qp_bra[1, 1][1])
    gN_tensor = env_ket[2] / nAA
    return project_to_basis(Bbasis, gN_tensor), nAA
end

println("===== RG cost function debug =====")
@show basis_file
@show basis_index
@show N
@show chi
@show pre_rg_steps
@show rg_steps
@show run_cold_start
@show run_warm_start

B = basis_peps(Bbasis, basis_index, gs)
qp_env0 = PEPSKit.qp_CTMRGEnv(env_gs)
trunc = truncerror(; atol=1.0e-10) & truncrank(chi)
pre_qp_alg = PEPSKit.SequentialQPCTMRG(;
    miniter=pre_rg_steps,
    maxiter=pre_rg_steps,
    tol=1.0e-6,
    trunc,
)
qp_alg = PEPSKit.SequentialQPCTMRG(;
    miniter=rg_steps,
    maxiter=rg_steps,
    tol=1.0e-6,
    trunc,
)

function check_rg_cost(label, B, qp_env_start, qp_alg, gs, ham_shift, Bbasis, basis_index)
    qp_env_ad_ref = Ref{Any}()

    E, gradH = withgradient(B) do x
        qp_peps_x = PEPSKit.InfiniteQPPEPS(gs, x)
        qp_env_x, = leading_boundary(qp_env_start, qp_peps_x, qp_alg; conv_level=4)
        ignore_derivatives() do
            qp_env_ad_ref[] = qp_env_x
        end
        PEPSKit.qp_energy(qp_peps_x, qp_env_x, ham_shift)
    end

    qp_peps = PEPSKit.InfiniteQPPEPS(gs, B)
    qp_env_ad = qp_env_ad_ref[]
    E_ad_env = PEPSKit.qp_energy(qp_peps, qp_env_ad, ham_shift)
    N_ad_env = PEPSKit.qp_norm(qp_peps, qp_env_ad)
    projN_ad_env, nAA_ad_env = norm_column(qp_peps, qp_env_ad, Bbasis)
    effN_ad_env_ii = projN_ad_env[basis_index]

    gH = only(gradH)
    projH = project_to_basis(Bbasis, gH[1])
    gH_B = projH[basis_index]

    println("\nCost function check with QP RG: $label")
    @show nAA_ad_env
    @show E_ad_env
    @show N_ad_env
    @show effN_ad_env_ii
    @show effN_ad_env_ii / N_ad_env
    @show E
    @show gH_B
    @show gH_B / 2
    @show gH_B / E
    @show (gH_B / 2) / E
    @show (gH_B / 2) / effN_ad_env_ii
    @show E_ad_env / N_ad_env
end

if run_cold_start
    check_rg_cost("cold start", B, qp_env0, qp_alg, gs, ham_shift, Bbasis, basis_index)
end

if run_warm_start
    println("\nRunning pre-RG warm start")
    qp_peps = PEPSKit.InfiniteQPPEPS(gs, B)
    qp_env_pre, = leading_boundary(qp_env0, qp_peps, pre_qp_alg; conv_level=4)
    check_rg_cost(
        "warm start after pre-RG", B, qp_env_pre, qp_alg, gs, ham_shift, Bbasis, basis_index
    )
end
