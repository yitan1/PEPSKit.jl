using JLD2
using PEPSKit
using TensorKit

# Step 2: load the cached ground state and compute the tangent basis.

cache_dir = joinpath(@__DIR__, "cache")
groundstate_file = joinpath(cache_dir, "groundstate.jld2")
isfile(groundstate_file) || error("Missing cache file: run qp_groundstate.jl first")

data = load(groundstate_file)
ham = data["ham"]
gs = data["gs"]
env_gs = data["env_gs"]

println("Computing shifted Hamiltonian")
ham_shift = PEPSKit.shift_hamiltonian(gs, env_gs, ham)

println("Computing tangent basis")
Bs = PEPSKit.tangent_basis(env_gs, gs)
Bbasis = Bs[1, 1]
N = dims(Bbasis)[2]

out = joinpath(cache_dir, "basis.jld2")
jldsave(out; ham, ham_shift, gs, env_gs, Bs, N)

println("Saved basis cache:")
println(out)
@show N
