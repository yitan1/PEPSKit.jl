using Random

using JLD2
using MatrixAlgebraKit
using PEPSKit
using TensorKit

# Step 1: compute and cache a ground state and its CTMRG environment.

cache_dir = joinpath(@__DIR__, "cache")
mkpath(cache_dir)

g = 3.1
D = 2
χ = 16
gradtol = 1.0e-3

trunc = truncerror(; atol=1.0e-10) & truncrank(χ)

ham = transverse_field_ising(InfiniteSquare(); g)
Random.seed!(2928528935)
peps0 = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))

println("Computing initial CTMRG environment")
env0 = PEPSKit.init_env(peps0)
env1, = leading_boundary(
    env0,
    peps0,
    SimultaneousCTMRG(; maxiter=50, tol=1.0e-6, trunc),
)

println("Optimizing ground state")
gs, env_gs, E, info = fixedpoint(ham, peps0, env1; tol=gradtol)

out = joinpath(cache_dir, "groundstate.jld2")
jldsave(out; ham, gs, env_gs, E, info, g, D, χ, gradtol)

println("Saved ground state cache:")
println(out)
@show E
