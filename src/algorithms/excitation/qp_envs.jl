# struct QPCTMRGEnv end
function _twist_north_east_edges!(env::CTMRGEnv; leg::Integer = 2)
    for r in axes(env.edges, 2), c in axes(env.edges, 3)
        env.edges[NORTH, r, c] = TensorKit.twist(env.edges[NORTH, r, c], leg)
        env.edges[EAST, r, c] = TensorKit.twist(env.edges[EAST, r, c], leg)
    end
    return env
end

"""
    init_env(peps) -> env0

Construct a fixed CTMRG initial environment for PEPSKit:

`env0 = CTMRGEnv(ones, ComplexF64, peps, oneunit(spacetype(peps.A[1])))`

Then twist the north/east edge tensors with `twist(_, 2)` on the second output leg.

For PEPS edge tensors (3 output legs) the convention is:
- north: `[χ_NNW D_N_above D_N_below; χ_NNE]`
- east:  `[χ_ENE D_E_above D_E_below; χ_ESE]`
so this twists the `D_*_above` leg.
"""
function init_env(peps)
    hasproperty(peps, :A) ||
        throw(ArgumentError("init_env expects `peps` to have field `A` (e.g. `PEPSKit.InfinitePEPS`)."))

    local_tensor = getproperty(peps, :A)[begin]
    t = local_tensor isa Tuple ? first(local_tensor) : local_tensor
    χ = TensorKit.oneunit(TensorKit.spacetype(t))

    env0 = CTMRGEnv(TensorKit.ones, ComplexF64, peps, χ)
    return _twist_north_east_edges!(env0)
end

function check_environment_virtualspace(E::NestedTensor)
    return isdual(space(E, 1)) &&
        throw(ArgumentError("Dual environment virtual spaces are not allowed (for now)."))
end

function qp_CTMRGEnv(env0::CTMRGEnv)
    Cs = map(a -> nested_single0(a), env0.corners)
    Es = map(a -> nested_single0(a), env0.edges)
    return CTMRGEnv(Cs, Es)
end

# env: NestedTensor -> A
function gs_CTMRGEnv(env0::CTMRGEnv{<:NestedTensor, <:NestedTensor})
    new_corners = map(x -> x[1], env0.corners)
    new_edges   = map(x -> x[1], env0.edges)
    return CTMRGEnv(new_corners, new_edges)
end

