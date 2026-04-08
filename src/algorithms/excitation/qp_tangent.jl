function to_vec(A)
    sA = space(A)
    sA_t = Tuple(sA[i] for i = 1:numind(A))
    Vl = fuse(sA_t...)
    Vr = *(sA_t...)
    u = isomorphism(storagetype(A), Vl, Vr)
    A_vec = u * permute(A, Tuple(1:numind(A)))
    return A_vec
end

function tangent_basis(env, bra, ket=bra)
    basis = [
        begin
            env_bra = _contract_env_bra((r, c), bra, env)
            left_null(to_vec(env_bra))
        end
        for r in axes(bra, 1), c in axes(bra, 2)
    ]
    return basis
end

function gs_norm1x1(network, env)
    return [_contract_site((r, c), network, env) for r in axes(network, 1), c in axes(network, 2)]
end

function gs_env_bra(env, bra)
    return [_contract_env_bra((r, c), bra, env) for r in axes(bra, 1), c in axes(bra, 2)]
end

function _contract_env_bra(ind::Tuple{Int,Int}, bra, env::CTMRGEnv)
    r, c = ind
    return _contract_env_bra(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c], env.edges[EAST, r, _next(c, end)],
        env.edges[SOUTH, _next(r, end), c], env.edges[WEST, r, _prev(c, end)],
        bra[r, c],
    )
end

function _contract_env_bra(
    C_northwest, C_northeast, C_southeast, C_southwest,
    E_north::CTMRG_PEPS_EdgeTensor, E_east::CTMRG_PEPS_EdgeTensor,
    E_south::CTMRG_PEPS_EdgeTensor, E_west::CTMRG_PEPS_EdgeTensor,
    bra
)
    @autoopt @tensor bra_env[d; D_N_above D_E_above D_S_above D_W_above] :=
        E_west[χ_WSW D_W_above D_W_below; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N_above D_N_below; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E_above D_E_below; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S_above D_S_below; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        conj(bra[d; D_N_below D_E_below D_S_below D_W_below])
    return bra_env
end