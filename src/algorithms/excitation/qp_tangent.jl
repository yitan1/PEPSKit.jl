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
            env_vec = to_vec(env_bra)
            null_row = right_null(transpose(env_vec))
            TensorMap(convert(Array, adjoint(null_row)), space(env_vec, 1) ← space(null_row, 1))
        end
        for r in axes(bra, 1), c in axes(bra, 2)
    ]
    return basis
end

function _basis_column_peps(Bbasis, i::Integer, reference::InfinitePEPS)
    B_tensor = TensorMap(Array(reshape(Bbasis[:, i], dims(reference[1]))), space(reference[1]))
    return InfinitePEPS([B_tensor;;])
end

function shift_hamiltonian(gs, env, ham)
    lat = physicalspace(ham)
    plat = PeriodicArray(lat)
    shifted_terms = map(ham.terms) do (inds, op)
        local_h = LocalOperator(lat, inds => op)
        exp_val = PEPSKit.expectation_value(gs, local_h, env)
        local_space = map(i -> plat[i], inds)
        local_id = TensorKit.id(reduce(⊗, local_space))
        return inds => (op - exp_val * local_id)
    end
    return LocalOperator(lat, shifted_terms...)
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

function _contract_env_bra(
    C_northwest, C_northeast, C_southeast, C_southwest,
    E_north::NestedTensor, E_east::NestedTensor,
    E_south::NestedTensor, E_west::NestedTensor,
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

function _contract_env_ket(ind::Tuple{Int,Int}, ket, env::CTMRGEnv)
    r, c = ind
    return _contract_env_ket(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c], env.edges[EAST, r, _next(c, end)],
        env.edges[SOUTH, _next(r, end), c], env.edges[WEST, r, _prev(c, end)],
        ket[r, c],
    )
end

function _contract_env_ket(
    C_northwest, C_northeast, C_southeast, C_southwest,
    E_north::CTMRG_PEPS_EdgeTensor, E_east::CTMRG_PEPS_EdgeTensor,
    E_south::CTMRG_PEPS_EdgeTensor, E_west::CTMRG_PEPS_EdgeTensor,
    ket
)
    @autoopt @tensor ket_env[d; D_N_below D_E_below D_S_below D_W_below] :=
        E_west[χ_WSW D_W_above D_W_below; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N_above D_N_below; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E_above D_E_below; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S_above D_S_below; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        ket[d; D_N_above D_E_above D_S_above D_W_above]
    return ket_env
end

function _contract_env_ket(
    C_northwest, C_northeast, C_southeast, C_southwest,
    E_north::NestedTensor, E_east::NestedTensor,
    E_south::NestedTensor, E_west::NestedTensor,
    ket
)
    @autoopt @tensor ket_env[d; D_N_below D_E_below D_S_below D_W_below] :=
        E_west[χ_WSW D_W_above D_W_below; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N_above D_N_below; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E_above D_E_below; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S_above D_S_below; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        ket[d; D_N_above D_E_above D_S_above D_W_above]
    return ket_env
end

function _contract_env_bra_ket(env_bra, ket)
    return @autoopt @tensor env_bra[d; D_N_above D_E_above D_S_above D_W_above] *
        ket[d; D_N_above D_E_above D_S_above D_W_above]
end

function _contract_env_ket_bra(env_ket, bra)
    return @autoopt @tensor env_ket[d; D_N_below D_E_below D_S_below D_W_below] *
        conj(bra[d; D_N_below D_E_below D_S_below D_W_below])
end
