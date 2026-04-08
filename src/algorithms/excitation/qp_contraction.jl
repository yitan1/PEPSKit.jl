function contract_local_operator(
        inds::NTuple{N, CartesianIndex{2}},
        O::AbstractTensorMap{T, S, N, N},
        ket::InfiniteQPPEPS, bra::InfiniteQPPEPS,
        env::CTMRGEnv,
    ) where {T, S, N}
    static_inds = Val.(inds)
    return _contract_local_operator(static_inds, O, (ket, bra), env)
end

@generated function _contract_local_operator(
        inds::NTuple{N, Val},
        O::AbstractTensorMap{T, S, N, N},
        state::Tuple{InfiniteQPPEPS, InfiniteQPPEPS},
        env::CTMRGEnv,
    ) where {T, S, N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) ||
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange, 2)
    operator = tensorexpr(
        :O,
        ntuple(i -> physicallabel(:O, 2, i), N),
        ntuple(i -> physicallabel(:O, 1, i), N),
    )
    ket, bra = _contract_state_expr(rowrange, colrange, 2, cartesian_inds)

    multiplication_ex = Expr(
        :call, :*,
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
        ket..., map(x -> Expr(:call, :conj, x), bra)...,
        operator,
    )

    returnex = quote
        @autoopt @tensor $multiplication_ex
    end
    return macroexpand(@__MODULE__, returnex)
end


function contract_local_norm(
        inds::NTuple{N, CartesianIndex{2}}, ket::InfiniteQPPEPS, bra::InfiniteQPPEPS, env::CTMRGEnv
    ) where {N}
    static_inds = Val.(inds)
    return _contract_local_norm(static_inds, (ket, bra), env)
end

@generated function _contract_local_norm(
        inds::NTuple{N, Val}, state::Tuple{InfiniteQPPEPS, InfiniteQPPEPS}, env::CTMRGEnv
    ) where {N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) || throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange, 2)
    ket, bra = _contract_state_expr(rowrange, colrange, 2)

    multiplication_ex = Expr(
        :call, :*,
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
        ket..., map(x -> Expr(:call, :conj, x), bra)...,
    )

    returnex = quote
        @autoopt @tensor $multiplication_ex
    end
    return macroexpand(@__MODULE__, returnex)
end



function _contract_site(
        C_northwest, C_northeast, C_southeast, C_southwest,
        E_north::NestedTensor, E_east::NestedTensor, 
        E_south::NestedTensor, E_west::NestedTensor,
        O,
    )
    return @autoopt @tensor E_west[χ_WSW D_W_above D_W_below; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N_above D_N_below; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E_above D_E_below; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S_above D_S_below; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end

function _contract_corners(
        ind::Tuple{Int, Int}, env::CTMRGEnv{<:NestedTensor, <:NestedTensor}
    )
    r, c = ind
    C_NW = env.corners[NORTHWEST, _prev(r, end), _prev(c, end)]
    C_NE = env.corners[NORTHEAST, _prev(r, end), c]
    C_SE = env.corners[SOUTHEAST, r, c]
    C_SW = env.corners[SOUTHWEST, r, _prev(c, end)]
    return @tensor C_NW[1; 2] * C_NE[2; 3] * C_SE[3; 4] * C_SW[4; 1]
end

function _contract_vertical_edges(
        ind::Tuple{Int, Int}, env::CTMRGEnv{<:NestedTensor, <:NestedTensor}
    )
    r, c = ind
    return _contract_vertical_edges(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), c],
        env.corners[SOUTHEAST, _next(r, end), c],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[EAST, r, c],
        env.edges[WEST, r, _prev(c, end)],
    )
end

@generated function _contract_vertical_edges(
        C_northwest::NestedTensor,
        C_northeast::NestedTensor,
        C_southeast::NestedTensor,
        C_southwest::NestedTensor,
        E_east::NestedTensor{<:CTMRGEdgeTensor{T, S, N}},
        E_west::NestedTensor{<:CTMRGEdgeTensor{T, S, N}},
    ) where {T, S, N}
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:NW),), (envlabel(:N),))
    C_northeast_e = tensorexpr(:C_northeast, (envlabel(:N),), (envlabel(:NE),))
    C_southeast_e = tensorexpr(:C_southeast, (envlabel(:SE),), (envlabel(:S),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:S),), (envlabel(:SW),))

    E_east_e = tensorexpr(
        :E_east, (envlabel(:NE), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:SE),)
    )
    E_west_e = tensorexpr(
        :E_west, (envlabel(:SW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:NW),)
    )

    rhs = Expr(
        :call, :*,
        E_west_e, C_northwest_e, C_northeast_e, E_east_e, C_southeast_e, C_southwest_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
end

function _contract_horizontal_edges(
        ind::Tuple{Int, Int}, env::CTMRGEnv{<:NestedTensor, <:NestedTensor}
    )
    r, c = ind
    return _contract_horizontal_edges(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, r, _next(c, end)],
        env.corners[SOUTHWEST, r, _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c],
        env.edges[SOUTH, r, c],
    )
end

@generated function _contract_horizontal_edges(
        C_northwest::NestedTensor,
        C_northeast::NestedTensor,
        C_southeast::NestedTensor,
        C_southwest::NestedTensor,
        E_north::NestedTensor{<:CTMRGEdgeTensor{T, S, N}},
        E_south::NestedTensor{<:CTMRGEdgeTensor{T, S, N}},
    ) where {T, S, N}
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:W),), (envlabel(:NW),))
    C_northeast_e = tensorexpr(:C_northeast, (envlabel(:NE),), (envlabel(:E),))
    C_southeast_e = tensorexpr(:C_southeast, (envlabel(:E),), (envlabel(:SE),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:SW),), (envlabel(:W),))

    E_north_e = tensorexpr(
        :E_north, (envlabel(:NW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:NE),)
    )
    E_south_e = tensorexpr(
        :E_south, (envlabel(:SE), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:SW),)
    )

    rhs = Expr(
        :call, :*,
        C_northwest_e, E_north_e, C_northeast_e, C_southeast_e, E_south_e, C_southwest_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
end