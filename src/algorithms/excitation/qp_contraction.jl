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


# reduced density matrix
function reduced_densitymatrix(
        inds::NTuple{N, CartesianIndex{2}}, ket::InfiniteQPPEPS, bra::InfiniteQPPEPS, env::CTMRGEnv
    ) where {N}
    static_inds = Val.(inds)
    return _contract_densitymatrix(static_inds, (ket, bra), env)
end

function reduced_densitymatrix(
        inds::NTuple{N, Tuple{Int, Int}}, ket::InfiniteQPPEPS, bra::InfiniteQPPEPS, env::CTMRGEnv
    ) where {N}
    return reduced_densitymatrix(CartesianIndex.(inds), ket, bra, env)
end

function reduced_densitymatrix(inds, ket::InfiniteQPPEPS, env::CTMRGEnv)
    return reduced_densitymatrix(inds, ket, ket, env)
end

# Special case 1x1 density matrix:
# Keep contraction order but try to optimize intermediate permutations:
# EE_SWA is largest object so keep largest legs to the front there
function reduced_densitymatrix(
        inds::Tuple{CartesianIndex{2}}, ket::InfiniteQPPEPS, bra::InfiniteQPPEPS, env::CTMRGEnv
    )
    row, col = Tuple(inds[1])

    # Unpack variables and absorb corners
    A = ket[mod1(row, end), mod1(col, end)]
    Ā = bra[mod1(row, end), mod1(col, end)]

    E_north =
        env.edges[NORTH, mod1(row - 1, end), mod1(col, end)] *
        twistdual(env.corners[NORTHEAST, mod1(row - 1, end), mod1(col + 1, end)], 1)
    E_east =
        env.edges[EAST, mod1(row, end), mod1(col + 1, end)] *
        twistdual(env.corners[SOUTHEAST, mod1(row + 1, end), mod1(col + 1, end)], 1)
    E_south =
        env.edges[SOUTH, mod1(row + 1, end), mod1(col, end)] *
        twistdual(env.corners[SOUTHWEST, mod1(row + 1, end), mod1(col - 1, end)], 1)
    E_west =
        env.edges[WEST, mod1(row, end), mod1(col - 1, end)] *
        twistdual(env.corners[NORTHWEST, mod1(row - 1, end), mod1(col - 1, end)], 1)

    @tensor EE_SW[χSE χNW DSb DWb; DSt DWt] :=
        E_south[χSE DSt DSb; χSW] * E_west[χSW DWt DWb; χNW]

    @tensor EE_SWA[χSE χNW DNt DEt; dt DSb DWb] :=
        EE_SW[χSE χNW DSb DWb; DSt DWt] * A[dt; DNt DEt DSt DWt]

    @tensor EE_NE[DNb DEb; χSE χNW DNt DEt] :=
        E_north[χNW DNt DNb; χNE] * E_east[χNE DEt DEb; χSE]

    @tensor EEAEE[dt; DNb DEb DSb DWb] :=
        EE_NE[DNb DEb; χSE χNW DNt DEt] * EE_SWA[χSE χNW DNt DEt; dt DSb DWb]

    @tensor ρ[dt; db] := EEAEE[dt; DNb DEb DSb DWb] * conj(Ā[db; DNb DEb DSb DWb])

    return ρ / (str(ρ)[1])
end

function reduced_densitymatrix(
        inds::NTuple{2, CartesianIndex{2}}, ket::InfiniteQPPEPS, bra::InfiniteQPPEPS, env::CTMRGEnv
    )
    if inds[2] - inds[1] == CartesianIndex(1, 0)
        return reduced_densitymatrix2x1(inds[1], ket, bra, env)
    elseif inds[2] - inds[1] == CartesianIndex(0, 1)
        return reduced_densitymatrix1x2(inds[1], ket, bra, env)
    else
        static_inds = Val.(inds)
        return _contract_densitymatrix(static_inds, (ket, bra), env)
    end
end

# Special case 2x1 density matrix:
# Keep contraction order but try to optimize intermediate permutations:
function reduced_densitymatrix2x1(
        ind::CartesianIndex, ket::InfiniteQPPEPS, bra::InfiniteQPPEPS, env::CTMRGEnv
    )
    row, col = Tuple(ind)

    # Unpack variables and absorb corners
    A_north = ket[mod1(row, end), mod1(col, end)]
    Ā_north = bra[mod1(row, end), mod1(col, end)]
    A_south = ket[mod1(row + 1, end), mod1(col, end)]
    Ā_south = bra[mod1(row + 1, end), mod1(col, end)]

    E_north =
        env.edges[NORTH, mod1(row - 1, end), mod1(col, end)] *
        twistdual(env.corners[NORTHEAST, mod1(row - 1, end), mod1(col + 1, end)], 1)
    E_northeast = env.edges[EAST, mod1(row, end), mod1(col + 1, end)]
    E_southeast =
        env.edges[EAST, mod1(row + 1, end), mod1(col + 1, end)] *
        twistdual(env.corners[SOUTHEAST, mod1(row + 2, end), mod1(col + 1, end)], 1)
    E_south =
        env.edges[SOUTH, mod1(row + 2, end), mod1(col, end)] *
        twistdual(env.corners[SOUTHWEST, mod1(row + 2, end), mod1(col - 1, end)], 1)
    E_southwest = env.edges[WEST, mod1(row + 1, end), mod1(col - 1, end)]
    E_northwest =
        env.edges[WEST, mod1(row, end), mod1(col - 1, end)] *
        twistdual(env.corners[NORTHWEST, mod1(row - 1, end), mod1(col - 1, end)], 1)

    @tensor EE_NW[χW χNE DNWt DNt; DNWb DNb] :=
        E_northwest[χW DNWt DNWb; χNW] * E_north[χNW DNt DNb; χNE]
    @tensor EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] :=
        EE_NW[χW χNE DNWt DNt; DNWb DNb] * conj(Ā_north[dNb; DNb DNEb DMb DNWb])
    @tensor EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] :=
        EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] * A_north[dNt; DNt DNEt DMt DNWt]
    @tensor EEEAA_N[dNt dNb; χW DMt DMb χE] :=
        EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] * E_northeast[χNE DNEt DNEb; χE]

    @tensor EE_SE[χE χSW DSEt DSt; DSEb DSb] :=
        E_southeast[χE DSEt DSEb; χSE] * E_south[χSE DSt DSb; χSW]
    @tensor EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] :=
        EE_SE[χE χSW DSEt DSt; DSEb DSb] * conj(Ā_south[dSb; DMb DSEb DSb DSWb])
    @tensor EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] :=
        EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] * A_south[dSt; DMt DSEt DSt DSWt]
    @tensor EEEAA_S[χW DMt DMb χE; dSt dSb] :=
        EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] * E_southwest[χSW DSWt DSWb; χW]

    @tensor ρ[dNt dSt; dNb dSb] :=
        EEEAA_N[dNt dNb; χW DMt DMb χE] * EEEAA_S[χW DMt DMb χE; dSt dSb]

    return ρ / (str(ρ)[1])
end

function reduced_densitymatrix1x2(
        ind::CartesianIndex, ket::InfiniteQPPEPS, bra::InfiniteQPPEPS, env::CTMRGEnv
    )
    row, col = Tuple(ind)

    # Unpack variables and absorb corners
    A_west = ket[mod1(row, end), mod1(col, end)]
    Ā_west = bra[mod1(row, end), mod1(col, end)]
    A_east = ket[mod1(row, end), mod1(col + 1, end)]
    Ā_east = bra[mod1(row, end), mod1(col + 1, end)]

    E_northwest = env.edges[NORTH, mod1(row - 1, end), mod1(col, end)]
    E_northeast =
        env.edges[NORTH, mod1(row - 1, end), mod1(col + 1, end)] *
        twistdual(env.corners[NORTHEAST, mod1(row - 1, end), mod1(col + 2, end)], 1)
    E_east =
        env.edges[EAST, mod1(row, end), mod1(col + 2, end)] *
        twistdual(env.corners[SOUTHEAST, mod1(row + 1, end), mod1(col + 2, end)], 1)
    E_southeast = env.edges[SOUTH, mod1(row + 1, end), mod1(col + 1, end)]
    E_southwest =
        env.edges[SOUTH, mod1(row + 1, end), mod1(col, end)] *
        twistdual(env.corners[SOUTHWEST, mod1(row + 1, end), mod1(col - 1, end)], 1)
    E_west =
        env.edges[WEST, mod1(row, end), mod1(col - 1, end)] *
        twistdual(env.corners[NORTHWEST, mod1(row - 1, end), mod1(col - 1, end)], 1)

    @tensor EE_SW[χS χNW DSWt DWt; DSWb DWb] :=
        E_southwest[χS DSWt DSWb; χSW] * E_west[χSW DWt DWb; χNW]
    @tensor EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] :=
        EE_SW[χS χNW DSWt DWt; DSWb DWb] * conj(Ā_west[dWb; DNWb DMb DSWb DWb])
    @tensor EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] :=
        EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] * A_west[dWt; DNWt DMt DSWt DWt]
    @tensor EEEAA_W[dWt dWb; χS DMt DMb χN] :=
        EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] * E_northwest[χNW DNWt DNWb; χN]

    @tensor EE_NE[χN χSE DNEt DEt; DNEb DEb] :=
        E_northeast[χN DNEt DNEb; χNE] * E_east[χNE DEt DEb; χSE]
    @tensor EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] :=
        EE_NE[χN χSE DNEt DEt; DNEb DEb] * conj(Ā_east[dEb; DNEb DEb DSEb DMb])
    @tensor EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] :=
        EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] * A_east[dEt; DNEt DEt DSEt DMt]
    @tensor EEEAA_E[χS DMt DMb χN; dEt dEb] :=
        EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] * E_southeast[χSE DSEt DSEb; χS]

    @tensor ρ[dWt dEt; dWb dEb] :=
        EEEAA_W[dWt dWb; χS DMt DMb χN] * EEEAA_E[χS DMt DMb χN; dEt dEb]

    return ρ / (str(ρ)[1])
end

@generated function _contract_densitymatrix(
        inds::NTuple{N, Val}, state::Tuple{InfiniteQPPEPS, InfiniteQPPEPS}, env::CTMRGEnv
    ) where {N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) ||
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange, 2)
    result = tensorexpr(
        :ρ,
        ntuple(i -> physicallabel(:O, 1, i), N),
        ntuple(i -> physicallabel(:O, 2, i), N),
    )
    ket, bra = _contract_state_expr(rowrange, colrange, 2, cartesian_inds)

    multiplication_ex = Expr(
        :call, :*,
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
        ket..., map(x -> Expr(:call, :conj, x), bra)...,
    )
    multex = :(@autoopt @tensor $result := $multiplication_ex)
    return quote
        $(macroexpand(@__MODULE__, multex))
        return ρ / (str(ρ)[1])
    end
end