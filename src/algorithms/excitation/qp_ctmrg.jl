struct SequentialQPCTMRG <: CTMRGAlgorithm
    kx::Float64
    ky::Float64
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::ProjectorAlgorithm
end

function SequentialQPCTMRG(;
    kx=0.0,
    ky=0.0,
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=Defaults.ctmrg_verbosity,
    trunc=(; alg=Defaults.trunc),
    decomposition_alg=(;),
    projector_alg=Defaults.projector_alg, # only allows for Symbol/NamedTuple to expose projector kwargs
)
    projector_algorithm = ProjectorAlgorithm(;
        alg=projector_alg, decomposition_alg, trunc, verbosity
    )

    return SequentialQPCTMRG(kx, ky, tol, maxiter, miniter, verbosity, projector_algorithm)
end

CTMRG_SYMBOLS[:sequentialqp] = SequentialQPCTMRG

function ctmrg_iteration(network, env::CTMRGEnv, alg::SequentialQPCTMRG)
    truncation_error = zero(real(scalartype(network)))
    condition_number = zero(real(scalartype(network)))

    ks = [(alg.kx * pi, 0.0), (0.0, alg.ky * pi), (-alg.kx * pi, 0.0), (0.0, -alg.ky * pi)]
    for (kx, ky) in ks # rotate, left -> top -> right -> bot
        for col in 1:size(network, 2) # left move column-wise
            env, info = ctmrg_leftmove(col, network, env, alg; kx=kx, ky=ky)
            truncation_error = max(truncation_error, info.truncation_error)
            condition_number = max(condition_number, info.condition_number)
        end
        network = rotate_north(network, EAST)
        env = rotate_north(env, EAST)
    end
    return env, (; truncation_error, condition_number)
end

"""
    ctmrg_leftmove(col::Int, network, env::CTMRGEnv, alg::SequentialCTMRG)

Perform sequential CTMRG left move on the `col`-th column.
"""
function ctmrg_leftmove(
    col::Int, network, env::CTMRGEnv, alg::SequentialQPCTMRG; kx=0.0, ky=0.0
)
    #=
        ----> left move
        C1 ← T1 ←   r-1
        ↓    ‖
        T4 = M ==   r
        ↓    ‖
        C4 → T3 →   r+1
        c-1  c 
    =#
    network_gs = gs_Network(network)
    env_gs = gs_CTMRGEnv(env)
    projectors, info = sequential_projectors(col, network_gs, env_gs, alg.projector_alg)
    env = renormalize_sequentially_qp(col, projectors, network, env; kx=kx, ky=ky)
    return env, info
end

"""
    renormalize_sequentially(col::Int, projectors, network, env)

Renormalize one column of the CTMRG environment.
"""
function renormalize_sequentially_qp(col::Int, projectors, network, env; kx=0.0, ky=0.0)
    corners = Zygote.Buffer(env.corners)
    edges = Zygote.Buffer(env.edges)

    for (dir, r, c) in eachcoordinate(network, 1:4)
        (c == col && dir in [SOUTHWEST, NORTHWEST]) && continue
        corners[dir, r, c] = env.corners[dir, r, c]
    end
    for (dir, r, c) in eachcoordinate(network, 1:4)
        (c == col && dir == WEST) && continue
        edges[dir, r, c] = env.edges[dir, r, c]
    end

    # Apply projectors to renormalize corners and edge
    for row in axes(env.corners, 2)
        C_southwest = renormalize_southwest_corner((row, col), env, projectors)
        corners[SOUTHWEST, row, col] = shift(C_southwest / norm(C_southwest), kx + ky)

        C_northwest = renormalize_northwest_corner((row, col), env, projectors)
        corners[NORTHWEST, row, col] = shift(C_northwest / norm(C_northwest), kx + ky)

        E_west = renormalize_west_edge((row, col), env, projectors, network)
        edges[WEST, row, col] = shift(E_west / norm(E_west), kx + ky)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end

function leading_boundary(
    env₀::CTMRGEnv, network::InfiniteSquareNetwork, alg::SequentialQPCTMRG; conv_level = 1
)
    CS_old = ignore_derivatives() do
        ntuple(i -> map(x -> svd_vals(x[i]), env₀.corners), conv_level)
    end
    TS_old = ignore_derivatives() do
        ntuple(i -> map(x -> svd_vals(x[i]), env₀.edges), conv_level)
    end

    η = one(real(scalartype(network)))
    env = deepcopy(env₀)
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))

    local info
    for iter in 1:(alg.maxiter)
        env, info = ctmrg_iteration(network, env, alg)  # Grow and renormalize in all 4 directions

        ηs = ignore_derivatives() do
            CS_new = ntuple(i -> map(x -> svd_vals(x[i]), env.corners), conv_level)
            TS_new = ntuple(i -> map(x -> svd_vals(x[i]), env.edges), conv_level)
            ηs_local = calc_pair_convergence(CS_new, TS_new, CS_old, TS_old, conv_level)
            CS_old, TS_old = CS_new, TS_new
            ηs_local
        end
        η = conv_level > 0 ? ηs[conv_level] : η

        ignore_derivatives() do
            LoggingExtras.withlevel(; alg.verbosity) do
                println("network value: ", qp_net_norm(network, env)[4])

                if η ≤ alg.tol && iter ≥ alg.miniter
                    ctmrg_logfinish!(log, iter, η, network, env)
                    if conv_level > 1
                        println("convergence: (A, B, Bd, BB) => $(ηs)")
                    end
                    return nothing
                end
                if iter == alg.maxiter
                    ctmrg_logcancel!(log, iter, η, network, env)
                    if conv_level > 1
                        println("convergence: (A, B, Bd, BB) => $(ηs)")
                    end
                else
                    ctmrg_logiter!(log, iter, η, network, env)
                    if conv_level > 1
                        println("convergence: (A, B, Bd, BB) => $(ηs)")
                    end
                end
                return nothing
            end
        end

        if η ≤ alg.tol && iter ≥ alg.miniter
            break
        end
    end

    return env, info
end

function calc_pair_convergence(CS_new, TS_new, CS_old, TS_old, conv_level)
    ΔCS = ntuple(
        i -> maximum(splat(_singular_value_distance), zip(CS_old[i], CS_new[i])), conv_level
    )
    ΔTS = ntuple(
        i -> maximum(splat(_singular_value_distance), zip(TS_old[i], TS_new[i])), conv_level
    )
    ΔS = ntuple(i -> max(ΔCS[i], ΔTS[i]), conv_level)
    return ΔS
end



@generated function renormalize_southwest_corner(
        C_southwest::NestedTensor{<:CTMRGCornerTensor{<:Any, S}},
        E_south::NestedTensor{<:CTMRGEdgeTensor{<:Any, S, N}},
        P_left::AbstractTensorMap{<:Any, S, N, 1},
    ) where {S, N}
    C_out_e = tensorexpr(:corner, (envlabel(:out),), (envlabel(:in),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:SSW),), (envlabel(:WSW),))
    E_south_e = tensorexpr(
        :E_south,
        (envlabel(:out), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:SSW),),
    )
    P_left_e = tensorexpr(
        :P_left,
        (envlabel(:WSW), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:in),),
    )
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $C_out_e := $E_south_e * $C_southwest_e * $P_left_e),
    )
end


@generated function renormalize_northwest_corner(
        C_northwest::NestedTensor{<:CTMRGCornerTensor{<:Any, S}},
        E_north::NestedTensor{<:CTMRGEdgeTensor{<:Any, S, N}},
        P_right::AbstractTensorMap{<:Any, S, 1, N},
    ) where {S, N}
    C_out_e = tensorexpr(:corner, (envlabel(:out),), (envlabel(:in),))
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:WNW),), (envlabel(:NNW),))
    E_north_e = tensorexpr(
        :E_north, (envlabel(:NNW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:in),)
    )
    P_right_e = tensorexpr(
        :P_right, (envlabel(:out),), (envlabel(:WNW), ntuple(i -> virtuallabel(i), N - 1)...)
    )
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $C_out_e := $E_north_e * $C_northwest_e * $P_right_e),
    )
end


function renormalize_west_edge(
        E_west::NestedTensor{<:CTMRG_PEPS_EdgeTensor}, P_left, P_right, A::Tuple{<:NestedTensor, <:NestedTensor}
    )
    # starting with P_bottom to save one permute in the end
    return @tensor begin
        # already putting χE in front here to make next permute cheaper
        PE[χS χNW DSb DWb; DSt DWt] := P_right[χS; χSW DSt DSb] * E_west[χSW DWt DWb; χNW]
        PEket[χS χNW DNt DEt; DSb DWb d] :=
            PE[χS χNW DSb DWb; DSt DWt] * ket(A)[d; DNt DEt DSt DWt]
        corner[χS DEt DEb; χNW DNt DNb] :=
            PEket[χS χNW DNt DEt; DSb DWb d] * conj(bra(A)[d; DNb DEb DSb DWb])
        edge[χS DEt DEb; χN] := corner[χS DEt DEb; χNW DNt DNb] * P_left[χNW DNt DNb; χN]
    end
end
