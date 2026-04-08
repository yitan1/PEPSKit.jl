# struct InfiniteQPExcitation
#     boundary_alg::CTMRGAlgorithm
# end

function excitation(
    ham, alg, A::InfinitePEPS, env::CTMRGEnv; pre_ctm_alg::CTMRGAlgorithm=nothing
)
    # compute basis
    B = A

    # initialize the PEPS and environment
    qp_peps0 = InfiniteQPPEPS(A, B)
    qp_env0 = qp_CTMRGEnv(env)

    # for i = 1:N
    #     println(" ============    Basis ", i)
    #     Bd = TensorMap(conj(Array(reshape(Bs[:, i], dims(Ad)))), space(Ad))
    #     B = copy(Bd')

    #     # BHB
    #     @time gH, gN = run_es(A, B, h)
    #     effH[:, i] = convert(Array, Bs'* to_vec(gH))
    #     effN[:, i] = convert(Array, Bs'* to_vec(gN))
    #     @show effH[i, i], effN[i, i]
    # end

    if pre_ctm_alg === nothing
        pre_ctm_alg = alg.boundary_alg
    end
    qp_env1, _ = leading_boundary(qp_env0, qp_peps0, pre_ctm_alg; conv_level=1)

    E, gs = withgradient(B) do x
        qp_peps = InfiniteQPPEPS(A, x)
        qp_env′ = qp_env1
        # qp_env′, info = hook_pullback(
        #     leading_boundary,
        #     qp_env1,
        #     qp_peps,
        #     alg.boundary_alg;
        #     conv_level=1,
        #     alg_rrule=nothing,
        # )
        # ignore_derivatives() do
        #     update!(qp_env0, qp_env′)
        # end
        return qp_energy(qp_peps, qp_env′, ham)
    end
    g = only(gs)

    return E, g
end

function run_es(A, B, env, ham)
    qp_peps = InfiniteQPPEPS(A, x)

end

# optimize by density matrix
function qp_expectation_value(
    qp_peps::InfiniteQPPEPS, O::LocalOperator, env::CTMRGEnv{T2,T3}; E_order::Int=4, N_order::Int=1
) where {T2<:NestedTensor,T3<:NestedTensor}
    # checklattice(qp_peps, O)
    peps_ket = ket(qp_peps)
    peps_bra = bra(qp_peps)
    term_vals = dtmap([O.terms...]) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        Ei = contract_local_operator(inds, operator, peps_ket, peps_bra, env)
        Ni = contract_local_norm(inds, peps_ket, peps_bra, env)
        Ei[E_order] / Ni[N_order]
    end
    return sum(term_vals)
end

function qp_energy(
    peps::InfiniteQPPEPS, env::CTMRGEnv{T2,T3}, O::LocalOperator
) where {T2<:NestedTensor,T3<:NestedTensor}
    E = qp_expectation_value(peps, O, env)
    ignore_derivatives() do
        isapprox(imag(E), 0; atol=sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end

function network_value(
    network::InfiniteSquareNetwork{Tuple{T1,T1}}, env::CTMRGEnv{T2,T3}
) where {T1<:NestedTensor,T2<:NestedTensor,T3<:NestedTensor}
    gs_net = gs_Network(network)
    gs_env = gs_CTMRGEnv(env)
    return network_value(gs_net, gs_env)
end

function qp_network_value(network::InfiniteSquareNetwork{<:Tuple{<:NestedTensor,<:NestedTensor}}, env::CTMRGEnv{T2,T3}) where {T2<:NestedTensor,T3<:NestedTensor}
    return prod(Iterators.product(axes(network)...)) do (r, c)
        upper = _contract_site((r, c), network, env) * _contract_corners((r, c), env)
        bottom = _contract_vertical_edges((r, c), env) * _contract_horizontal_edges((r, c), env)
        return upper / bottom
    end
end


