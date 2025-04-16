function network_value(
    network::InfiniteSquareNetwork{Tuple{T1, T1}},
    env::CTMRGEnv{T2, T3}
) where {T1<:NestedTensor, T2<:NestedTensor, T3<:NestedTensor}
    gs_net = gs_Network(network)
    gs_env = gs_CTMRGEnv(env)
    return network_value(gs_net, gs_env)
end


function excitation(H, alg, A::InfinitePEPS, env::CTMRGEnv)
end