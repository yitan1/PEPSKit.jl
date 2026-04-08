# struct QPCTMRGEnv end

function check_environment_virtualspace(E::NestedTensor)
    return isdual(space(E, 1)) &&
        throw(ArgumentError("Dual environment virtual spaces are not allowed (for now)."))
end

function qp_CTMRGEnv(env0::CTMRGEnv)
    Cs = map(a -> nested_single(a), env0.corners)
    Es = map(a -> nested_single(a), env0.edges)
    return CTMRGEnv(Cs, Es)
end

# env: NestedTensor -> A
function gs_CTMRGEnv(env0::CTMRGEnv{<:NestedTensor, <:NestedTensor})
    new_corners = map(x -> x[1], env0.corners)
    new_edges   = map(x -> x[1], env0.edges)
    return CTMRGEnv(new_corners, new_edges)
end

