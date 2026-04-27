struct InfiniteQPPEPS{T<:PEPSTensor}
    AB::Matrix{NestedTensor{T}}
    InfiniteQPPEPS{T}(AB::Matrix{NestedTensor{T}}) where {T<:PEPSTensor} = new{T}(AB)
    function InfiniteQPPEPS(A::Array{NestedTensor{T},2}) where {T<:PEPSTensor}
        return new{T}(A)
    end
end

function InfiniteQPPEPS(A::InfinitePEPS{T}, B::InfinitePEPS{T}) where {T<:PEPSTensor}
    size(A) == size(B) || throw(DimensionMismatch("A and B unit cell sizes do not match."))
    AB = map(unitcell(A), unitcell(B)) do a, b
        z = VectorInterface.zerovector(a)
        NestedTensor([a, b, z, z])
    end
    return InfiniteQPPEPS(AB)
end

function ChainRulesCore.rrule(
    ::Type{InfiniteQPPEPS}, A::InfinitePEPS{T}, B::InfinitePEPS{T}
) where {T<:PEPSTensor}
    qp = InfiniteQPPEPS(A, B)

    function pullback(Δqp_)
        Δqp = ChainRulesCore.unthunk(Δqp_)
        Δqp isa ChainRulesCore.AbstractZero &&
            return ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent()
        ΔAB = unitcell(Δqp)
        ΔA = InfinitePEPS(map(nt -> nt[1], ΔAB))
        ΔB = InfinitePEPS(map(nt -> nt[2], ΔAB))
        return ChainRulesCore.NoTangent(), ΔA, ΔB
    end

    return qp, pullback
end

function InfiniteQPPEPS(A::InfinitePEPS{T}, B::Array{T, 2}) where {T<:PEPSTensor}
    return InfiniteQPPEPS(A, InfinitePEPS(B))
end

function InfiniteQPPEPS(A::Array{T, 2}, B::Array{T, 2}) where {T<:PEPSTensor}
    return InfiniteQPPEPS(InfinitePEPS(A), InfinitePEPS(B))
end

function ChainRulesCore.rrule(
    ::Type{InfiniteQPPEPS}, AB::Matrix{NestedTensor{T}}
) where {T<:PEPSTensor}
    qp = InfiniteQPPEPS(AB)

    function InfiniteQPPEPS_pullback(Δqp_)
        Δqp = ChainRulesCore.unthunk(Δqp_)
        Δqp isa ChainRulesCore.AbstractZero &&
            return ChainRulesCore.NoTangent(), VectorInterface.zerovector(AB)
        return ChainRulesCore.NoTangent(), unitcell(Δqp)
    end

    return qp, InfiniteQPPEPS_pullback
end

unitcell(qp::InfiniteQPPEPS) = qp.AB
ket(qp_peps::InfiniteQPPEPS) = qp_peps
bra(qp_peps::InfiniteQPPEPS) = InfiniteQPPEPS(exchange_B_Bd(unitcell(qp_peps)))

function ChainRulesCore.rrule(::typeof(unitcell), qp::InfiniteQPPEPS)
    AB = unitcell(qp)

    function unitcell_pullback(ΔAB_)
        ΔAB = ChainRulesCore.unthunk(ΔAB_)
        ΔAB isa ChainRulesCore.AbstractZero &&
            return ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent()
        return ChainRulesCore.NoTangent(), InfiniteQPPEPS(ΔAB)
    end

    return AB, unitcell_pullback
end

Base.size(qp::InfiniteQPPEPS, args...) = size(unitcell(qp), args...)
Base.length(qp::InfiniteQPPEPS) = length(unitcell(qp))

Base.copy(qp::InfiniteQPPEPS) = InfiniteQPPEPS(copy(unitcell(qp)))

Base.getindex(qp::InfiniteQPPEPS, args...) = getindex(unitcell(qp), args...)
Base.setindex!(qp::InfiniteQPPEPS, args...) = (setindex!(unitcell(qp), args...); qp)
Base.axes(qp::InfiniteQPPEPS, args...) = axes(unitcell(qp), args...)

function VectorInterface.scalartype(::Type{<:InfiniteQPPEPS{T}}) where {T<:PEPSTensor}
    return VectorInterface.scalartype(NestedTensor{T})
end
function VectorInterface.scalartype(qp::InfiniteQPPEPS)
    return VectorInterface.scalartype(unitcell(qp)[1])
end
function VectorInterface.zerovector(qp::InfiniteQPPEPS)
    return InfiniteQPPEPS(map(VectorInterface.zerovector, unitcell(qp)))
end

Base.:+(qp1::InfiniteQPPEPS, qp2::InfiniteQPPEPS) = InfiniteQPPEPS(unitcell(qp1) .+ unitcell(qp2))
Base.:-(qp1::InfiniteQPPEPS, qp2::InfiniteQPPEPS) = InfiniteQPPEPS(unitcell(qp1) .- unitcell(qp2))
Base.:*(α::Number, qp::InfiniteQPPEPS) = InfiniteQPPEPS(α .* unitcell(qp))
Base.:*(qp::InfiniteQPPEPS, α::Number) = α * qp
Base.:/(qp::InfiniteQPPEPS, α::Number) = InfiniteQPPEPS(unitcell(qp) ./ α)
LinearAlgebra.norm(qp::InfiniteQPPEPS) = norm(unitcell(qp))

physicalspace(qp::InfiniteQPPEPS) = physicalspace.(unitcell(qp))

function virtualspace(O, dir)
    return virtualspace(O[1], dir) ⊗ virtualspace(O[2], dir)'
end
virtualspace(O::NestedTensor, dir) = virtualspace(O[1], dir)


function ChainRulesCore.rrule(::typeof(Base.getindex), qp::InfiniteQPPEPS, args...)
    tensor = qp[args...]

    function getindex_pullback(Δtensor_)
        Δtensor = ChainRulesCore.unthunk(Δtensor_)
        Δqp = VectorInterface.zerovector(qp)
        Δqp[args...] = Δtensor
        return ChainRulesCore.NoTangent(), Δqp, ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent()
    end

    return tensor, getindex_pullback
end

# interface with InfiniteSquareNetwork
function InfiniteSquareNetwork(top::InfiniteQPPEPS, bot::InfiniteQPPEPS=top)
    size(top) == size(bot) || throw(
        ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have the same length")
    )
    return InfiniteSquareNetwork(map(tuple, unitcell(top), exchange_B_Bd(unitcell(bot))))
end

function ChainRulesCore.rrule(
    ::Type{InfiniteSquareNetwork}, A::Matrix{O}
) where {O<:Tuple{<:NestedTensor, <:NestedTensor}}
    network = InfiniteSquareNetwork(A)

    function InfiniteSquareNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        Δnetwork isa AbstractZero && return NoTangent(), VectorInterface.zerovector(A)
        return NoTangent(), unitcell(Δnetwork)
    end

    return network, InfiniteSquareNetwork_pullback
end

ket(O::Tuple{<:NestedTensor, <:NestedTensor}) = O[1]
bra(O::Tuple{<:NestedTensor, <:NestedTensor}) = O[2]

function exchange_B_Bd(A::Matrix{T}) where {T<:NestedTensor}
    return map(a -> NestedTensor([a[1], a[3], a[2], a[4]]), A)
end

# Network: NestedTensor -> A
function gs_Network(network::InfiniteSquareNetwork{<:Tuple{<:NestedTensor, <:NestedTensor}})
    new_network = map(a -> (a[1][1], a[2][1]), unitcell(network))
    return InfiniteSquareNetwork(new_network)
end

function ChainRulesCore.rrule(
    ::Type{InfiniteSquareNetwork}, top::InfiniteQPPEPS, bot::InfiniteQPPEPS
)
    network = InfiniteSquareNetwork(top, bot)

    function InfiniteSquareNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        Δnetwork isa AbstractZero &&
            return NoTangent(), ZeroTangent(), ZeroTangent()
        Δtop = InfiniteQPPEPS(map(ket, unitcell(Δnetwork)))
        # `bot` is wrapped with `exchange_B_Bd` in the forward pass, so undo it here.
        Δbot = InfiniteQPPEPS(exchange_B_Bd(map(bra, unitcell(Δnetwork))))
        return NoTangent(), Δtop, Δbot
    end

    return network, InfiniteSquareNetwork_pullback
end
