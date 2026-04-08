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
    A_uc = unitcell(A)
    B_uc = unitcell(B)
    projA = ChainRulesCore.ProjectTo(A)
    projB = ChainRulesCore.ProjectTo(B)

    function pullback(Δqp_)
        Δqp = ChainRulesCore.unthunk(Δqp_)
        Δqp isa ChainRulesCore.AbstractZero &&
            return ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(), ChainRulesCore.ZeroTangent()

        ΔAB = Δqp isa InfiniteQPPEPS ? unitcell(Δqp) : ChainRulesCore.unthunk(getproperty(Δqp, :AB))

        comp(Δnt, i) = begin
            Δnt = ChainRulesCore.unthunk(Δnt)
            if Δnt isa NestedTensor
                return Δnt[i]
            else
                return ChainRulesCore.unthunk(getproperty(Δnt, :Ts))[i]
            end
        end

        dA_uc = similar(A_uc)
        dB_uc = similar(B_uc)
        @inbounds for I in eachindex(ΔAB)
            dA_I = ChainRulesCore.unthunk(comp(ΔAB[I], 1))
            dB_I = ChainRulesCore.unthunk(comp(ΔAB[I], 2))

            dA_uc[I] = dA_I isa ChainRulesCore.AbstractZero ?
                        VectorInterface.zerovector(A_uc[I]) :
                        ChainRulesCore.ProjectTo(A_uc[I])(dA_I)
            dB_uc[I] = dB_I isa ChainRulesCore.AbstractZero ?
                        VectorInterface.zerovector(B_uc[I]) :
                        ChainRulesCore.ProjectTo(B_uc[I])(dB_I)
        end

        dA = InfinitePEPS(dA_uc)
        dB = InfinitePEPS(dB_uc)

        return ChainRulesCore.NoTangent(), projA(dA), projB(dB)
    end

    return qp, pullback
end

function InfiniteQPPEPS(A::Array{T, 2}, B::Array{T, 2}) where {T<:PEPSTensor}
    return InfiniteQPPEPS(InfinitePEPS(A), InfinitePEPS(B))
end

unitcell(qp::InfiniteQPPEPS) = qp.AB
ket(qp_peps::InfiniteQPPEPS) = qp_peps
bra(qp_peps::InfiniteQPPEPS) = InfiniteQPPEPS(exchange_B_Bd(unitcell(qp_peps)))

Base.size(qp::InfiniteQPPEPS, args...) = size(unitcell(qp), args...)
Base.length(qp::InfiniteQPPEPS) = length(unitcell(qp))

Base.copy(qp::InfiniteQPPEPS) = InfiniteQPPEPS(copy(unitcell(qp)))

Base.getindex(qp::InfiniteQPPEPS, args...) = getindex(unitcell(qp), args...)
Base.setindex!(qp::InfiniteQPPEPS, args...) = (setindex!(unitcell(qp), args...); qp)
Base.axes(qp::InfiniteQPPEPS, args...) = axes(unitcell(qp), args...)

physicalspace(qp::InfiniteQPPEPS) = physicalspace.(qp.AB)

function virtualspace(O, dir)
    return virtualspace(O[1], dir) ⊗ virtualspace(O[2], dir)'
end
virtualspace(O::NestedTensor, dir) = virtualspace(O[1], dir)


# interface with InfiniteSquareNetwork
function InfiniteSquareNetwork(top::InfiniteQPPEPS, bot::InfiniteQPPEPS=top)
    size(top) == size(bot) || throw(
        ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have the same length")
    )
    return InfiniteSquareNetwork(map(tuple, unitcell(top), exchange_B_Bd(unitcell(bot))))
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
        Δtop = InfiniteQPPEPS(map(ket, unitcell(Δnetwork)))
        # `bot` is wrapped with `exchange_B_Bd` in the forward pass, so undo it here.
        Δbot = InfiniteQPPEPS(exchange_B_Bd(map(bra, unitcell(Δnetwork))))
        return NoTangent(), Δtop, Δbot
    end

    return network, InfiniteSquareNetwork_pullback
end

# This is a patch for InfiniteSquareNetwork; QP algorithms calls the rrule when computing projector by gs_network function
function ChainRulesCore.rrule(::Type{InfiniteSquareNetwork}, A::AbstractMatrix)
    network = InfiniteSquareNetwork(A)

    function InfiniteSquareNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        ΔA = Δnetwork isa InfiniteSquareNetwork ?
             unitcell(Δnetwork) :
             unthunk(getproperty(Δnetwork, :A))
        return NoTangent(), ΔA
    end

    return network, InfiniteSquareNetwork_pullback
end