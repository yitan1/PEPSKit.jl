function ChainRulesCore.rrule(
    ::Type{NestedTensor}, Ts::Vector{C}
) where {C<:AbstractTensorMap}
    nt = NestedTensor(Ts)
    function pullback(Δnt_)
        Δnt = unthunk(Δnt_)
        return NoTangent(), Δnt.Ts
    end
    return nt, pullback
end

function ChainRulesCore.rrule(::typeof(getindex), t::NestedTensor, i::Int)
    y = getindex(t, i)
    project_t = ProjectTo(t)
    project_y = ProjectTo(y)

    function pullback(Δy_)
        Δy = unthunk(Δy_)
        if Δy isa AbstractZero
            return NoTangent(), Δy, NoTangent()
        end

        dTs = [zerovector(tt) for tt in t.Ts]
        dTs[i] = project_y(Δy)
        return NoTangent(), project_t(NestedTensor(dTs)), NoTangent()
    end

    return y, pullback
end

function _rrule_via_ad_project(config, f, projectors::Tuple, args...)
    y, pb = rrule_via_ad(config, f, args...)
    function pullback(Δy)
        d = pb(Δy)
        return (
            d[1],
            ntuple(length(projectors)) do i
                di = d[i + 1]
                p = projectors[i]
                return p === nothing ? di : p(unthunk(di))
            end...,
        )
    end
    return y, pullback
end

function ChainRulesCore.rrule(::typeof(TO.tensorscalar), C::NestedTensor)
    y = TO.tensorscalar(C)
    function pullback(Δy_)
        Δy = unthunk(Δy_)
        ΔTs = Δy.Ts
        dT = TO.tensoralloc(typeof(C), TO.tensorstructure(C))
        return NoTangent(), ProjectTo(C)(fill!(dT, ΔTs))
    end
    return y, pullback
end

function ChainRulesCore.rrule(
    ::typeof(TO.tensoradd!),
    C::NestedTensor,
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    α::Number,
    β::Number,
    ba...,
)
    function f(C, A, pA, conjA, α, β, ba...)
        Ts = Zygote.Buffer(C.Ts)
        for i in eachindex(C.Ts)
            Ts[i] = C.Ts[i]
        end
        for i in 1:4
            Ts[i] = TO.tensoradd!(Ts[i], A.Ts[i], pA, conjA, α, β, ba...)
        end
        return NestedTensor(copy(Ts))
    end

    config = Zygote.ZygoteRuleConfig()
    projectors = (
        ProjectTo(C),
        ProjectTo(A),
        nothing,
        nothing,
        nothing,
        nothing,
        ntuple(_ -> nothing, length(ba))...,
    )
    return _rrule_via_ad_project(config, f, projectors, C, A, pA, conjA, α, β, ba...)
end

function ChainRulesCore.rrule(
    ::typeof(TO.tensorcontract!),
    C::NestedTensor,
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    B::AbstractTensorMap,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple,
    α::Number,
    β::Number,
    ba...,
)
    function f(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)
        Ts = Zygote.Buffer(C.Ts)
        for i in eachindex(C.Ts)
            Ts[i] = C.Ts[i]
        end
        for i in 1:4
            Ts[i] = TO.tensorcontract!(
                Ts[i], A.Ts[i], pA, conjA, B, pB, conjB, pAB, α, β, ba...
            )
        end
        return NestedTensor(copy(Ts))
    end

    config = Zygote.ZygoteRuleConfig()
    projectors = (
        ProjectTo(C),
        ProjectTo(A),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        ntuple(_ -> nothing, length(ba))...,
    )
    return _rrule_via_ad_project(
        config, f, projectors, C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...
    )
end

function ChainRulesCore.rrule(
    ::typeof(TO.tensorcontract!),
    C::NestedTensor,
    A::AbstractTensorMap,
    pA::Index2Tuple,
    conjA::Bool,
    B::NestedTensor,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple,
    α::Number,
    β::Number,
    ba...,
)
    function f(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)
        Ts = Zygote.Buffer(C.Ts)

        for i in eachindex(C.Ts)
            Ts[i] = C.Ts[i]
        end
        for i in 1:4
            Ts[i] = TO.tensorcontract!(
                Ts[i], A, pA, conjA, B[i], pB, conjB, pAB, α, β, ba...
            )
        end

        return NestedTensor(copy(Ts))
    end

    config = Zygote.ZygoteRuleConfig()
    projectors = (
        ProjectTo(C),
        nothing,
        nothing,
        nothing,
        ProjectTo(B),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        ntuple(_ -> nothing, length(ba))...,
    )
    return _rrule_via_ad_project(
        config, f, projectors, C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...
    )
end

function ChainRulesCore.rrule(
    ::typeof(TO.tensorcontract!),
    C::NestedTensor,
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    B::NestedTensor,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple,
    α::Number,
    β::Number,
    ba...,
)
    function f(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)
        # index order:(iC, iA, iB, β)
        rules = [
            (1, 1, 1, β),
            (2, 1, 2, β),
            (2, 2, 1, VectorInterface.One()),
            (3, 1, 3, β),
            (3, 3, 1, VectorInterface.One()),
            (4, 1, 4, β),
            (4, 4, 1, VectorInterface.One()),
            (4, 2, 3, VectorInterface.One()),
            (4, 3, 2, VectorInterface.One()),
        ]
        Ts = Zygote.Buffer(C.Ts)
        for i in eachindex(C.Ts)
            Ts[i] = C.Ts[i]
        end

        for (iC, iA, iB, β′) in rules
            Ts[iC] = TO.tensorcontract!(
                Ts[iC], A.Ts[iA], pA, conjA, B.Ts[iB], pB, conjB, pAB, α, β′, ba...
            )
        end

        return NestedTensor(copy(Ts))
    end

    config = Zygote.ZygoteRuleConfig()
    projectors = (
        ProjectTo(C),
        ProjectTo(A),
        nothing,
        nothing,
        ProjectTo(B),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        ntuple(_ -> nothing, length(ba))...,
    )
    return _rrule_via_ad_project(
        config, f, projectors, C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...
    )
end

function ChainRulesCore.rrule(
    ::typeof(TO.tensortrace!),
    C::NestedTensor,
    A::NestedTensor,
    p::Index2Tuple,
    q::Index2Tuple,
    conjA::Bool,
    α::Number,
    β::Number,
    ba...,
)
    function f(C, A, p, q, conjA, α, β, ba...)
        Ts = Zygote.Buffer(C.Ts)
        for i in eachindex(C.Ts)
            Ts[i] = C.Ts[i]
        end
        for i in 1:4
            Ts[i] = TO.tensortrace!(Ts[i], A.Ts[i], p, q, conjA, α, β, ba...)
        end
        return NestedTensor(copy(Ts))
    end

    config = Zygote.ZygoteRuleConfig()
    projectors = (
        ProjectTo(C),
        ProjectTo(A),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        ntuple(_ -> nothing, length(ba))...,
    )
    return _rrule_via_ad_project(config, f, projectors, C, A, p, q, conjA, α, β, ba...)
end

# function ChainRulesCore.rrule(
#     ::typeof(TO.tensoradd!),
#     C::NestedTensor,
#     A::NestedTensor,
#     pA::Index2Tuple,
#     conjA::Bool,
#     α::Number,
#     β::Number,
#     ba...,
# )
#     C′ = tensoradd!(copy(C), A, pA, conjA, α, β, backend, allocator)

#     pullbacks = map(1:4) do i
#         _, pb = ChainRulesCore.rrule(
#             TO.tensoradd!, C[i], A[i], pA, conjA, α, β, backend, allocator
#         )
#         return pb
#     end

#     function pullback(ΔC′)
#         ΔC = ΔC′
#         dC = Vector{Any}(undef, 4)
#         dA = Vector{Any}(undef, 4)
#         for i in 1:4
#             _, dCi, dAi, _, _, _, dα, dβ, _ = pullbacks[i](ΔC[i])
#             dC[i] = unthunk(dCi)
#             dA[i] = unthunk(dAi)
#         end

#         dα = NoTangent()
#         dβ = NoTangent()
#         dba = map(_ -> NoTangent(), ba)

#         return NoTangent(), dC, dA, NoTangent(), NoTangent(), dα, dβ, dba...
#     end
#     return C′, pullback
# end

# Nested * TensorMap
# function ChainRulesCore.rrule(
#     ::typeof(TO.tensorcontract!),
#     C::NestedTensor,
#     A::NestedTensor,
#     pA::Index2Tuple,
#     conjA::Bool,
#     B::AbstractTensorMap,
#     pB::Index2Tuple,
#     conjB::Bool,
#     pAB::Index2Tuple,
#     α::Number,
#     β::Number,
#     ba...,
# )
#     C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

#     pullbacks = map(1:4) do i
#         _, pb = ChainRulesCore.rrule(
#             TO.tensorcontract!, C[i], A[i], pA, conjA, B, pB, conjB, pAB, α, β, ba...
#         )
#         return pb
#     end

#     function pullback(ΔC′)
#         ΔC = unthunk(ΔC′)
#         dC = Vector{Any}(undef, 4)
#         dA = Vector{Any}(undef, 4)
#         dB = zero(B)
#         for i in 1:4
#             _, dCi, dAi, _, _, dBi, _ = pullbacks[i](ΔC[i])
#             dC[i] = unthunk(dCi)
#             dA[i] = unthunk(dAi)
#             dB += unthunk(dBi)
#         end

#         dα = NoTangent()
#         dβ = NoTangent()
#         dba = map(_ -> NoTangent(), ba)
#         return NoTangent(),
#         dC,
#         dA,
#         NoTangent(),
#         NoTangent(),
#         dB,
#         NoTangent(),
#         NoTangent(),
#         NoTangent(),
#         dα,
#         dβ,
#         dba...
#     end
#     return C′, pullback
# end

# TensorMap * Nested
# function ChainRulesCore.rrule(
#     ::typeof(TO.tensorcontract!),
#     C::NestedTensor,
#     A::AbstractTensorMap,
#     pA::Index2Tuple,
#     conjA::Bool,
#     B::NestedTensor,
#     pB::Index2Tuple,
#     conjB::Bool,
#     pAB::Index2Tuple,
#     α::Number,
#     β::Number,
#     ba...,
# )
#     C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

#     pullbacks = map(1:4) do i
#         _, pb = ChainRulesCore.rrule(
#             TO.tensorcontract!, C[i], A[i], pA, conjA, B[i], pB, conjB, pAB, α, β, ba...
#         )
#         return pb
#     end

#     function pullback(ΔC′)
#         ΔC = ΔC′
#         dC = Vector{Any}(undef, 4)
#         dA = zero(A)
#         dB = Vector{Any}(undef, 4)
#         for i in 1:4
#             _, dCi, dAi, _, _, dBi, _ = pullbacks[i](ΔC[i])
#             dC[i] = unthunk(dCi)
#             dA += unthunk(dAi)
#             dB[i] = unthunk(dBi)
#         end

#         dα = NoTangent()
#         dβ = NoTangent()
#         dba = map(_ -> NoTangent(), ba)
#         return NoTangent(),
#         dC,
#         dA,
#         NoTangent(),
#         NoTangent(),
#         dB,
#         NoTangent(),
#         NoTangent(),
#         NoTangent(),
#         dα,
#         dβ,
#         dba...
#     end
#     return C′, pullback
# end

# TensorMap * TensorMap
# function ChainRulesCore.rrule(
#     ::typeof(TO.tensorcontract!),
#     C::NestedTensor,
#     A::NestedTensor,
#     pA::Index2Tuple,
#     conjA::Bool,
#     B::NestedTensor,
#     pB::Index2Tuple,
#     conjB::Bool,
#     pAB::Index2Tuple,
#     α::Number,
#     β::Number,
#     ba...,
# )
#     C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

#     # index order:(iC, iA, iB, β)
#     rules = [
#         (1, 1, 1, β),
#         (2, 1, 2, β),
#         (2, 2, 1, VectorInterface.One()),
#         (3, 1, 3, β),
#         (3, 3, 1, VectorInterface.One()),
#         (4, 1, 4, β),
#         (4, 4, 1, VectorInterface.One()),
#         (4, 2, 3, VectorInterface.One()),
#         (4, 3, 2, VectorInterface.One()),
#     ]
#     pullbacks = map(1:9) do i
#         iC, iA, iB, β′ = rules[i]
#         _, pb = ChainRulesCore.rrule(
#             TO.tensorcontract!,
#             C[iC],
#             A[iA],
#             pA,
#             conjA,
#             B[iB],
#             pB,
#             conjB,
#             pAB,
#             α,
#             β′,
#             ba...,
#         )
#         return pb
#     end

#     function pullback(ΔC′)
#         ΔC = ΔC′
#         dA = Vector{Any}(undef, 4)
#         dB = Vector{Any}(undef, 4)

#         # i = 4
#         _, dC4, dA3_4, _, _, dB2_4, _ = pullbacks[9](ΔC[4])
#         _, dC4, dA2_4, _, _, dB3_4, _ = pullbacks[8](dC4)
#         _, dC4, dA4_4, _, _, dB1_4, _ = pullbacks[7](dC4)
#         _, dC4, dA1_4, _, _, dB4_4, _ = pullbacks[6](dC4)

#         # i = 3
#         _, dC3, dA3_3, _, _, dB1_3, _ = pullbacks[5](ΔC[3])
#         _, dC3, dA1_3, _, _, dB3_3, _ = pullbacks[4](dC3)

#         # i = 2
#         _, dC2, dA2_2, _, _, dB1_2, _ = pullbacks[3](ΔC[2])
#         _, dC2, dA1_2, _, _, dB2_2, _ = pullbacks[2](dC2)

#         # i = 1
#         _, dC1, dA1_1, _, _, dB1_1, _ = pullbacks[1](ΔC[1])

#         dC = [dC1, dC2, dC3, dC4]
#         dA = [dA1_1 + dA1_2 + dA1_3 + dA1_4, dA2_2 + dA2_4, dA3_3 + dA3_4, dA4_4]
#         dB = [dB1_1 + dB1_2 + dB1_3 + dB1_4, dB2_2 + dB2_4, dB3_3 + dB3_4, dB4_4]

#         dα = NoTangent()
#         dβ = NoTangent()
#         dba = map(_ -> NoTangent(), ba)
#         return NoTangent(),
#         dC,
#         dA,
#         NoTangent(),
#         NoTangent(),
#         dB,
#         NoTangent(),
#         NoTangent(),
#         NoTangent(),
#         dα,
#         dβ,
#         dba...
#     end
#     return C′, pullback
# end

# function ChainRulesCore.rrule(
#     ::typeof(TO.tensortrace!),
#     C::NestedTensor,
#     A::NestedTensor,
#     p::Index2Tuple,
#     q::Index2Tuple,
#     conjA::Bool,
#     α::Number,
#     β::Number,
#     ba...,
# )
#     C′ = tensortrace!(copy(C), A, p, q, conjA, α, β, ba...)

#     pullbacks = map(1:4) do i
#         _, pb = ChainRulesCore.rrule(TO.tensortrace!, C[i], A[i], p, q, conjA, α, β, ba...)
#         return pb
#     end

#     function pullback(ΔC′)
#         ΔC = ΔC′
#         dC = Vector{Any}(undef, 4)
#         dA = Vector{Any}(undef, 4)
#         for i in 1:4
#             _, dCi, dAi, _ = pullbacks[i](ΔC[i])
#             dC[i] = unthunk(dCi)
#             dA[i] = unthunk(dAi)
#         end

#         dα = NoTangent()
#         dβ = NoTangent()
#         dba = map(_ -> NoTangent(), ba)
#         return NoTangent(), dC, dA, NoTangent(), NoTangent(), NoTangent(), dα, dβ, dba...
#     end
#     return C′, pullback
# end