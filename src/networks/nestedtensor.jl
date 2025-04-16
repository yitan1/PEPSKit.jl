# using LinearAlgebra
# using TensorKit
# using TensorKit: VectorInterface
# using TensorOperations
const TO = TensorOperations
"""
A Nested tensor contains the following variants (some may be empty):

    - :attr:`tensors[1]`: regular tensor (no B or Bd)
    - :attr:`tensors[2]`: (terms with) a single B tensor
    - :attr:`tensors[3]`: (terms with) a single Bd tensor
    - :attr:`tensors[4]`: (terms with) both a B and a Bd tensor

When two Nested tensors x,y are contracted, all combinations are taken into account
and the result is again a Nested tensor, filled with the following variants:

    - :attr:`tensors[1]: x[1] * y[1]`
    - :attr:`tensors[2]: x[2] * y[1] + x[1] * y[2]`
    - :attr:`tensors[3]: x[3] * y[1] + x[1] * y[3]`
    - :attr:`tensors[4]: x[4] * y[1] + x[3] * y[2] + x[2] * y[3] + x[1] * y[4]`

By using Nested tensors in a (large) contraction, the many different terms are
resummed on the fly, leading to a potentially reduced computational cost

Note:
    Most implented functions act as wrappers for the corresponding `numpy` functions
    on the individual tensors
"""
struct NestedTensor{T,S,N1,N2,C<:AbstractTensorMap{T,S,N1,N2}} <:
       AbstractTensorMap{T,S,N1,N2}
    Ts::Vector{C}
    function NestedTensor(Ts::Vector{C}) where {T,S,N1,N2,C<:AbstractTensorMap{T,S,N1,N2}}
        @assert length(Ts) == 4 "NestedTensor must have 4 components"
        return new{T,S,N1,N2,C}(Ts)
    end
end

# Convenience constructor from a single tensor
NestedTensor(a) = NestedTensor([copy(a) for _ in 1:4])

Base.length(t::NestedTensor) = 4
Base.eltype(t::NestedTensor) = eltype(t.Ts[1])

Base.copy(t::NestedTensor) = NestedTensor(copy.(t.Ts))

Base.getindex(t::NestedTensor, i::Int) = t.Ts[i]
Base.setindex!(t::NestedTensor, i::Int, v) = t.Ts[i] = v

function shift(t::NestedTensor, phi)
    eϕ = exp(im*phi)
    eϕ⁻ = exp(-im*phi)
    return NestedTensor([
        t[1],        # regular term unchanged
        t[2] * eϕ,  # B term picks up +φ
        t[3] * eϕ⁻,  # Bd term picks up -φ
        t[4],        # B·Bd term unchanged
    ])
end

# VectorInterface interface
TensorKit.VectorInterface.scalartype(t::NestedTensor) = VectorInterface.scalartype(t[1])

## Math
# function Base.:+(A₁::NestedTensor, A₂::NestedTensor)
#     return NestedTensor(A₁.Ts + A₂.Ts)
# end
# function Base.:-(A₁::NestedTensor, A₂::NestedTensor)
#     return NestedTensor(A₁.Ts - A₂.Ts)
# end
Base.:*(α::Number, A::NestedTensor) = NestedTensor(α * A.Ts)
Base.:*(A::NestedTensor, α::Number) = α * A
Base.:/(A::NestedTensor, α::Number) = NestedTensor(A.Ts / α)
# # LinearAlgebra.dot(A₁::NestedTensor, A₂::NestedTensor) = dot(unitcell(A₁), unitcell(A₂))
LinearAlgebra.norm(A::NestedTensor) = norm(A.Ts[1])

function Base.:*(A::NestedTensor, B::NestedTensor) 
    t1 = A[1] * B[1]
    t2 = A[2] * B[1] + A[1] * B[2]
    t3 = A[3] * B[1] + A[1] * B[3]
    t4 = A[4] * B[1] + A[3] * B[2] + A[2] * B[3] + A[1] * B[4]
    return NestedTensor([t1, t2, t3, t4])
end

function LinearAlgebra.tr(A::NestedTensor)
    return [tr(A.Ts[i]) for i in 1:4]
end

# Rotations 
Base.rotl90(t::NestedTensor) = NestedTensor(map(rotl90, t.Ts))
Base.rotr90(t::NestedTensor) = NestedTensor(map(rotr90, t.Ts))
Base.rot180(t::NestedTensor) = NestedTensor(map(rot180, t.Ts))




TO.tensorstructure(t::NestedTensor) = TO.tensorstructure(t[1])
function TO.tensorstructure(t::NestedTensor, iA::Int, conjA::Bool)
    return TO.tensorstructure(t[1], iA, conjA)
end

function TO.tensoralloc(
    ::Type{NT},
    structure::TensorMapSpace{S,N₁,N₂},
    istemp::Val,
    allocator=TO.DefaultAllocator(),
) where {T,S,N₁,N₂,NT<:NestedTensor{T,S,N₁,N₂}}
    Ts = [TO.tensoralloc(TensorMap{T}, structure, istemp, allocator) for _ in 1:4]
    return NestedTensor(Ts)
end

function TO.tensorfree!(nt::NestedTensor, allocator=TO.DefaultAllocator())
    for T in nt.Ts
        TO.tensorfree!(T, allocator)
    end
    return nothing
end

TO.tensorscalar(t::NestedTensor) = scalar.(t.Ts)

# tensoradd!
function TO.tensoradd!(
    C::NestedTensor,
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    α::Number,
    β::Number,
    backend,
    allocator,
)
    for i in eachindex(C.Ts)
        TO.tensoradd!(C.Ts[i], A.Ts[i], pA, conjA, α, β, backend, allocator)
    end

    return C
end

# return the first parameter of tensoralloc(ttype, structure) 
function TO.tensoradd_type(
    TC, A::NestedTensor, pA::Index2Tuple{N₁,N₂}, conjA::Bool
) where {N₁,N₂}
    M = TO.tensoradd_type(TC, A[1], pA, conjA)
    return NestedTensor{eltype(M),spacetype(M),numout(M),numin(M),M}
end

# return the second parameter of tensoralloc(ttype, structure)
function TO.tensoradd_structure(
    A::NestedTensor, pA::Index2Tuple{N₁,N₂}, conjA::Bool
) where {N₁,N₂}
    return TO.tensoradd_structure(A[1], pA, conjA)
end

function TO.tensortrace!(
    C::NestedTensor,
    A::NestedTensor,
    p::Index2Tuple,
    q::Index2Tuple,
    conjA::Bool,
    α::Number,
    β::Number,
    backend,
    allocator,
)
    for i in eachindex(C.Ts)
        TO.tensortrace!(C.Ts[i], A.Ts[i], p, q, conjA, α, β, backend, allocator)
    end

    return C
end

function TO.tensorcontract!(
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
    backend,
    allocator,
)
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
    for (iC, iA, iB, β′) in rules
        TO.tensorcontract!(
            C.Ts[iC],
            A.Ts[iA],
            pA,
            conjA,
            B.Ts[iB],
            pB,
            conjB,
            pAB,
            α,
            β′,
            backend,
            allocator,
        )
    end

    return C
end

function TO.tensorcontract!(
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
    backend,
    allocator,
)
    for i in 1:4
        TO.tensorcontract!(
            C[i],
            A[i],
            pA,
            conjA,
            B,
            pB,
            conjB,
            pAB,
            α,
            β,
            backend,
            allocator,
        )
    end
    return C
end

function TO.tensorcontract!(
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
    backend,
    allocator,
)
    for i in 1:4
        TO.tensorcontract!(
            C[i],
            A,
            pA,
            conjA,
            B[i],
            pB,
            conjB,
            pAB,
            α,
            β,
            backend,
            allocator,
        )
    end
    return C
end

# return the first parameter of tensoralloc(ttype, structure) 
function TO.tensorcontract_type(
    TC,
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    B::NestedTensor,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    M = TO.tensorcontract_type(TC, A[1], pA, conjA, B[1], pB, conjB, pAB)
    return NestedTensor{eltype(M),spacetype(M),numout(M),numin(M),M}
end

function TO.tensorcontract_type(
    TC,
    A::AbstractTensorMap,
    pA::Index2Tuple,
    conjA::Bool,
    B::NestedTensor,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    M = TO.tensorcontract_type(TC, A, pA, conjA, B[1], pB, conjB, pAB)
    return NestedTensor{eltype(M), spacetype(M), numout(M), numin(M), M}
end

function TO.tensorcontract_type(
    TC,
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    B::AbstractTensorMap,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    M = TO.tensorcontract_type(TC, A[1], pA, conjA, B, pB, conjB, pAB)
    return NestedTensor{eltype(M), spacetype(M), numout(M), numin(M), M}
end

# return the second parameter of tensoralloc(ttype, structure)
function TO.tensorcontract_structure(
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    B::NestedTensor,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    return TO.tensorcontract_structure(A[1], pA, conjA, B[1], pB, conjB, pAB)
end

function TO.tensorcontract_structure(
    A::AbstractTensorMap,
    pA::Index2Tuple,
    conjA::Bool,
    B::NestedTensor,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    return TO.tensorcontract_structure(A, pA, conjA, B[1], pB, conjB, pAB)
end

function TO.tensorcontract_structure(
    A::NestedTensor,
    pA::Index2Tuple,
    conjA::Bool,
    B::AbstractTensorMap,
    pB::Index2Tuple,
    conjB::Bool,
    pAB::Index2Tuple{N₁,N₂},
) where {N₁,N₂}
    return TO.tensorcontract_structure(A[1], pA, conjA, B, pB, conjB, pAB)
end


# PEPSTensor interface
# const PEPSTensorLike = PEPSTensor
# Base.rotl90(t::PEPSTensor) = permute(t, ((1,), (3, 4, 5, 2)))
# Base.rotr90(t::PEPSTensor) = permute(t, ((1,), (5, 2, 3, 4)))
# Base.rot180(t::PEPSTensor) = permute(t, ((1,), (4, 5, 2, 3)))

physicalspace(t::NestedTensor) = space(t[1], 1)
virtualspace(t::NestedTensor, dir) = space(t[1], dir + 1)

TensorKit.space(t::NestedTensor) = space(t[1])
