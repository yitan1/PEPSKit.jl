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
struct NestedTensor{C}
    Ts::Vector{C}
    function NestedTensor{C}(Ts::Vector{C}) where {C}
        @assert length(Ts) == 4 "NestedTensor must have 4 components"
        return new{C}(Ts)
    end
end

function NestedTensor(Ts::AbstractVector{C}) where {C}
    @assert length(Ts) == 4 "NestedTensor must have 4 components"
    return NestedTensor{C}(Ts)
end

# Convenience constructor from a single tensor
nested_single(a) = NestedTensor([copy(a) for _ in 1:4])
function nested_single0(a)
    zero_a = VectorInterface.zerovector(a)
    return NestedTensor([copy(a), zero_a, zero_a, zero_a])
end

Base.length(t::NestedTensor) = 4
Base.eltype(t::NestedTensor) = eltype(t.Ts[1])

Base.copy(t::NestedTensor) = NestedTensor(copy.(t.Ts))

Base.getindex(t::NestedTensor, i::Int) = t.Ts[i]
Base.setindex!(t::NestedTensor, v, i::Int) = (t.Ts[i] = v; t)
# Base.iterate(n::NestedTensor, args...) = iterate(n.Ts, args...)
# Base.eachindex(A::NestedTensor) = eachindex(A.data)

physicalspace(t::NestedTensor) = physicalspace(t[1])


function shift(t::NestedTensor, phi)
    eϕ = exp(im * phi)
    eϕ⁻ = exp(-im * phi)
    return NestedTensor([
        t[1],        # regular term unchanged
        t[2] * eϕ,  # B term picks up +φ
        t[3] * eϕ⁻,  # Bd term picks up -φ
        t[4],        # B·Bd term unchanged
    ])
end

# VectorInterface interface
function VectorInterface.scalartype(::Type{<:NestedTensor{C}}) where {C}
    return VectorInterface.scalartype(C)
end

function VectorInterface.scalartype(t::NestedTensor)
    return VectorInterface.scalartype(t.Ts[1])
end

function VectorInterface.zerovector(A::NestedTensor, ::Type{S}) where {S<:Number}
    return NestedTensor([VectorInterface.zerovector(t, S) for t in A.Ts])
end

Base.zero(A::NestedTensor) = VectorInterface.zerovector(A, VectorInterface.scalartype(A))

## Math
function Base.:+(A₁::NestedTensor, A₂::NestedTensor)
    return NestedTensor(A₁.Ts .+ A₂.Ts)
end
function Base.:-(A₁::NestedTensor, A₂::NestedTensor)
    return NestedTensor(A₁.Ts .- A₂.Ts)
end
Base.:*(α::Number, A::NestedTensor) = NestedTensor(α * A.Ts)
Base.:*(A::NestedTensor, α::Number) = α * A
Base.:/(A::NestedTensor, α::Number) = NestedTensor(A.Ts / α)
# # LinearAlgebra.dot(A₁::NestedTensor, A₂::NestedTensor) = dot(unitcell(A₁), unitcell(A₂))
function maxabs_element(t::AbstractTensorMap)
    mapreduce(max, blocks(t); init = zero(real(scalartype(t)))) do (_, blk)
        maximum(abs, blk)
    end
end
LinearAlgebra.norm(A::NestedTensor) = norm(A.Ts[1])

function Base.:*(A::NestedTensor, B::NestedTensor)
    t1 = A[1] * B[1]
    t2 = A[2] * B[1] + A[1] * B[2]
    t3 = A[3] * B[1] + A[1] * B[3]
    t4 = A[4] * B[1] + A[3] * B[2] + A[2] * B[3] + A[1] * B[4]
    return NestedTensor([t1, t2, t3, t4])
end

function Base.:/(A::NestedTensor{T}, B::NestedTensor{T}) where {T<:Number}
    t1 = A[1] / B[1]
    t2 = A[2] / B[1] + A[1] / B[2]
    t3 = A[3] / B[1] + A[1] / B[3]
    t4 = A[4] / B[1] + A[3] / B[2] + A[2] / B[3] + A[1] / B[4]
    return NestedTensor([t1, t2, t3, t4])
end

function LinearAlgebra.tr(A::NestedTensor)
    # return NestedTensor([tr(A.Ts[i]) for i in 1:4])
    return map(tr, A.Ts)
end

function Base.fill!(t::NestedTensor, v)
    for i in 1:4
        fill!(t.Ts[i], v[i])
    end
    return t
end

# Rotations 
Base.rotl90(t::NestedTensor) = NestedTensor(map(rotl90, t.Ts))
Base.rotr90(t::NestedTensor) = NestedTensor(map(rotr90, t.Ts))
Base.rot180(t::NestedTensor) = NestedTensor(map(rot180, t.Ts))

# TensorKit interface
TensorKit.space(t::NestedTensor) = space(t[1])
TensorKit.space(t::NestedTensor, dir::Int) = space(t[1], dir)

TensorKit.permute(t::NestedTensor, perm) = NestedTensor(map(t -> permute(t, perm), t.Ts))
function TensorKit.permute(t::NestedTensor, perm::Index2Tuple)
    return NestedTensor(map(t -> permute(t, perm), t.Ts))
end

""" From PEPSKit utility/util.jl
    twistdual(t::AbstractTensorMap, i)
    twistdual!(t::AbstractTensorMap, i)

Twist the i-th leg of a tensor `t` if it represents a dual space.
"""
function twistdual!(t::NestedTensor, i::Int)
    isdual(space(t[1], i)) || return t
    return NestedTensor([twist!(t, i) for t in t.Ts])
end
function twistdual!(t::NestedTensor, is)
    is′ = filter(i -> isdual(space(t[1], i)), is)
    return NestedTensor([twist!(t, is′) for t in t.Ts])
end
twistdual(t::NestedTensor, is) = twistdual!(copy(t), is)

"""
    str(t)

Fermionic supertrace by using `@tensor`.
"""
str(t::NestedTensor) = _str(BraidingStyle(sectortype(t[1])), t)
_str(::Bosonic, t::NestedTensor) = map(tr, t.Ts)
@generated function _str(::Fermionic, t::NestedTensor{<:AbstractTensorMap{<:Any, <:Any, N, N}}) where {N}
    tex = tensorexpr(:t, ntuple(identity, N), ntuple(identity, N))
    return macroexpand(@__MODULE__, :(@tensor $tex))
end

"""
    trmul(H, ρ)

Compute `tr(H * ρ)` without forming `H * ρ`.
"""
@generated function trmul(
        H::AbstractTensorMap{<:Any, S, N, N}, ρ::NestedTensor{<:AbstractTensorMap{<:Any, S, N, N}}
    ) where {S, N}
    Hex = tensorexpr(:H, ntuple(identity, N), ntuple(i -> i + N, N))
    ρex = tensorexpr(:ρ, ntuple(i -> i + N, N), ntuple(identity, N))
    return macroexpand(@__MODULE__, :(@tensor $Hex * $ρex))
end

# TensorOperations interface
# conj is not exchange the location of B and Bd

TO.tensorstructure(t::NestedTensor) = TO.tensorstructure(t[1])
function TO.tensorstructure(t::NestedTensor, iA::Int, conjA::Bool)
    return TO.tensorstructure(t[1], iA, conjA)
end

function TO.tensoralloc(
    ::Type{NT},
    structure::TensorMapSpace,
    istemp::Val,
    allocator=TO.DefaultAllocator(),
    # ) where {T,S,N₁,N₂,NT<:NestedTensor{T,S,N₁,N₂}}
) where {C<:AbstractTensorMap,NT<:NestedTensor{C}}
    Ts = [TO.tensoralloc(C, structure, istemp, allocator) for _ in 1:4]
    return NestedTensor(Ts)
end

function TO.tensorfree!(nt::NestedTensor, allocator=TO.DefaultAllocator())
    for T in nt.Ts
        TO.tensorfree!(T, allocator)
    end
    return nothing
end

TO.tensorscalar(t::NestedTensor) = NestedTensor(TO.tensorscalar.(t.Ts))

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
    return NestedTensor{M}
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
    for i in 1:4
        C.Ts[i] = TO.tensortrace!(C.Ts[i], A.Ts[i], p, q, conjA, α, β, backend, allocator)
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
            C[i], A[i], pA, conjA, B, pB, conjB, pAB, α, β, backend, allocator
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
            C[i], A, pA, conjA, B[i], pB, conjB, pAB, α, β, backend, allocator
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
    return NestedTensor{M}
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
    return NestedTensor{M}
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
    return NestedTensor{M}
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
