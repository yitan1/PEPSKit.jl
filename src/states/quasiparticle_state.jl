# struct InfiniteQPPEPS{T<:PEPSTensor} 
#     A::Matrix{NestedTensor{T}}
#     InfiniteQPPEPS{T}(A::Matrix{NestedTensor{T}}) where {T<:PEPSTensor} = new{T}(A)
#     function InfiniteQPPEPS(A::Array{NestedTensor{T},2}) where {T<:PEPSTensor}
#         for (d, w) in Tuple.(CartesianIndices(A))
#             north_virtualspace(A[d, w]) == south_virtualspace(A[_prev(d, end), w])' ||
#                 throw(
#                     SpaceMismatch("North virtual space at site $((d, w)) does not match.")
#                 )
#             east_virtualspace(A[d, w]) == west_virtualspace(A[d, _next(w, end)])' ||
#                 throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
#             dim(space(A[d, w])) > 0 || @warn "no fusion channels at site ($d, $w)"
#         end
#         return new{T}(A)
#     end
# end

# function InfiniteQPPEPS(A::InfinitePEPS, B::InfinitePEPS)
#     InfiniteQPPEPS(unitcell(A), unitcell(B))
# end

# function InfiniteQPPEPS(A::Array{T, 2}, B::Array{T, 2}) where {T<:PEPSTensor}
#     return InfiniteQPPEPS(map((a,b) -> NestedTensor([a, b, VectorInterface.zerovector(a), VectorInterface.zerovector(a)]), A, B))
# end

function qp_InfinitePEPS(A::InfinitePEPS, B::InfinitePEPS)
    qp_InfinitePEPS(unitcell(A), unitcell(B))
end

function qp_InfinitePEPS(A::Array{T, 2}, B::Array{T, 2}) where {T<:PEPSTensor}
    return InfinitePEPS(map((a,b) -> NestedTensor([a, b, VectorInterface.zerovector(a), VectorInterface.zerovector(a)]), A, B))
end

function InfiniteSquareNetwork(top::InfinitePEPS{T}, bot::InfinitePEPS{T}=top) where {T<:NestedTensor}
    size(top) == size(bot) || throw(
        ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have the same length")
    )
    return InfiniteSquareNetwork(map(tuple, unitcell(top), exchange_B_Bd(unitcell(bot))))
end

function exchange_B_Bd(A::Matrix{T}) where {T<:NestedTensor}
    return map(a -> NestedTensor([a[1], a[3], a[2], a[4]]), A)
end

# Network: NestedTensor -> A
function gs_Network(network::InfiniteSquareNetwork{<:Tuple{<:NestedTensor, <:NestedTensor}})
    new_network = map(a -> (a[1][1], a[2][1]), unitcell(network))
    return InfiniteSquareNetwork(new_network)
end
