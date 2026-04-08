using PEPSKit
using PEPSKit: NestedTensor
import TensorOperations as TO
using Zygote

# 1. operations for nestedtensor
# For number
t = rand(4)
T = NestedTensor(t)

# For Array
t = randn(2, 2)
T = NestedTensor([t, t, t, t])

# For TensorMap
t = rand(ℂ^2 * ℂ^3 * ℂ^4)
T = NestedTensor([t, t, t, t])

# test contraction and rrule
function test_alloc()
    t1 = rand(ComplexF64, ℂ^2, ℂ^2)
    T = TO.tensoralloc(
        NestedTensor{TensorMap{ComplexF64}}, TO.tensorstructure(t1), Val(false)
    )
    @assert TO.tensorstructure(T) == TO.tensorstructure(t1)
    @assert isa(T, NestedTensor)
    @assert isa(T[1], TensorMap{ComplexF64})
end

function test_c()
    t1 = rand(ℂ^10 * ℂ^5 * ℂ^3)
    T1 = NestedTensor([t1, t1, t1, t1])

    t2 = rand((ℂ^3)', ℂ^5 * ℂ^8)
    T2 = NestedTensor([t2, t2, t2, t2])

    t3 = rand(ℂ^8, (ℂ^10))
    T3 = NestedTensor([t3, t3, t3, t3])

    @time @tensor C1[:] := t1[-1 2 1] * t2[1 2 3] * t3[3 -2]
    @time @tensor C2[:] := t1[-1 2 1] * T2[1 2 3] * T3[3 -2]

    @time @tensor res = T1[4 2 1] * T2[1 2 3] * T3[3 4]
    return println("res = ", res)
end

test_c()


# AD test for nestedtensor
function test_ad(t1, t2, t3)
    T1 = NestedTensor([t1, t1, t1, t1])
    T2 = NestedTensor([t2, t2, t2, t2])
    T3 = NestedTensor([t3, t3, t3, t3])
    # @time @tensor C1[:] := t1[-1 2 1] * t2[1 2 3] * t3[3 -2]
    # @time @tensor C2[:] := t1[-1 2 1] * T2[1 2 3] * T3[3 -2]

    @tensor res = T1[4 2 1] * T2[1 2 3] * T3[3 4]
    println("res = ", res)
    return res[1]
end

t1 = rand(ℂ^10 * ℂ^5 * ℂ^3)
t2 = rand((ℂ^3)', ℂ^5 * ℂ^8)
t3 = rand(ℂ^8, (ℂ^10))
test_ad(t1, t2, t3)
gradient(t -> test_ad(t, t2, t3), t1)


function f1(t1)
    T = NestedTensor([t1, t2, t2, t2])
    res = TO.tensortrace(T, ((), ()), ((1,), (2,)), false)
    return TO.tensorscalar(res)[1]
end
t1 = rand(ℂ^5, ℂ^5)
t2 = rand(ℂ^5, ℂ^5)
T1 = NestedTensor([t1, t1, t1, t1])
f1(t1)
withgradient(f1, t1)