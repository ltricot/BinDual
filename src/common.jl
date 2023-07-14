using LinearAlgebra


function _disturb(A, λ)
    n = size(A, 1)
    D = UpperTriangular(randn(n, n))
    A + λ * (D - D')
end


function _posneg!(A)
    n = size(A, 1)

    for _ in 1:5
        i = 1 + mod(rand(Int), n)
        for j in 1:n
            A[j, i] += A[i, j]
            A[i, j] = 0
        end
    end
end


function _assign!(T, M)
    W = eltype(T)

    T .= zero(W)
    _, indices = findmax(M, dims=2)
    T[indices] .= one(W)
end