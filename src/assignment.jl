using Printf


function _knapsack!(Q, M, c)
    W = eltype(Q)
    m = size(Q, 1)

    Q .= zero(W)
    for k in 1:m
        Q[k, partialsortperm(M[k, :], 1:c[k], rev=true)] .= one(W)
    end
end


function _saddle_assignment!(T, Q, A, V, c, Σ, Λ; maxit)
    lT = copy(T)

    MQ = -Λ' - Σ'
    MT = Λ - Σ + V
    S2 = 2 * Σ

    for _ in 1:maxit
        _knapsack!(Q, T' * A + MQ + S2' .* T', c)
        _assign!(T, A * Q' + MT + S2 .* Q')

        if all(lT .== T)
            break
        end

        lT .= T
    end

    T, Q
end


function _objective_assignment(T, A, V)
    sum((T * T') .* A) + sum(T .* V)
end


function gassign(A, V, c; maxit, tol=0.0, decay=0.999, lr=1.0, lrdecay=sqrt(0.999), disturb=1.0, ddecay=sqrt(0.999), pneg=true, bound=Inf)
    W = eltype(A)
    n, m = size(V)

    Λ = zeros(W, (n, m))
    Σ = zeros(W, (n, m))

    T = zeros(W, (n, m))
    Q = zeros(W, (m, n))

    Tbest = copy(T)
    best = -Inf

    M = _disturb(A, disturb)

    for it in 1:maxit
        # game changer (most times)
        if pneg
            _posneg!(M)
        end

        _saddle_assignment!(T, Q, M, V, c, Σ, Λ, maxit=10000)

        ∇ = Q' - T
        norm = sum(abs.(∇))

        if norm <= tol * n * m
            obj = _objective_assignment(T, A, V)
            if obj > best
                Tbest, best = copy(T), obj
                @printf("best = %5.3f\tlr = %5.3f\tdisturb = %5.3f\tit = %d\n",
                    best, lr, disturb, it)
                flush(stdout)
            end

            if obj >= bound
                break
            end

            # game changer (less sensitive to learning rate)
            Σ *= decay
            Λ *= decay
            lr *= lrdecay
            disturb *= ddecay

            # game changer (sometimes)
            M .= _disturb(A, disturb)
        end

        Σ += lr * abs.(∇)
        Λ += lr * ∇
    end

    Tbest
end