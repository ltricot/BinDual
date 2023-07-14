using Printf


function _select!(Q, M)
    W = eltype(Q)
    Q .= ifelse.(M .> 0.0, one(W), zero(W))
end


function _saddle_partition!(T, Q, A, Σ, Λ; maxit)
    lT = copy(T)

    MQ = -Λ' - Σ'
    MT = Λ - Σ
    S2 = 2 * Σ

    for _ in 1:maxit
        _select!(Q, T' * A + MQ + S2' .* T')
        _assign!(T, A * Q' + MT + S2 .* Q')

        if all(lT .== T)
            break
        end

        lT .= T
    end

    T, Q
end


function _objective_partition(T, A)
    sum((T * T') .* A)
end


function partition(A, m; maxit, tol=0.0, decay=0.999, lr=1.0, lrdecay=sqrt(0.999), disturb=1.0, ddecay=sqrt(0.999), pneg=true, bound=Inf)
    W = eltype(A)
    n = size(A, 1)

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

        _saddle_partition!(T, Q, M, Σ, Λ, maxit=100)

        ∇ = Q' - T
        norm = sum(abs.(∇))

        if norm <= tol * n * m
            obj = _objective_partition(T, A)
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
            if disturb >= 0
                M .= _disturb(A, disturb)
            end
        end

        Σ += lr * abs.(∇)
        Λ += lr * ∇
    end

    Tbest
end