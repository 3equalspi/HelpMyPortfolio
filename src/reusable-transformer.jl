module Transformer
using Flux
using Flux: Dense, gelu, softmax, LayerNorm

function apply_dense_time(d::Dense, x_bsd::AbstractArray)
    B, S, Din = size(x_bsd)
    X = reshape(permutedims(x_bsd, (3, 2, 1)), Din, S * B)
    Y = d(X)
    Dout = size(Y, 1)
    return permutedims(reshape(Y, (Dout, S, B)), (3, 2, 1))
end

function apply_layernorm_time(ln::LayerNorm, x_bsd::AbstractArray)
    B, S, D = size(x_bsd)
    X = reshape(permutedims(x_bsd, (3, 2, 1)), D, S * B)
    Y = ln(X)
    return permutedims(reshape(Y, (D, S, B)), (3, 2, 1))
end

struct MyMHA
    Wq::Dense
    Wk::Dense
    Wv::Dense
    Wo::Dense
    n_heads::Int
    head_dim::Int
end
Flux.@functor MyMHA

function MyMHA(d_model::Int, n_heads::Int)
    @assert d_model % n_heads == 0
    hd = div(d_model, n_heads)
    MyMHA(Dense(d_model, d_model), Dense(d_model, d_model),
          Dense(d_model, d_model), Dense(d_model, d_model), n_heads, hd)
end

function (m::MyMHA)(x::AbstractArray)  # (B, S, D)
    q = apply_dense_time(m.Wq, x)
    k = apply_dense_time(m.Wk, x)
    v = apply_dense_time(m.Wv, x)

    B, S, D = size(q)
    H, hd = m.n_heads, m.head_dim

    Q = permutedims(reshape(q, B, S, H, hd), (1, 3, 2, 4))
    K = permutedims(reshape(k, B, S, H, hd), (1, 3, 2, 4))
    V = permutedims(reshape(v, B, S, H, hd), (1, 3, 2, 4))

    Z = Array{eltype(x)}(undef, B, S, H * hd)
    inv_sqrt_hd = 1f0 / sqrt(Float32(hd))

    @inbounds for b in 1:B
        for h in 1:H
            Qi = reshape(@view(Q[b, h, :, :]), S, hd)
            Ki = reshape(@view(K[b, h, :, :]), S, hd)
            Vi = reshape(@view(V[b, h, :, :]), S, hd)

            scores = (Qi * Ki') .* inv_sqrt_hd
            A = softmax(scores; dims=2)
            Zi = A * Vi

            @view(Z[b, :, (h-1)*hd + 1:h*hd]) .= Zi
        end
    end

    return apply_dense_time(m.Wo, Z)
end

struct EncoderBlock
    mha::MyMHA
    ln1::LayerNorm
    ff1::Dense
    ff2::Dense
    ln2::LayerNorm
end
Flux.@functor EncoderBlock

function EncoderBlock(d_model::Int; ff_mult::Int=4, n_heads::Int=8)
    EncoderBlock(
        MyMHA(d_model, n_heads),
        LayerNorm(d_model),
        Dense(d_model, ff_mult * d_model),
        Dense(ff_mult * d_model, d_model),
        LayerNorm(d_model),
    )
end

function (eb::EncoderBlock)(x::AbstractArray)
    y = x .+ eb.mha(x)
    y = apply_layernorm_time(eb.ln1, y)
    h = gelu.(apply_dense_time(eb.ff1, y))
    h = apply_dense_time(eb.ff2, h)
    z = y .+ h
    return apply_layernorm_time(eb.ln2, z)
end

end #model
