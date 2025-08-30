module FinancialTransformer

using Flux
using Flux: Dense, sigmoid
using ..Transformer: EncoderBlock

struct FinancialTransformer
    blocks::Vector{EncoderBlock}
    price_head::Dense
    direction_head::Dense
    vol_head::Dense
end
Flux.@functor FinancialTransformer

function FinancialTransformer(; d_model=256, n_layers=6, n_heads=8, ff_mult=4)
    blocks = [EncoderBlock(d_model; ff_mult=ff_mult, n_heads=n_heads) for _ in 1:n_layers]
    FinancialTransformer(
        blocks,
        Dense(d_model, 1),
        Dense(d_model, 1),
        Dense(d_model, 1),
    )
end

function (m::FinancialTransformer)(x::AbstractArray)
    y = Float32.(x)
    for blk in m.blocks
        y = blk(y)
    end

    last = @view y[:, end, :]   # (B, D)
    T = permutedims(last, (2, 1))   # (D, B)

    price      = m.price_head(T)
    direction  = sigmoid.(m.direction_head(T))
    volatility = Flux.softplus.(m.vol_head(T))

    return hcat(permutedims(price, (2, 1)),
                permutedims(direction, (2, 1)),
                permutedims(volatility, (2, 1)))
end

end # module
