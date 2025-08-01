module MixedGWR

using LinearAlgebra
using Statistics


include("kernel.jl")
include("gwr_q.jl")
include("gwr_mixed.jl")
include("gwr_mixed_trace.jl")


function predict(model, x1, x2)
  β_local = model[:local]
  β_global = model[:global]
  N = size(x1, 1)
  R = zeros(N)

  for i in 1:N
    _β_local = β_local[i, :]
    _β_global = β_global[i, :]
    _x1 = x1[i, :]'
    _x2 = x2[i, :]'
    R[i] = _x1 * _β_local + _x2 * _β_global
  end
  R
end

export gwr_mixed, gwr_mixed_trace, gwr_q, gw_weight_vec, gw_reg, fitted
export predict

end # module MGWR
