"""
GWR with specified distance matrix

# Arguments
- `x`: control variables, [n_control, k_local]
- `y`: response variable
- `dMat`: distance matrix, [n_control, n_target]
- `β`: [n_target, k_local]
提前算好权重，进行加速
"""
function GWR!(β::AbstractMatrix{T}, 
  X::AbstractMatrix{T}, Y::AbstractVector{T},
  wMat::AbstractMatrix{T})::Matrix{T} where {T<:Real}
  # k_local = size(X, 2)
  # n_control = size(x, 1)
  # β = zeros(T, n_target, k_local)
  n_target = size(wMat, 2)
  @inbounds @threads for i in 1:n_target
    w = @view wMat[:, i]
    _β = solve_reg(X, Y, w) # [1, k_local]
    β[i, :] = _β
  end
  return β
end

# AbstractVecOrMat
function GWR(X::AbstractMatrix{T}, Y::AbstractVector{T},
  wMat::AbstractMatrix{T})::Matrix{T} where {T<:Real}
  n_target = size(wMat, 2)
  k_local = size(X, 2)
  β = zeros(T, n_target, k_local)
  GWR!(β, X, Y, wMat)
end

function GWR(model::MGWR)
  (; x1, y, wMat_rp) = model
  GWR(x1, y, wMat_rp)
end
