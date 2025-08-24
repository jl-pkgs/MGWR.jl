export GWR


"""
GWR with specified distance matrix

# Arguments
- `x`: control variables, [n_control, k_local]
- `y`: response variable
- `dMat`: distance matrix, [n_control, n_target]

提前算好权重，进行加速
"""
function GWR!(β::AbstractMatrix{T}, X::Matrix{T}, y::Vector{T}, wMat::AbstractMatrix{T})::Matrix{T} where {T<:Real}
  k_local = size(X, 2)
  n_target = size(wMat, 2)
  # n_control = size(x, 1)
  # β = zeros(T, n_target, k_local)
  @inbounds @threads for i in 1:n_target
    w = @view wMat[:, i]
    _β = solve_reg(X, y, w) # [1, k_local]
    β[i, :] = _β
  end
  return β
end


function GWR(x::Matrix{T}, y::Vector{T}, wMat::AbstractMatrix{T})::Matrix{T} where {T<:Real}
  n_target = size(wMat, 2)
  k_local = size(x, 2)
  β = zeros(T, n_target, k_local)
  GWR!(β, x, y, wMat)
end

function GWR(model::MGWR)
  (; x1, y, wMat_rp) = model
  GWR(x1, y, wMat_rp)
end

#! deprecated, low efficiency
function GWR(x::Matrix{T}, y::Vector{T}, dMat::AbstractMatrix{T}, bw::T;
  kernel::Int, adaptive::Bool=false)::Matrix{T} where {T<:Real}

  n_control = size(x, 1)
  k_local = size(x, 2)
  n_target = size(dMat, 2)

  w = zeros(T, n_control)
  β = zeros(T, n_target, k_local)

  @inbounds for i in 1:n_target
    distv = dMat[:, i]
    gw_weight!(w, distv, bw; kernel, adaptive)
    β[i, :] = solve_reg(x, y, w)
  end
  return β
end
