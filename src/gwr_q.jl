export gwr_q


"Geographically weighted regression for single location"
function gw_reg(x::Matrix{T}, y::Vector{T}, w::AbstractVector{T})::Matrix{T} where {T<:Real}
  w_sqrt = sqrt.(w)
  xw = x .* w_sqrt # 重新分配内存
  yw = y .* w_sqrt

  # Solve weighted least squares
  β = try
    (xw' * xw) \ (xw' * yw)
  catch e
    # Handle singular matrix
    @warn "Singular matrix encountered, using pseudo-inverse"
    pinv(xw' * xw) * (xw' * yw)
  end
  return reshape(β, 1, :)  # Return as row vector
end


"""
GWR with specified distance matrix

# Arguments
- `x`: control variables, [n_control, k_local]
- `y`: response variable
- `dMat`: distance matrix, [n_control, n_target]

提前算好权重，进行加速
"""
function gwr_q!(β::AbstractMatrix{T}, x::Matrix{T}, y::Vector{T}, wMat::AbstractMatrix{T})::Matrix{T} where {T<:Real}
  k_local = size(x, 2)
  n_target = size(wMat, 2)
  # n_control = size(x, 1)
  # β = zeros(T, n_target, k_local)
  @inbounds for i in 1:n_target
    w = @view wMat[:, i]
    _β = gw_reg(x, y, w) # [1, k_local]
    β[i, :] = _β
  end
  return β
end


function gwr_q(x::Matrix{T}, y::Vector{T}, wMat::AbstractMatrix{T})::Matrix{T} where {T<:Real}
  n_target = size(wMat, 2)
  k_local = size(x, 2)
  β = zeros(T, n_target, k_local)
  gwr_q!(β, x, y, wMat)
end


#! deprecated, low efficiency
function gwr_q(x::Matrix{T}, y::Vector{T}, dMat::AbstractMatrix{T}, bw::T;
  kernel::Int, adaptive::Bool=false)::Matrix{T} where {T<:Real}

  n_control = size(x, 1)
  k_local = size(x, 2)
  n_target = size(dMat, 2)

  w = zeros(T, n_control)
  β = zeros(T, n_target, k_local)

  @inbounds for i in 1:n_target
    distv = dMat[:, i]
    gw_weight!(w, distv, bw; kernel, adaptive)
    β[i, :] = gw_reg(x, y, w)
  end
  return β
end
