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
