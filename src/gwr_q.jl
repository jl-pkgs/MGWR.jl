"Geographically weighted regression for single location"
function gw_reg(x::Matrix{Float64}, y::Vector{Float64}, w::Vector{Float64})::Matrix{Float64}
  # Create weighted design matrix
  w_sqrt = sqrt.(w)
  xw = x .* w_sqrt
  yw = y .* w_sqrt

  # Solve weighted least squares
  beta = try
    (xw' * xw) \ (xw' * yw)
  catch e
    # Handle singular matrix
    @warn "Singular matrix encountered, using pseudo-inverse"
    pinv(xw' * xw) * (xw' * yw)
  end
  return reshape(beta, 1, :)  # Return as row vector
end

"GWR with specified distance matrix"
function gwr_q(x::Matrix{T}, y::Vector{T}, dMat::AbstractMatrix{T}, bw::T;
  kernel::Int, adaptive::Bool=false)::Matrix{T} where {T<:Real}

  N = size(x, 1)    # observations
  w = zeros(T, N)

  m = size(x, 2)    # n_local variables
  n = size(dMat, 2)
  β = zeros(T, n, m)

  @inbounds for i in 1:n
    distv = dMat[:, i]
    gw_weight!(w, distv, bw; kernel, adaptive)
    β[i, :] = gw_reg(x, y, w)
  end
  return β
end


"Fitted values from coefficient matrix and design matrix"
function fitted(x::Matrix{Float64}, beta::Matrix{Float64})::Vector{Float64}
  return vec(sum(x .* beta, dims=2))
end
