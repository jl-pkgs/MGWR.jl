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
function gwr_q(x::Matrix{Float64}, y::Vector{Float64},
  dMat::Matrix{Float64}, bw::Float64, kernel::Int, adaptive::Bool=false)::Matrix{Float64}
  n = size(dMat, 2)
  m = size(x, 2)
  β = zeros(Float64, n, m)

  for i in 1:n
    distv = dMat[:, i]
    w = gw_weight(distv, bw, kernel, adaptive)
    β[i, :] = gw_reg(x, y, w)
  end
  return β
end


"Fitted values from coefficient matrix and design matrix"
function fitted(x::Matrix{Float64}, beta::Matrix{Float64})::Vector{Float64}
  return vec(sum(x .* beta, dims=2))
end
