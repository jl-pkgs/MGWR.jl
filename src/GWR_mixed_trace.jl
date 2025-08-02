"Create unit vector (equivalent to e_vec in C++)"
function e_vec(m::Int, n::Int)::Vector{Float64}
  ret = zeros(Float64, n)
  ret[m] = 1.0
  return ret
end


"""
Calculate trace of hat matrix for Mixed GWR (for effective degrees of freedom)
"""
function GWR_mixed_trace(x1::Matrix{T}, x2::Matrix{T}, dMat::Matrix{T}, bw::T;
  kernel::Int=0, adaptive::Bool=false)::T where {T}

  n_control = size(x1, 1)
  p_global = size(x2, 2)

  # Initialize matrices
  x3 = zeros(T, n_control, p_global)

  wMat = gw_weight(dMat, bw; kernel, adaptive)
  wMat_ols = gw_weight(dMat, 100000.0; kernel=BOXCAR, adaptive=true)

  # Step 1: Orthogonalize global variables (same as in gwr_mixed_2)
  for i in 1:p_global
    # β = GWR(x1, x2[:, i], dMat, bw; kernel, adaptive) # [n_control, k_local]
    β = GWR(x1, x2[:, i], wMat) # [n_control, k_local]
    x3[:, i] = x2[:, i] - fitted(x1, β) # [n_control, 1], global - local_effect
  end

  # Step 2: Calculate diagonal elements of hat matrix
  # Note: This is computationally expensive but follows the C++ logic
  # ei = zeros(T, n)
  hii = zeros(T, n_control)
  @inbounds for i in 1:n_control
    # Create unit vector for position i
    # ei[i] = 1.0
    ei = e_vec(i, n_control)

    # Regression of unit vector on local variables
    β = GWR(x1, ei, wMat) # [n_control, k_local]
    y2 = ei - fitted(x1, β) # [n_control, 1]

    # Global regression
    β2 = GWR(x3, y2, wMat_ols)
    y3 = ei - fitted(x2, β2)

    # Local regression at position i only
    β1 = GWR(x1, y3, wMat[:, i:i])
    β2_i = GWR(x3, y2, wMat_ols[:, i:i])

    # Calculate hat matrix diagonal elements
    s1 = fitted(x1[i:i, :], β1)[1]
    s2 = fitted(x2[i:i, :], β2_i)[1]
    hii[i] = s1 + s2
  end
  return sum(hii)
end


function GWR_mixed_trace(model::MGWR{T}) where {T}
  (; x1, x2, x3, p_global, n_control, wMat, wMat_ols) = model
  x3 .= T(0)
  
  # Step 1: Orthogonalize global variables (same as in gwr_mixed_2)
  for i in 1:p_global
    β = GWR(x1, x2[:, i], wMat) # [n_control, k_local]
    x3[:, i] = x2[:, i] - fitted(x1, β) # [n_control, 1], global - local_effect
  end

  # Step 2: Calculate diagonal elements of hat matrix
  # Note: This is computationally expensive but follows the C++ logic
  # ei = zeros(T, n)
  hii = zeros(T, n_control)
  @inbounds for i in 1:n_control
    ei = e_vec(i, n_control) # unit vector for position i

    # Regression of unit vector on local variables
    β = GWR(x1, ei, wMat) # [n_control, k_local]
    y2 = ei - fitted(x1, β) # [n_control, 1]

    # Global regression
    β2 = GWR(x3, y2, wMat_ols)
    y3 = ei - fitted(x2, β2)

    # Local regression at position i only
    β1 = GWR(x1, y3, wMat[:, i:i])
    β2_i = GWR(x3, y2, wMat_ols[:, i:i])

    # Calculate hat matrix diagonal elements
    s1 = fitted(x1[i:i, :], β1)[1]
    s2 = fitted(x2[i:i, :], β2_i)[1]
    hii[i] = s1 + s2
  end
  return sum(hii)
end
