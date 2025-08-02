"Create unit vector (equivalent to e_vec in C++)"
function e_vec(m::Int, n::Int)::Vector{Float64}
  ret = zeros(Float64, n)
  ret[m] = 1.0
  return ret
end


"""
Calculate trace of hat matrix for Mixed GWR (for effective degrees of freedom)
"""
function gwr_mixed_trace(x1::Matrix{T}, x2::Matrix{T}, dMat::Matrix{T}, bw::T;
  kernel::Int=0, adaptive::Bool=false)::T where {T}

  n = size(x1, 1)
  k_global = size(x2, 2)

  # Initialize matrices
  x3 = zeros(Float64, n, k_global)
  hii = zeros(Float64, n)

  # Step 1: Orthogonalize global variables (same as in gwr_mixed_2)
  for i in 1:k_global
    mtemp = gwr_q(x1, x2[:, i], dMat, bw; kernel, adaptive)
    x3[:, i] = x2[:, i] - fitted(x1, mtemp)
  end

  # Step 2: Calculate diagonal elements of hat matrix
  # Note: This is computationally expensive but follows the C++ logic
  # ei = zeros(T, n)
  @inbounds for i in 1:n
    # Create unit vector for position i
    # ei[i] = 1.0
    ei = e_vec(i, n)

    # Regression of unit vector on local variables
    mtemp = gwr_q(x1, ei, dMat, bw; kernel, adaptive)
    y2 = ei - fitted(x1, mtemp)

    # Global regression
    model2 = gwr_q(x3, y2, dMat, 100000.0; kernel=BOXCAR, adaptive=true)
    y3 = ei - fitted(x2, model2)

    # Local regression at position i only
    dMat_i = @view dMat[:, i:i] ## TODO: debug here
    model1 = gwr_q(x1, y3, dMat_i, bw; kernel, adaptive)
    model2_i = gwr_q(x3, y2, dMat_i, 100000.0; kernel=BOXCAR, adaptive=true)

    # Calculate hat matrix diagonal elements
    s1 = fitted(x1[i:i, :], model1)[1]
    s2 = fitted(x2[i:i, :], model2_i)[1]
    
    hii[i] = s1 + s2
  end
  return sum(hii)
end
