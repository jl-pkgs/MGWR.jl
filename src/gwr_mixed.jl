"Mixed GWR implementation"
function gwr_mixed(x1::Matrix{Float64}, x2::Matrix{Float64}, y::Vector{Float64},
  dMat::Matrix{Float64}, dMat_rp::Matrix{Float64}, bw::Float64;
  kernel::Int=0, adaptive::Bool=false)

  n = size(x1, 1)
  nc2 = size(x2, 2)
  x3 = zeros(Float64, n, nc2)

  # Step 1: Orthogonalize global variables (calculate x3)
  for i in 1:nc2
    beta = gwr_q(x1, x2[:, i], dMat, bw; kernel, adaptive)
    x3[:, i] = x2[:, i] - fitted(x1, beta)
  end

  # Step 2: Fit local part and get residuals
  mtemp = gwr_q(x1, y, dMat, bw; kernel, adaptive)  # Fit local
  y2 = y - fitted(x1, mtemp)                        # y - local = global

  # Step 3: Fit global coefficients (first time)
  model2 = gwr_q(x3, y2, dMat, 100000.0; kernel=BOXCAR, adaptive=true)  # Fit global

  # Step 4: Re-fit local coefficients removing global effects
  model1 = gwr_q(x1, y - fitted(x2, model2), dMat_rp, bw; kernel, adaptive)  # Remove global, fit local

  # Step 5: Final global coefficients
  model2 = gwr_q(x3, y2, dMat_rp, 100000.0; kernel=BOXCAR, adaptive=true)  # Final global fit
  (; :local => model1, :global => model2)
  # Dict("local" => model1, "global" => model2)
end
