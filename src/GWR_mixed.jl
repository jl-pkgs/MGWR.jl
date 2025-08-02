"""
Mixed GWR implementation

# Arguments
- `dMat`   : with the dimension of [n_control, n_control]
- `dMat_rp`: with the dimension of [n_control, n_target]
"""
function GWR_mixed(x1::Matrix{T}, x2::Matrix{T}, y::Vector{T},
  dMat::Matrix{T}, dMat_rp::Matrix{T}, bw::T;
  kernel::Int=0, adaptive::Bool=false) where {T<:Real}

  n_control = size(x1, 1)
  p_global = size(x2, 2)

  wMat = gw_weight(dMat, bw; kernel, adaptive)
  wMat_ols = gw_weight(dMat, 100000.0; kernel=BOXCAR, adaptive=true)

  wMat_rp = gw_weight(dMat_rp, bw; kernel, adaptive)
  wMat_rp_ols = gw_weight(dMat_rp, 100000.0; kernel=BOXCAR, adaptive=true)


  # Step 1: Orthogonalize global variables (calculate x3)
  x3 = zeros(T, n_control, p_global)
  @inbounds for i in 1:p_global
    β = GWR(x1, x2[:, i], wMat)
    x3[:, i] = x2[:, i] - fitted(x1, β)
  end

  # Step 2: Fit local part and get residuals
  β = GWR(x1, y, wMat)  # Fit local
  y2 = y - fitted(x1, β)                        # y - local = global

  # Step 3: Fit global coefficients (first time)
  β_global = GWR(x3, y2, wMat_ols)  # Fit global

  # Step 4: Re-fit local coefficients removing global effects
  β_local = GWR(x1, y - fitted(x2, β_global), wMat_rp)  # Remove global, fit local

  # Step 5: Final global coefficients
  β_global = GWR(x3, y2, wMat_rp_ols)  # Final global fit
  (; :local => β_local, :global => β_global)
  # return β_local, β_global
end


function GWR_mixed(model::MGWR{T}) where {T}
  (; x1, x2, x3, y, wMat, wMat_ols, wMat_rp, wMat_rp_ols, n_control, p_global) = model
  x3 .= T(0)
  # Step 1: Orthogonalize global variables (calculate x3)
  x3 = zeros(T, n_control, p_global)
  @inbounds for i in 1:p_global
    β = GWR(x1, x2[:, i], wMat)
    x3[:, i] = x2[:, i] - fitted(x1, β)
  end

  # Step 2: Fit local part and get residuals
  β = GWR(x1, y, wMat)  # Fit local
  y2 = y - fitted(x1, β)                        # y - local = global

  # Step 3: Fit global coefficients (first time)
  β_global = GWR(x3, y2, wMat_ols)  # Fit global

  # Step 4: Re-fit local coefficients removing global effects
  β_local = GWR(x1, y - fitted(x2, β_global), wMat_rp)  # Remove global, fit local

  # Step 5: Final global coefficients
  β_global = GWR(x3, y2, wMat_rp_ols)  # Final global fit

  model.β1 .= β_local
  model.β2 .= β_global
  # return β_local, β_global
  (; :local => β_local, :global => β_global)
end
