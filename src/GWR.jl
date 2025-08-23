export GWR
export GWR_calib


"Geographically weighted regression for single location"
function gw_reg(X::Matrix{T}, y::Vector{T}, w::AbstractVector{T})::Matrix{T} where {T<:Real}
  w_sqrt = sqrt.(w)
  xw = X .* w_sqrt # 重新分配内存
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
    gw_reg2(X::Matrix{T}, y::Vector{T}, w::AbstractVector{T})

β = C_i y = [(X' W_i X)⁻¹ X' W_i] y
̂y = S y = x_i C_i y
"""
function gw_reg2(X::Matrix{T}, y::Vector{T}, w::AbstractVector{T}) where {T<:Real}
  XtW = (X .* w)'
  XtWx = XtW * X
  XtWy = X' * (w .* y)

  XtWx_inv = inv(XtWx)
  βᵢ = XtWx_inv * XtWy  # Li 2019, Eq. 9
  Cᵢ = XtWx_inv * XtW   # Li 2019, Eq. 10
  return βᵢ, Cᵢ
end


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
  @inbounds for i in 1:n_target
    w = @view wMat[:, i]
    _β = gw_reg(X, y, w) # [1, k_local]
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
  (; x1, y, wMat) = model
  GWR(x1, y, wMat)
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
    β[i, :] = gw_reg(x, y, w)
  end
  return β
end



function GWR_calib(model::MGWR)
  (; x1, y, wMat) = model
  GWR_calib(x1, y, wMat)
end


function GWR_calib(x::Matrix{T}, y::Vector{T}, wMat::AbstractMatrix{T}) where {T<:Real}
  p_local = size(x, 2)
  n_control, n_target = size(wMat)

  varC = zeros(n_target, p_local)
  ypred = zeros(n_target)

  tr = T(0)
  # tr_StS = T(0)
  # q_diag = T(0)
  βs = zeros(n_target, p_local)

  for i in 1:n_control
    βᵢ, Cᵢ = gw_reg2(x, y, wMat[:, i]) # βᵢ: [1, p_local], Cᵢ: [p_local, n_control]
    βs[i, :] = βᵢ
    varC[i, :] = diag(Cᵢ * Cᵢ')

    sᵢ = x[i:i, :] * Cᵢ # [1, n_control]
    tr += sᵢ[i]      # s_hat1, Stewart 2002, Eq. 2.16
    # tr_StS += sum(sᵢ.^2) # s_hat2
  end

  n = n_control
  ypred = fitted(x, βs)
  ϵ = y - ypred
  RSS = sum(ϵ .^ 2)
  RMSE = sqrt(RSS / n)
  σ = sqrt(RSS / (n - tr))

  se_β = σ * sqrt.(varC)
  r = cor(y, ypred)

  AIC = (log(RSS / (n - tr)) + log(2pi) + (n + tr) / (n - 2 - tr)) * n # Li 2019, Eq. 16
  (; β=βs, se_β, n=n_target, r, RSS, RMSE, σ, trace=tr, AIC)
end
