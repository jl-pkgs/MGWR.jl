export GWR_calib, solver_reg2

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
    βᵢ, Cᵢ = _solver_reg2(x, y, wMat[:, i]) # βᵢ: [1, p_local], Cᵢ: [p_local, n_control]
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


"""
    solver_reg2(X::Matrix{T}, y::Vector{T}, w::AbstractVector{T})

β = C_i y = [(X' W_i X)⁻¹ X' W_i] y
̂y = S y = x_i C_i y
"""
function _solver_reg2(X::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}) where {T<:Real}
  XtW = (X .* w)'
  XtWx = XtW * X
  XtWy = X' * (w .* y)

  XtWx_inv = inv(XtWx)
  βᵢ = XtWx_inv * XtWy  # Li 2019, Eq. 9
  Cᵢ = XtWx_inv * XtW   # Li 2019, Eq. 10
  return βᵢ, Cᵢ
end
