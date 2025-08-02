using Parameters


@with_kw struct MGWR{T<:Real}
  n_control::Int = 100
  n_target::Int = 10
  p_local::Int = 2
  p_global::Int = 2

  x1::Matrix{T} = zeros(T, n_control, p_local)         # [n_control, p_local], local
  x2::Matrix{T} = zeros(T, n_control, p_global)        # [n_control, p_global], global
  x3::Matrix{T} = zeros(T, n_control, p_global)        # [n_control, p_global], intermediate, global - local_effect
  y::Vector{T} = zeros(T, n_control)                   # [n_control]
  dMat::Matrix{T} = zeros(T, n_control, n_control)     # [n_control, n_control]
  dMat_rp::Matrix{T} = zeros(T, n_control, n_target)   # [n_control, n_target]

  wMat::Matrix{T} = zeros(T, n_control, n_control)     # [n_control, n_control], intermediate
  wMat_ols::Matrix{T} = zeros(T, n_control, n_control) # [n_control, n_control], intermediate
  wMat_rp::Matrix{T} = zeros(T, n_control, n_target)   # [n_control, n_target], intermediate
  wMat_rp_ols::Matrix{T} = zeros(T, n_control, n_target) # [n_control, n_target], intermediate

  β1::Matrix{T} = zeros(T, n_control, p_local)         # [n_control, p_local]
  β2::Matrix{T} = zeros(T, n_control, p_global)        # [n_control, p_global]
  bw::T = 5.0
  kernel::Int = 0
  adaptive::Bool = false
end


function MGWR(x1::Matrix{T}, x2::Matrix{T}, y::Vector{T}, dMat::Matrix{T}, dMat_rp::Matrix{T}=dMat;
  bw=5.0, kernel=0, adaptive::Bool=false) where {T}

  n_control, n_target = size(dMat_rp)
  p_local = size(x1, 2)
  p_global = size(x2, 2)

  # 几个权重都可以进行计算了
  kernel_ols = BOXCAR
  bw_ols = 100000.0

  wMat = gw_weight(dMat, bw; kernel, adaptive)
  wMat_ols = gw_weight(dMat, bw_ols; kernel=kernel_ols, adaptive=true)
  wMat_rp = gw_weight(dMat_rp, bw; kernel, adaptive)
  wMat_rp_ols = gw_weight(dMat_rp, bw_ols; kernel=kernel_ols, adaptive=true)

  MGWR{T}(;
    n_control, n_target, p_local, p_global,
    x1, x2, y, dMat, dMat_rp,
    wMat, wMat_ols, wMat_rp, wMat_rp_ols,
    bw, kernel, adaptive
  )
end


"Fitted values from coefficient matrix and design matrix"
function fitted(x::Matrix{Float64}, β::Matrix{Float64})::Vector{Float64}
  return vec(sum(x .* β, dims=2))
end


function predict(model::MGWR)
  (; x1, x2, β1, β2) = model
  return fitted(x1, β1) + fitted(x2, β2) # ypred
end

function summary(model::MGWR)
  (; y) = model
  n = length(y)

  tr = gwr_mixed_trace(model)
  ypred = predict(model)

  ϵ = y - ypred
  RSS = sum(ϵ .^ 2)
  RMSE = sqrt(RSS / n)
  σ = sqrt(RSS / (n - tr))
  r = cor(y, ypred)

  AIC = (log(RSS / (n - tr)) + log(2pi) + (n + tr) / (n - 2 - tr)) * n # Li 2019, Eq. 16
  return (; n, r, RSS, RMSE, σ, trace=tr, AIC)
end
