# https://chatgpt.com/c/68aadc51-a978-8326-8880-9df82cecb413
using LinearAlgebra
export GWRSolver
export solve_chol!, solve_chol


# 这里存储的是临时变量
@with_kw mutable struct GWRSolver{T}
  n_control::Int = 100
  p::Int = 2
  n_time::Int = 10
  # X::Matrix{T} = zeros(T, n_control, p)           # k×p
  # Y::Matrix{T} = zeros(T, n_control, n_time)
  # w::Vector{T} = zeros(T, n_control)              # k, 
  WY::Matrix{T} = zeros(T, n_control, n_time)
  XtWY::Matrix{T} = zeros(T, p, n_time)
  β::Matrix{T} = zeros(T, p, n_time)
end

function GWRSolver(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T<:Real}
  n_control, p = size(X)
  n_time = size(Y, 2)
  GWRSolver{T}(; n_control, p, n_time)
end


"""
# Arguments
- `X`: [k, p]
- `w`: [k]
- `Y`: [k, ntime]
"""
function solve_chol!(
  β::AbstractMatrix{T}, WY::AbstractMatrix{T}, XtWY::AbstractMatrix{T},
  X::AbstractMatrix{T}, w::AbstractVector{T}, Y::AbstractMatrix{T};
  λ::T=T(1e-8)) where {T<:Real}
  n_control, p = size(X)
  _, ntime = size(Y)

  Xw = X .* w                            # k×p 行加权
  XtWX = Symmetric(transpose(X) * Xw)    # p×p
  @inbounds @simd for i in axes(XtWX, 1)
    XtWX[i, i] += λ # 小岭，稳健
  end
  F = cholesky!(XtWX; check=false) # [p, p], small matrix

  @inbounds @threads for i in 1:n_control
    wi = w[i]
    @simd for j in 1:ntime
      WY[i, j] = Y[i, j] * wi
    end
  end
  mul!(XtWY, transpose(X), WY) # R = X' * WY
  ldiv!(β, F, XtWY)
  return β
end


function solve_chol!(solver::GWRSolver{T}, 
  X::AbstractMatrix{T}, w::AbstractVector{T}, Y::AbstractMatrix{T};
  λ::T=T(1e-8)) where {T<:Real}
  (; β, WY, XtWY) = solver
  solve_chol!(β, WY, XtWY, X, w, Y; λ)
end


function solve_chol(
  X::AbstractMatrix{T}, w::AbstractVector{T}, Y::AbstractMatrix{T}; λ::T=T(1e-8)) where {T<:Real}
  solver = GWRSolver(X, Y)
  solve_chol!(solver, X, w, Y; λ)
end
