export solve_reg, solve_reg2


"""
    solver_reg2(X::Matrix{T}, y::Vector{T}, w::AbstractVector{T})

β = C_i y = [(X' W_i X)⁻¹ X' W_i] y
̂y = S y = x_i C_i y
"""
function solve_reg2(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, w::AbstractVector{T}) where {T<:Real}
  Xt = X'
  XtW = (X .* w)'
  XtWx = XtW * X
  XtWx_inv = inv(XtWx)

  n, p = size(X)
  ntime = size(Y, 2)
  β = zeros(T, p, ntime)

  @inbounds @threads for i in 1:ntime
    y = @view Y[:, i]
    XtWy = Xt * (w .* y)
    βᵢ = XtWx_inv * XtWy  # Li 2019, Eq. 9  
    β[:, i] .= βᵢ[:]
  end
  return β
end


"Geographically weighted regression for single location"
function solve_reg(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, w::AbstractVector{T}) where {T<:Real}
  w_sqrt = sqrt.(w)
  xw = X .* w_sqrt # 重新分配内存
  XtWX = (xw' * xw)
  n, p = size(X)
  ntime = size(Y, 2)
  β = zeros(T, p, ntime)

  @inbounds @threads for i in 1:ntime
    y = @view Y[:, i]
    yw = y .* w_sqrt
    βᵢ = XtWX \ (xw' * yw)
    β[:, i] .= βᵢ[:]
  end
  return β
end


Base.Matrix(x::Vector) = reshape(x, length(x), 1)

solve_reg(X::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}) where {T<:Real} =
  solve_reg(X, Matrix(y), w)

solve_reg2(X::AbstractMatrix{T}, y::AbstractVector{T}, w::AbstractVector{T}) where {T<:Real} =
  solve_reg2(X, Matrix(y), w)
