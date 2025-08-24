using Base.Threads
using Polyester
using BenchmarkTools

w = dMat[:, 1]
ntime = 30000
Y = rand(length(y), ntime)


function fun2(x1::Matrix{T}, w::Vector{T}, Y::Matrix{T}) where {T}
  ntime = size(Y, 2)
  p = size(x1, 2)
  R = zeros(p, ntime)

  @inbounds @batch for i in 1:ntime
    y = @view Y[:, i]
    β = gw_reg(x1, y, w)
    R[:, i] .= β[:]
  end
  R
end


# @time β1 = solve_chol(x1, w, Y)
solver = GWRSolver(x1, Y)
@time β1 = solve_chol!(solver, x1, w, Y);
@time β2 = fun2(x1, w, Y);
β1 ≈ β2


# about 18 times faster
@benchmark β1 = fun1(x1, w, Y; WY)
@benchmark β2 = fun2(x1, w, Y)

@profview β1 = fun1(x1, w, Y; WY)
β1 ≈ β2

