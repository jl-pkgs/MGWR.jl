using BenchmarkTools

w = dMat[:, 1]
ntime = 30000
Y = rand(length(y), ntime)

# @time β1 = solve_chol(x1, w, Y)
solver = GWRSolver(x1, Y)
@btime β = solve_chol!(solver, x1, Y, w)
@btime β1 = solve_reg(x1, Y, w)
@btime β2 = solve_reg2(x1, Y, w)

# about 18 times faster
β ≈ β1
β ≈ β2
