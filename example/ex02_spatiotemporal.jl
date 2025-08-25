using Base.Threads
using Ipaper


# 目标网格
begin
  ra = rast("data/dem_ShiYan_1km.tif")
  X1 = st_coords(ra)
  Points = map(x -> x, eachrow(X1))
  dMat_rp = pairwise(fun_dist, points, Points)

  X2 = (ra.A[:] * 1.0)
  X = cbind(X1, X2)
  n_target = size(X1, 1)
  
  Xpred = X1
end


begin
  kernel = BISQUARE
  adaptive = true
  bw = 20.0 # 6个站点
  wMat_rp = gw_weight(dMat_rp, bw; kernel, adaptive)

  X = x1
  Y = repeat(y, outer=(1, 1000))
end



function main_GWR(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
  wMat::AbstractMatrix{T}; Xpred::AbstractMatrix{T}) where {T<:Real}
  k_local = size(X, 2)
  n_target = size(wMat, 2)
  ntime = size(Y, 2)
  # β = zeros(T, n_target, k_local, ntime)
  n_target = size(wMat, 2)
  solvers = map(i -> GWRSolver(X, Y), 1:Threads.nthreads())
  
  Ypred = zeros(T, n_target, ntime)
  p = Progress(n_target)
  @inbounds @threads for i in 1:n_target
    next!(p)
    k = Threads.threadid()
    solver = solvers[k]

    w = @view wMat[:, i]
    _β = solve_chol!(solver, X, Y, w) # [np, ntime]
    # β[i, :, :] .= _β

    _X = Xpred[i, :] # [np]
    _pred = @view Ypred[i, :] # [ntime]
    fitted!(_pred, _X, _β)
  end
  return Ypred
end

# 5 second, (69696, 1000) [n_target, n_time]
@time Ypred = main_GWR(x1, Y, wMat_rp; Xpred);
obj_size(Ypred) # 10Gb
