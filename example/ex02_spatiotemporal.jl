using Base.Threads
using Ipaper


# load data
begin
  indir = "$(@__DIR__)/.." |> abspath
  d = fread("$indir/data/prcp_st174_shiyan.csv")

  coords = Matrix(d[:, [:lon, :lat]])
  points = map(x -> x, eachrow(coords))

  fun_dist = Haversine(6378.388)
  dMat = pairwise(fun_dist, points)

  x1 = d[:, [:lon, :lat]] |> Matrix
  x2 = d[:, [:alt]] .* 1.0 |> Matrix
  y = d[:, :prcp]

  X = d[:, [:lon, :lat, :alt]] |> Matrix
  Y = repeat(y, outer=(1, 1000))
end

# 目标网格
begin
  ra = rast("data/dem_ShiYan_1km.tif")
  X1 = st_coords(ra)
  Points = map(x -> x, eachrow(X1))
  dMat_rp = pairwise(fun_dist, points, Points)

  X2 = (ra.A[:] * 1.0)
  Xpred = cbind(X1, X2)
end

begin
  kernel = BISQUARE
  adaptive = true
  bw = 20.0 # 6个站点
  wMat_rp = gw_weight(dMat_rp, bw; kernel, adaptive)
end

# 5 second, (69696, 1000) [n_target, n_time]

np = 2 # lon + lat
@time Ypred2 = main_GWR(X[:, 1:np], Y, wMat_rp; Xpred = Xpred[:, 1:np]);

np = 3 # lon + lat + alt
@time Ypred3 = main_GWR(X[:, 1:np], Y, wMat_rp; Xpred=Xpred[:, 1:np]);
