# ~3 minutes
using MixedGWR, SpatRasters, ArchGDAL, Distances
using Ipaper, RTableTools, NetCDFTools
using JLD2, UnPack, NaNStatistics, DataFrames

fun_dist = Haversine(6378.388)

# 目标网格
begin
  ra = rast("data/dem_ShiYan_2km.tif", FT=Float64)
  X1 = st_coords(ra)
  Points = map(x -> x, eachrow(X1))

  X2 = (ra.A[:] * 1.0)
  Xpred = cbind(X1, X2)
end

f = "/mnt/z/GitHub/jl-pkgs/SpatRasters.jl/Project_十堰/data/ShiYan_Pobs_interpolated_by_IDW.jld2" |> path_mnt
l = jldopen(f)
@unpack st, dates, P = l

# INPUT
begin
  coords = Matrix(st[:, [:lon, :lat]])
  points = map(x -> (x[1], x[2]), eachrow(coords))
  dMat = pairwise(fun_dist, points)
  dMat_rp = pairwise(fun_dist, points, Points)

  x1 = st[:, [:lon, :lat]] |> Matrix
  x2 = st_extract(ra, points).value' |> Matrix
  X = cbind(x1, x2)

  ## 只对有降水的日期进行插值
  _prcp = NaNStatistics.nansum(P, dims=1)[:]
  inds = findall(_prcp .>= 1.0) # 所有站点总降水大于1mm日期，才进行插值, ~1/3
  info = DataFrame(; time=dates[inds], inds, prcp=_prcp[inds])
  Y = P[:, inds]
end

bw = 10 # 3.7
bw = 20 # 11.4
bw = 30 # 22.8
map(x -> sum(x .<= bw), eachrow(dMat)) |> mean

## 5 second, (69696, 1000) [n_target, n_time]

adaptive = true
bandwidths = [6.0, 10.0, 20.0]

adaptive = false
bandwidths = [10.0, 20, 30, 50] # in km

## 考虑高程效果更好一些
outdir = "/mnt/z/GitHub/jl-pkgs/SpatRasters.jl/Project_十堰/OUTPUT" |> path_mnt

for bw in bandwidths
  fout = "$outdir/ShiYan_Prcp_Gauged237_201404-202501_2km_GWR3(adaptive=$adaptive,bw=$bw).nc"
  isfile(fout) && continue

  kernel = BISQUARE
  wMat_rp = gw_weight(dMat_rp, bw; kernel, adaptive)

  np = 3 # lon + lat + alt
  @time Ypred = ST_GWR(X[:, 1:np], Y, wMat_rp; Xpred=Xpred[:, 1:np])

  nlon, nlat = size(ra)[1:2]
  R = zeros(Float32, nlon, nlat, length(dates))
  R[:, :, inds] .= reshape(Ypred, nlon, nlat, length(inds))

  lon, lat = st_dims(ra)
  dims = (; lon, lat, time=dates)
  ncsave(fout, true, (; units="mm h-1"); dims, P=R)
end
