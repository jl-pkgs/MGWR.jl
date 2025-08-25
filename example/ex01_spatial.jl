using MixedGWR, RTableTools, Test
using SpatRasters, Distances, ArchGDAL

Base.Matrix(x::AbstractVector) = reshape(x, length(x), 1)


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
  x = d[:, [:lon, :lat, :alt]] |> Matrix
end


# 目标网格
begin
  ra = rast("data/dem_ShiYan_1km.tif")
  X1 = st_coords(ra)
  Points = map(x -> x, eachrow(X1))
  dMat_rp = pairwise(fun_dist, points, Points)

  X2 = (ra.A[:] * 1.0)
  X = cbind(X1, X2)
  n_target = size(X1, 1)
end


# bw: in km
model = MGWR(x1, x2, y, dMat, dMat_rp; kernel=BISQUARE, adaptive=true, bw=20.0)
β = GWR(model)
y_GWR = fitted(X1, β)

β1, β2 = GWR_mixed(model)
y_MGWR = fitted(X1, β1) + fitted(Matrix(X2), β2)

model = MGWR(x, x2, y, dMat, dMat_rp; kernel=BISQUARE, adaptive=true, bw=20.0)
@time β = GWR(model)
y_GWR2 = fitted(X, β)



using MakieLayers, GLMakie

lon, lat = st_dims(ra)
nlon, nlat = length(lon), length(lat)

res = rast(zeros(size(ra.A)..., 3), ra)
res.A[:, :, 1] .= reshape(y_GWR, nlon, nlat)
res.A[:, :, 2] .= reshape(y_GWR2, nlon, nlat)
res.A[:, :, 3] .= reshape(y_MGWR, nlon, nlat)

begin
  fig = Figure(; size=(1400, 900))
  imagesc!(fig, lon, lat, res.A;
    colorrange=(0, 60),
    titles=[
      "(a) GWR: (lon + lat)",
      "(b) GWR: (lon + lat + alt)",
      "(c) MGWR: (lon + lat) + alt"])
  fig
end
