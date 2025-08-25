@testset "ST_GWR" begin
  coords = Matrix(d[:, [:lon, :lat]])
  points = map(x -> x, eachrow(coords))

  fun_dist = Haversine(6378.388)
  dMat = pairwise(fun_dist, points)

  x1 = d[:, [:lon, :lat]] |> Matrix
  x2 = d[:, [:alt]] .* 1.0 |> Matrix
  y = d[:, :prcp]
  X = d[:, [:lon, :lat, :alt]] |> Matrix
  Y = repeat(y, outer = (1, 10))

  ## Parameters
  kernel = BISQUARE
  adaptive = true
  bw = 20.0 # 6个站点
  wMat = gw_weight(dMat, bw; kernel, adaptive)

  ## Run
  np = 2 # lon + lat
  @time Ypred2 = ST_GWR(X[:, 1:np], Y, wMat; Xpred=X[:, 1:np])
  @test Ypred2[:, 1] == Ypred2[:, 2]
end
