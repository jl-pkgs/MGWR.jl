using MixedGWR, RTableTools, Distances, Test
using RCall
R"""
library(GWmodel)
"""


function GWR_mixed_r(x1, x2, y, dMat; kernel=BISQUARE)
  R"GWmodel:::gwr_mixed_2($x1, $x2, $y, $dMat, $dMat, 20.0, $kernel+1, TRUE)" |> rcopy
end

function GWR_mixed_trace_r(x1, x2, y, dMat; kernel=BISQUARE)
  R"GWmodel:::gwr_mixed_trace($x1, $x2, $y, $dMat, 20.0, $kernel+1, TRUE)" |> rcopy
end


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
  y = d[:, :prcp];
end
