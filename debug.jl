using MixedGWR, RTableTools, Distances
using RCall
using Test

# load data
begin
  d = fread("data/prcp_st174_shiyan.csv")

  coords = Matrix(d[:, [:lon, :lat]])
  points = map(x -> x, eachrow(coords))

  fun_dist = Haversine(6378.388)
  dMat = pairwise(fun_dist, points)

  x1 = d[:, [:lon, :lat]] |> Matrix
  x2 = d[:, [:alt]] .* 1.0 |> Matrix
  y = d[:, :prcp]
end


R"""
pacman::p_load(GWmodel)
"""

function gwr_mixed_r(x1, x2, y, dMat; kernel=BISQUARE)
  R"GWmodel:::gwr_mixed_2($x1, $x2, $y, $dMat, $dMat, 20.0, $kernel+1, TRUE)" |> rcopy
end

function gwr_mixed_trace_r(x1, x2, y, dMat; kernel=BISQUARE)
  R"GWmodel:::gwr_mixed_trace($x1, $x2, $y, $dMat, 20.0, $kernel+1, TRUE)" |> rcopy
end


@testset "gwr_q" begin
  adaptive = true
  # adaptive = false
  for kernel in 0:4
    r = R"GWmodel:::gwr_q($x1, $y, $dMat, 20.0, $kernel + 1, $adaptive)" |> rcopy
    jl = gwr_q(x1, y, dMat, 20.0, kernel, adaptive)
    @test r ≈ jl
  end
end

# w = rand(size(x1, 1))
# R"GWmodel:::gw_reg_1($x1, $y, $w)" #|> rcopy
model_r = gwr_mixed_r(x1, x2, y, dMat)
model = gwr_mixed(x1, x2, y, dMat, dMat, 20.0, BISQUARE, true)

@test model_r[:local] ≈ model[:local]
@test model_r[:global] ≈ model[:global]

y_jl = predict(model, x1, x2)
y_r = predict(model_r, x1, x2)

## 测试带宽的影响

using HydroTools
GOF(y[:], y_jl)
GOF(y[:], y_r)

using DataFrames
DataFrame(; y, y_jl, y_r)

# model["local"] 
@time trace = gwr_mixed_trace(x1, x2, dMat, 20.0, BISQUARE, true)


@time trace_r = gwr_mixed_trace_r(x1, x2, y, dMat)



model["local"]
model["global"]
model_r[:local]
model_r[:global]

# function test_mixed_gwr()
begin
  # Generate synthetic data
  # Run Mixed GWR
  println("Mixed GWR completed successfully!")
  println("Local coefficients shape: ", size(result["local"]))
  println("Global coefficients shape: ", size(result["global"]))
  return result
end
