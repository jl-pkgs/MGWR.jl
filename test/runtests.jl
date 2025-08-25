using MixedGWR, RTableTools, Distances, Test
include("main_pkgs.jl")


include("test-solver.jl")
model = MGWR(x1, x2, y, dMat; kernel=BISQUARE, adaptive=true, bw=20.0)

# 加权线性回归 求解方法

@testset "GWR" begin
  for adaptive in [true, false]
    for kernel in 0:4
      r = R"GWmodel:::gwr_q($x1, $y, $dMat, 20.0, $kernel + 1, $adaptive)" |> rcopy
      jl = GWR(x1, y, dMat, 20.0; kernel, adaptive)
      @test r ≈ jl
    end
  end
end


@testset "GWR_calib" begin
  β = GWR(model)
  res = GWR_calib(model)
  β2 = res.β
  β ≈ β2
  res.trace ≈ 29.296714491744776
  res.AIC ≈ 1308.7723907048241
end

include("test-GWR_mixed.jl")
