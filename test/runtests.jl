using MixedGWR, RTableTools, Distances, Test
include("main_pkgs.jl")


@testset "gwr_q" begin
  for adaptive in [true, false]
    for kernel in 0:4
      r = R"GWmodel:::gwr_q($x1, $y, $dMat, 20.0, $kernel + 1, $adaptive)" |> rcopy
      jl = gwr_q(x1, y, dMat, 20.0; kernel, adaptive)
      @test r ≈ jl
    end
  end
end


@testset "gwr_mixed" begin
  @time model_r = gwr_mixed_r(x1, x2, y, dMat)
  @time model = gwr_mixed(x1, x2, y, dMat, dMat, 20.0; kernel=BISQUARE, adaptive=true)

  @test model_r[:local] ≈ model[:local]
  @test model_r[:global] ≈ model[:global]
end


@testset "gwr_mixed_trace" begin
  @time trace = gwr_mixed_trace(x1, x2, dMat, 20.0; kernel=BISQUARE, adaptive=true)
  @time trace_r = gwr_mixed_trace_r(x1, x2, y, dMat)
  @test trace ≈ trace_r
end
