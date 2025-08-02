using MixedGWR, RTableTools, Distances, Test
include("main_pkgs.jl")
model = MGWR(x1, x2, y, dMat; kernel=BISQUARE, adaptive=true, bw=20.0)



@testset "GWR_mixed summary" begin
  @time βs = GWR_mixed(model)
  @test model.β1 == βs[:local]
  @test model.β2 == βs[:global]
  
  ypred = predict(model)
  @test cor(y, ypred) >= 0.85
  
  s = summary(model)
  @test s.AIC ≈ 1312.0799214329443
  @test s.σ ≈ 8.428717667622848
end


@testset "GWR" begin
  for adaptive in [true, false]
    for kernel in 0:4
      r = R"GWmodel:::gwr_q($x1, $y, $dMat, 20.0, $kernel + 1, $adaptive)" |> rcopy
      jl = GWR(x1, y, dMat, 20.0; kernel, adaptive)
      @test r ≈ jl
    end
  end
end


# ~4 times faster
@testset "GWR_mixed" begin
  @time res_r = GWR_mixed_r(x1, x2, y, dMat)
  @time res1 = GWR_mixed(x1, x2, y, dMat, dMat, 20.0; kernel=BISQUARE, adaptive=true)
  @time res2 = GWR_mixed(model)

  @test res1 == res2
  @test res_r[:local] ≈ res1[:local]
  @test res_r[:global] ≈ res1[:global]
end


# ~30 times faster
@testset "gwr_mixed_trace" begin
  @time res1 = GWR_mixed_trace(x1, x2, dMat, 20.0; kernel=BISQUARE, adaptive=true)
  @time res2 = GWR_mixed_trace(model)
  @test res1 == res2

  @time res_r = GWR_mixed_trace_r(x1, x2, y, dMat)
  @test res1 ≈ res_r
end
