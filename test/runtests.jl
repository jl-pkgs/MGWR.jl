using MixedGWR, RTableTools, Distances, Test
include("main_pkgs.jl")

@testset "GWR" begin
  for adaptive in [true, false]
    for kernel in 0:4
      r = R"GWmodel:::gwr_q($x1, $y, $dMat, 20.0, $kernel + 1, $adaptive)" |> rcopy
      jl = GWR(x1, y, dMat, 20.0; kernel, adaptive)
      @test r ≈ jl
    end
  end
end

include("test-solver.jl")
include("test-ST_GWR.jl")
include("test-GWR_mixed.jl")
