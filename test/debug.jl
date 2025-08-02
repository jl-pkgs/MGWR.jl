

@profview trace = gwr_mixed_trace(x1, x2, dMat, 20.0, BISQUARE, true)
@time trace = gwr_mixed_trace(x1, x2, dMat, 20.0, BISQUARE, true)



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
