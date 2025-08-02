using HydroTools, DataFrames


## 超参数优化
@profview trace = gwr_mixed_trace(x1, x2, dMat, 20.0; kernel=BISQUARE, adaptive=true)
@time trace = gwr_mixed_trace(x1, x2, dMat, 20.0; kernel=BISQUARE, adaptive=true) # 5 times faster
# @time trace_r = gwr_mixed_trace_r(x1, x2, y, dMat)

y_jl = predict(model, x1, x2)
y_r = predict(model_r, x1, x2)

## 测试带宽的影响
GOF(y[:], y_jl)
GOF(y[:], y_r)
DataFrame(; y, y_jl, y_r)
