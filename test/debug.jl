using HydroTools, DataFrames
using Ipaper



## 超参数优化
@profview trace = gwr_mixed_trace(x1, x2, dMat, 20.0; kernel=BISQUARE, adaptive=true)
@time trace = gwr_mixed_trace(x1, x2, dMat, 20.0; kernel=BISQUARE, adaptive=true) # 5 times faster
# @time trace_r = gwr_mixed_trace_r(x1, x2, y, dMat)


@time model = gwr_mixed(x1, x2, y, dMat, dMat, 20.0; kernel=BISQUARE, adaptive=true)
y_jl = predict(model, x1, x2)
# y_r = predict(model_r, x1, x2)
## 测试带宽的影响

bandwidths = [3., 5.0, 8., 10., 15, 20.]

## 需要结合kfold进行判断
@time info = map(bw -> begin
    model = gwr_mixed(x1, x2, y, dMat, dMat, bw; kernel=BISQUARE, adaptive=true)
    ypred = predict(model, x1, x2)
    gof = GOF(y, ypred)
    (; bw, gof...)
  end, bandwidths) |> DataFrame
DataFrame(info)




# summary
begin
  model = gwr_mixed(x1, x2, y, dMat, dMat, bw; kernel, adaptive)
  ypred = predict(model, x1, x2)
  ϵ = y - ypred
  RSS = sum(ϵ .^ 2)

  tr = gwr_mixed_trace(x1, x2, dMat, bw; kernel, adaptive)
  n = length(y)
  AIC = (ln(RSS / (n - tr)) + ln(2pi) + (n + tr) / (n - 2 - tr)) * n
end



# r.ss <- sum((y - gw_fitted(model2$global, x2) - gw_fitted(model2$local, x1))^2)
# n1 <- length(y)  
# sigma.aic <- r.ss / n1

# aic <- log(sigma.aic * 2 * pi) + 1 + 2 * (edf + 1) / (n1 - edf - 2)
# aic <- n1 * aic

# res$aic <- aic
# res$bic <- n1 * log(sigma.aic) + n1 * log(2 * pi) + edf * log(n1)
# res$df.used <- edf
# res$r.ss <- r.ss
