module MixedGWR

using LinearAlgebra, Statistics
using Parameters
using Base.Threads
import Base.summary
# using Polyester: @batch

export MGWR
export gwr_mixed, gwr_mixed_trace, gwr_q, gw_weight_vec, gw_reg, fitted
export fitted, predict, summary

export cor


include("MGWR.jl")
include("kernel.jl")
include("gw_weight.jl")
include("gwr_q.jl")
include("gwr_mixed.jl")
include("gwr_mixed_trace.jl")

end # module MGWR
