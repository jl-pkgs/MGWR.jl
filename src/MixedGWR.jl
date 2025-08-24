module MixedGWR

using LinearAlgebra, Statistics
using Parameters
using Base.Threads
import Base.summary
# using Polyester: @batch

export MGWR, update_weight!
export GWR_mixed, GWR_mixed_trace, gwr_q, gw_weight_vec, solver_reg, fitted
export fitted, predict, summary

export cor


include("MGWR.jl")
include("kernel.jl")
include("gw_weight.jl")
include("solve_chol.jl")
include("solve_reg.jl")
include("GWR.jl")
include("GWR_calib.jl")

include("GWR_mixed.jl")
include("GWR_mixed_trace.jl")

end # module MGWR
