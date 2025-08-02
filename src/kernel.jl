# Kernel function types
const GAUSSIAN = 0
const EXPONENTIAL = 1
const BISQUARE = 2
const TRICUBE = 3
const BOXCAR = 4


"""
Calculate the GW weights using different kernel functions
"""
function gw_weight_gaussian(dist::T, bw::T)::T where {T<:Real}
  return exp(dist^2 / (-2 * bw^2))
end

function gw_weight_exponential(dist::T, bw::T)::T where {T<:Real}
  return exp(-dist / bw)
end

function gw_weight_bisquare(dist::T, bw::T)::T where {T<:Real}
  return dist > bw ? 0.0 : (1 - (dist / bw)^2)^2
end

function gw_weight_tricube(dist::T, bw::T)::T where {T<:Real}
  return dist > bw ? 0.0 : (1 - (dist / bw)^3)^3
end

function gw_weight_boxcar(dist::T, bw::T)::T where {T<:Real}
  return dist > bw ? 0.0 : 1.0
end


# Kernel function array
const GWR_KERNELS = [
  gw_weight_gaussian,
  gw_weight_exponential,
  gw_weight_bisquare,
  gw_weight_tricube,
  gw_weight_boxcar
]


export GAUSSIAN, EXPONENTIAL, BISQUARE, TRICUBE, BOXCAR
