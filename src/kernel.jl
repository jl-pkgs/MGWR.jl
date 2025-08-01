# Kernel function types
const GAUSSIAN = 0
const EXPONENTIAL = 1
const BISQUARE = 2
const TRICUBE = 3
const BOXCAR = 4

"""
Calculate the GW weights using different kernel functions
"""
function gw_weight_gaussian(dist::Float64, bw::Float64)::Float64
  return exp(dist^2 / (-2 * bw^2))
end

function gw_weight_exponential(dist::Float64, bw::Float64)::Float64
  return exp(-dist / bw)
end

function gw_weight_bisquare(dist::Float64, bw::Float64)::Float64
  return dist > bw ? 0.0 : (1 - (dist / bw)^2)^2
end

function gw_weight_tricube(dist::Float64, bw::Float64)::Float64
  return dist > bw ? 0.0 : (1 - (dist / bw)^3)^3
end

function gw_weight_boxcar(dist::Float64, bw::Float64)::Float64
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


"Calculate weight vector for GWR"
function gw_weight(vdist::Vector{Float64}, bw::Float64, kernel::Int, adaptive::Bool)::Vector{Float64}
  kerf = GWR_KERNELS[kernel+1]  # Julia is 1-indexed
  n = length(vdist)
  wv = zeros(Float64, n)

  if adaptive
    dn = bw / n
    if dn <= 1
      svdist = sort(vdist)
      bw = svdist[Int(bw)]  # Julia is 1-indexed
    else
      bw = dn * maximum(vdist)
    end
  end

  for i in 1:n
    wv[i] = kerf(vdist[i], bw)
  end
  return wv
end

export GAUSSIAN, EXPONENTIAL, BISQUARE, TRICUBE, BOXCAR
export gw_weight
