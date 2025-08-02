"Calculate weight vector for GWR"
function gw_weight!(wv::AbstractVector{T}, vdist::AbstractVector{T}, bw::T;
  kernel::Int=0, adaptive::Bool=true) where {T<:Real}

  kerf::Function = GWR_KERNELS[kernel+1]  # Julia is 1-indexed
  
  n = length(vdist)
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
    @inbounds wv[i] = kerf(vdist[i], bw)
  end
  return wv
end

# function gw_weight(vdist::AbstractVector{T}, bw::T;
#   kernel::Int=0, adaptive::Bool=true) where {T<:Real}
#   wv = zeros(T, size(vdist))
#   gw_weight!(wv, vdist, bw; kernel, adaptive)
# end


## 二维数据
function gw_weight!(ws::AbstractMatrix{T}, dist::AbstractMatrix{T}, bw::T;
  kernel::Int=0, adaptive::Bool=true) where {T<:Real}
  n_control, n_target = size(dist)
  
  @inbounds for j in 1:n_target
    _w = @view ws[:, j]
    _dist = @view dist[:, j]
    gw_weight!(_w, _dist, bw; kernel, adaptive)
  end
  return ws
end


function gw_weight(dist::AbstractMatrix{T}, bw::T; kernel::Int=0, adaptive::Bool=true) where {T<:Real}
  wMat = zeros(T, size(dist))
  gw_weight!(wMat, dist, bw; kernel, adaptive)
end


export gw_weight, gw_weight!
