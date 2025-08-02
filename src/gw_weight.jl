"Calculate weight vector for GWR"
function gw_weight!(wv::AbstractVector{T}, vdist::AbstractVector{T}, bw::T;
  kernel::Int=0, adaptive::Bool=true) where {T<:Real}

  kerf = GWR_KERNELS[kernel+1]  # Julia is 1-indexed
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

  @inbounds for i in 1:n
    wv[i] = kerf(vdist[i], bw)
  end
  return wv
end

function gw_weight(vdist::AbstractVector{T}, bw::T;
  kernel::Int=0, adaptive::Bool=true) where {T<:Real}
  wv = zeros(T, length(vdist))
  gw_weight!(wv, vdist, bw; kernel, adaptive)
end


export gw_weight, gw_weight!
