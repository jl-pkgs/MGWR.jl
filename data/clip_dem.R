pacman::p_load(
  Ipaper, data.table, dplyr, lubridate, 
  terra, sf2
)

f = "X:/rpkgs/VICTools.R/inst/database/dem_china_1km_SRTM.tif"
ra = rast(f)

poly = st_rect(c(109.4, 111.6, 31.2, 33.4))
ra2 = crop(ra, poly)

writeRaster(ra2, "./data/dem_ShiYan_1km.tif")
