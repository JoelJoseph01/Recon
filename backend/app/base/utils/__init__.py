import statistics as stat


# Returns mean of coordinates
def getMapCenter(sights):
    latitudes  = [float(x.latitude) for x in sights]
    longitudes = [float(x.longitude) for x in sights]
    center_lat = str(round(stat.mean(latitudes),6))
    center_lon = str(round(stat.mean(longitudes),6))
    return center_lat, center_lon