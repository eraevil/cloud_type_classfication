from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import cartopy.crs as ccrs
import shapefile
import matplotlib as mpl
import xarray as xr
from matplotlib.font_manager import FontProperties
import netCDF4 as nc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib
ZHfont = matplotlib.font_manager.FontProperties(fname='/Users/zhpfu/Documents/fonts/SimSun.ttf')

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=[12,18])
ax = fig.add_subplot(111)


def basemask(cs, ax, map, shpfile):

    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        if shape_rec.record[0] >= 0:
            pts = shape_rec.shape.points
            prt = list(shape_rec.shape.parts) + [len(pts)]
            for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i+1]):
                    vertices.append(map(pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                codes += [Path.CLOSEPOLY]
            clip = Path(vertices, codes)
            clip = PathPatch(clip, transform = ax.transData)
    for contour in cs.collections:
        contour.set_clip_path(clip)



def makedegreelabel(degreelist):
    labels=[str(x)+u'°E' for x in degreelist]
    return labels


ds = xr.open_dataset('EC-Interim_monthly_2018.nc')
lat = ds.latitude
lon = ds.longitude
data = (ds['t2m'][0,::-1,:] - 273.15) # 把温度转换为℃

# [west,east,south,north]
m = Basemap(llcrnrlon=70.,
    llcrnrlat=25,
    urcrnrlon=110,
    urcrnrlat=50,
    resolution = None,
    projection = 'cyl')


cbar_kwargs = {
    'orientation': 'horizontal',
    'label': 'Temperature(℃)',
    'ticks': np.arange(-30,30+1,10),
    'pad': -0.35,
    'shrink': 0.9
}

# 画图
levels = np.arange(-30,30+1,1)
cs = data.plot.contourf(ax=ax,levels=levels,cbar_kwargs=cbar_kwargs, cmap='Spectral_r')

m.readshapefile('west','West',color='k',linewidth=1.2)
basemask(cs, ax, m, 'west')
# draw parallels and meridians.
# label parallels on right and top
# meridians on bottom and left
parallels = np.arange(25.,50.+1,5.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[True,True,True,True],color='dimgrey',dashes=[2, 3],fontsize= 14)  # ha= 'right'
meridians = np.arange(70.,110.+1,5.)
m.drawmeridians(meridians,labels=[True,True,False,True],color='dimgrey',dashes=[2, 3],fontsize= 14)

plt.ylabel('')    #Remove the defult  lat / lon label
plt.xlabel('')


plt.rcParams.update({'font.size':25})
ax.set_title(u'中国西部地区部分省份',color='blue',fontsize= 25 ,fontproperties=ZHfont) # 2m Temperature

#经度：87.68 ， 纬度：43.77
bill0 = 87.68
tip0  =  43.77
plt.scatter(bill0, tip0,marker='.',s=120 ,color ="r",zorder=2)

#经度：103.73 ， 纬度：36.03
bill1 =  103.73
tip1  = 36.03
plt.scatter(bill1, tip1,marker='.',s=120 ,color ="r" ,zorder=2)

#经度：101.74 ， 纬度：36.56
bill2 =  101.74
tip2  = 36.56
plt.scatter(bill2, tip2,marker='.',s=120 ,color ="r",zorder=2 )


bill3 =  104.1
tip3  = 30.65
plt.scatter(bill3, tip3,marker='.',s=120 ,color ="r",zorder=2 )


bill4 =  91.11
tip4  = 29.97

plt.scatter(bill4, tip4,marker='.',s=120 ,color ="r",zorder=2)



plt.rcParams.update({'font.size':18})

plt.text(bill0-2.0, tip0+0.3, u"乌鲁木齐",fontsize= 18 ,color ="r",fontproperties=ZHfont)
plt.text(bill1-1., tip1+0.3, u"兰州"    ,fontsize= 18 ,color ="r",fontproperties=ZHfont)
plt.text(bill2-1., tip2+0.3, u"西宁"    ,fontsize= 18 ,color ="r",fontproperties=ZHfont)
plt.text(bill3-1., tip3+0.3, u"成都"    ,fontsize= 18 ,color ="r",fontproperties=ZHfont)
plt.text(bill4-1., tip4+0.3, u"拉萨"    ,fontsize= 18 ,color ="r",fontproperties=ZHfont)


# Save & Show figure
plt.savefig("West_China_mask.png", dpi=300, bbox_inches='tight')
plt.show()