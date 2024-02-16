# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:36:49 2020

@author: Ammara
"""
##########################################################################################################
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm


SOLAR_CONSTANT=1367
WATTS_TO_MJ_PER_DAY=0.0864
STEFAN_WATTS=0.0000000567
##  mutiply above two values
STEFAN_MJ_PER_DAY=(WATTS_TO_MJ_PER_DAY*STEFAN_WATTS)
SFCEMISS=0.96  # surface emissivity

ALBEDO=0.25

lat=44.2

# convert degrees to radian
def degrees_to_rads(degrees):
    return degrees*(np.pi/180)

    
def declin(day_of_year):
    return 0.41*np.cos(2*np.pi*(day_of_year-172)/365)

## sunrise angle 
    
def sunrise_angle(day_of_year,lat):
    return np.arccos(-1*np.tan(declin(day_of_year))*np.tan(degrees_to_rads(lat)))
  

def sunrise_hour(day_of_year,lat):
    return 12-(12/np.pi)*sunrise_angle(day_of_year,lat)
    
    
def day_hours(day_of_year,lat):
    return 24-2*sunrise_hour(day_of_year,lat)
    

#### find components that will be used in clr_ratio
### clr_ratio is part of calculating what fraction of lw will be used

def av_eir(day_of_year):
    return SOLAR_CONSTANT*(1+0.035*np.cos(2*np.pi*day_of_year/365))
      
def to_eir(day_of_year,lat):
    return (0.0864/np.pi)*av_eir(day_of_year)*\
    (sunrise_angle(day_of_year,lat)*
     np.sin(declin(day_of_year))*
     np.sin(degrees_to_rads(lat))+
     np.cos(declin(day_of_year))*
     np.cos(degrees_to_rads(lat))*
     np.sin(sunrise_angle(day_of_year,lat)))
    
def to_clr(day_of_year,lat):
    return to_eir(day_of_year,lat)*(-0.7+0.86*day_hours(day_of_year,lat))/day_hours(day_of_year,lat)


## longwave upward uses surafce emissivity and temperature
def lwu(avg_temp):
    return SFCEMISS*STEFAN_MJ_PER_DAY*(273.15+avg_temp)**4

## slope of saturation vapor curve

def sfactor(avg_temp):
    return 0.398+(0.0171*avg_temp)-(0.000142*avg_temp*avg_temp)

# clear sky emissivity (dimentionless) calculated y using the method of Idso (1981)
# before correction

def sky_emiss(avg_v_press,avg_temp):
    if(avg_v_press>0.5).any():
        return 0.7+(5.95e-4)*avg_v_press*np.exp(1500/(273+avg_temp))
    else:
        return (1-0.261*np.exp(-0.000777*avg_temp*avg_temp))    
  
## calcultae 1- clear sky emissivity factor for Long wave

def angstrom(avg_v_press,avg_temp):
    return 1-sky_emiss(avg_v_press,avg_temp)/SFCEMISS

# ratio of measured insolation divided by the theoratical value  calculated for clear-air conditions
def clr_ratio (d_to_sol,day_of_year,lat):
    tc=to_clr(day_of_year,lat)
    # never return higher than 1
    if (d_to_sol/tc>1).any():
        return 1
    else:
        return d_to_sol/tc

## calculate net thermal infrared influx term (Ln) of the total net radiation consisting of the two directional terms upwelling and downwelling

def lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat):
    return angstrom(avg_v_press,avg_temp)*lwu(avg_temp)*clr_ratio(d_to_sol,day_of_year,lat)


def et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat):
    ## calculate lwnet
    lwnnet=lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat)       
   ## calculate R_n 
    net_radiation=(1-ALBEDO)*d_to_sol-lwnnet
    ## formula for evapotranspiration
    ret1=1.26*sfactor(avg_temp)*net_radiation
    # assume 62.3 is the conversion factor but unable to determine 
    return ret1/62.3

##############################################################################################################

## temperature  in celcius
# avg_v_pressure in kpa (kilopascal)
# d_to_sol is insolation reading in MJ/day (Mega joules/day)
#lat is latitude in fractional degrees
###############################################################################################################
data=pd.read_csv('data.csv')

df=data

df["month"]= pd.to_datetime(df['TIMESTAMP']).dt.month
mask = (df['month'] >=5) & (df['month'] <= 8) # select growing season data
df = df.loc[mask]
df.index = np.arange(0, len(df))

pot_temp_c=(data["Tair"]-32 )*(5/9)  # observed temp
avg_temp=data["inso_temp_c"].astype(float)
avg_v_press=data["inso_vp_kpa"].astype(float)
d_to_sol=data["inso_rad"].astype(float)
day_of_year=pd.to_datetime(data["TIMESTAMP"]).dt.dayofyear
data["day_of_year"]=day_of_year

###############################################################################################################
# calculate longwave
long_wave= lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat) 

plt.plot(-long_wave*11.4) #11.4 is for unit conversion to Wm-2

data["lwnet_idso"]=(-long_wave*11.4)
data["idso_emis"]=sky_emiss(avg_v_press,avg_temp) 

data["wisp_PET_idso"]=et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat)

# bring canopy data to convert PET to ET

data['canop']=data['canop'] # use that for other sites

def adj_AET (data):
    if data['canop']>=80:
        return data["wisp_PET_idso"]
    else:
        return data["wisp_PET_idso"]*((data['canop']/80+0.0833))

data["wisp_AET_idso"]=data.apply(adj_AET, axis = 1)

##############################################################################################################
#Next run code to calculate corrected emissivity 

# after correction: make sure to include new corrected coefficients.       
def sky_emiss(avg_v_press,avg_temp):
        if(avg_v_press>0.5).any():
            return 0.544+(6.4e-04)*avg_v_press*np.exp(1500/(273+avg_temp))
        else:
            return (1-0.261*np.exp(-0.000777*avg_temp*avg_temp))    
        
data["idso_emis_corr"]=sky_emiss(avg_v_press,avg_temp)        

# calculated corrected longwave based on corrected sky emisssivity 
long_wave= lwnet(avg_v_press,avg_temp,d_to_sol,day_of_year,lat) 
 
data["lwnet_idso_corr"]=(-long_wave*11.4)

# calculate corrected WISP ETa
data["wisp_PET_idso_corr"]=et(avg_temp,avg_v_press,d_to_sol,day_of_year,lat)

def adj_AET (data):
    if data['canop']>=80:
        return data["wisp_PET_idso_corr"]
    else:
        return data["wisp_PET_idso_corr"]*((data['canop']/80+0.0833))

data["wisp_AET_idso_corr"]=data.apply(adj_AET, axis = 1)
##############################################################################################################

# calculate observed emissivity from flux tower data to compare it with idso model emissivity 

# potential solar irridance in Wm-2 calculated based on Hulstrom, R., Bird, (1985) method
df["pot_solar"]=df["Solar_W_m-2"]
df["SW-in"]=df["SW_Wm-2"] # incoming short wave radiation from flux tower 

df["sol_diff"]=df["pot_solar"]-df["SW_in"]

#### use data only for hours with more solar radiation 
mask = (df['Hr'] >=10) & (df['Hr'] <= 14)
df = df.loc[mask]
df.index = np.arange(0, len(df))

# use only days when incoming solar radiation is >=90 % of potential irradiance, that will give clear sky conditions  
mask = (df['SW_in'] >=(.9*df['pot_solar']))
df.loc[mask]
df = df.loc[mask]
df.index = np.arange(0, len(df))

# calculate hourly observed emissivity 

df["es_obs"]=0.96*(df["LW_in"]/df["LW_out"])

df.to_csv('emiss_hour.csv', index=False, header=True)

########################################################################################################
# calculate correct emissivity

data=pd.read_csv('emiss.csv')# all emissivity data
data1=pd.read_csv('data.csv')


X=avg_v_press*np.exp(1500/(273+avg_temp))  # model emissivity. X of linear equation

Y=df["es_obs"] # observed emissivity 

plt.scatter(X, Y)
plt.xlabel('obs_emiss')
plt.ylabel('vpress*exp(1500/273+avg temp)')
plt.show()

# fit ordinarly least square to calculate slope and intercept 
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
results.params
 
results.summary()       

## compare results 

Y=df["es_obs"]  #obs
X=0.7+(5.95e-4)*avg_v_press*np.exp(1500/(273+avg_temp)) #idso_emissivity
X_corr=0.544+(6.4e-04)*avg_v_press*np.exp(1500/(273+avg_temp))  # idso_corrected_emissivity
  
plt.plot(Y,'r--', label="obs_emiss") 
plt.plot(X_corr,'b--', label="idso_emisscorr" )

############################################################################################################

# compare results after running WISP model 


x1=data["ET_obs"]
y2=df["wisp_ET"]
y1=data["wisp_corr_ET"]

pbias=(np.sum(x1-y2)/np.sum(x1))*100
pbias

x1=pd.DataFrame(x1)
y1=pd.DataFrame(y1)
reg = LinearRegression(fit_intercept=True).fit(x1, y1)
a=reg.coef_

reg.coef_
#reg.intercept_
reg.score(x1, y1)
pred=(y1/reg.coef_)

import matplotlib.lines as mlines
plt.rcParams['figure.figsize'] = (2.2,2.1)
fig, ax = plt.subplots()
ax.plot(x1,y1,'x', color="indigo",markersize=2,label="AET")
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.3)
line = mlines.Line2D([0, 1], [0, 1], color='black',linewidth=0.5)
#ax.plot(x1,a*x1,'r-')
ax.plot(x1,a*x1,'r-', label='Slope=1.38,R2=0.74')
#ax.plot(x1,pred,'x',color="black",markersize=2, label='R2=0.53')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.legend(loc='upper right',prop={'size': 6})
ax.set_ylabel('model ET')  # we already handled the x-label with ax1
ax.set_xlabel('obs ET')  # we already handled the x-label with ax1
ax.locator_params(axis='y', nbins=6)
ax.locator_params(axis='x', nbins=6)
ax.yaxis.label.set_size(5)
ax.xaxis.label.set_size(5) #there is no label 
ax.tick_params(axis = 'y', which = 'major', labelsize = 5,direction='in')
ax.tick_params(axis = 'x', which = 'major', labelsize = 5,direction='in')
ax.set_ylim(0,0.3)
ax.set_xlim(0,0.3)
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.show()
fig.savefig('testfig.png',dpi=300, bbox_inches = "tight")
plt.tight_layout()

########################################################################################################