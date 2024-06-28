from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian   #wake model
from py_wake.examples.data.iea37 import IEA37Site,IEA37_WindTurbines        #wind turbines and site used
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import XRSite
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent #cost model
import numpy as np
import pandas as pd
import xarray as xr
from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.examples.iea37 import get_iea37_initial, get_iea37_constraints, get_iea37_cost
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
##Set up wind turbine data
WT_tabular=pd.read_csv('./Predefined_data/IEA_15MW_240_RWT.csv')
u = np.array(WT_tabular.WindSpeed)
ct =np.array(WT_tabular.Ct)
power =np.array(WT_tabular.Power)
##define wind turbine in pywake class as "wt_IEA15'
wt_IEA15 = WindTurbine(name='IEA15MW',
                    diameter=150,
                    hub_height=240,
                    powerCtFunction=PowerCtTabular(u,power,'kW',ct))
windTurbines = wt_IEA15
#%%
# Wind farm boundary & constraint
p4=(120.7452392287392,   24.963961563689494)
p1=(120.80978390507431,  25.049835677916448)
p2=(120.76068879954698, 25.080312824428248)
p3=(120.6717681776592,     25.00690609719345)
boundary = 111*np.array([p1,p2,p3,p4])*1000 ##Trans to meter
wt_space=5*150  ##Set for 5D, diameter=150m, then trans to km.
n_wt = 50
n_wd = 36
#%%
#Site_data = np.load('./Predefined_data/siteData.npz')
Site_data=np.load("./Predefined_data/siteData_2010.npz")
ws=Site_data['arr2']
wd=Site_data['arr3']

# Define bins for wind speed and direction
wind_speed_bins = np.linspace(0, 30.5, 61, endpoint=False)  # Bins from 0 to 30 m/s
wind_direction_bins = np.linspace(0, 370,37,endpoint=False)  # Bins every 10 degrees
print(wind_direction_bins)
# Digitize the wind data into these bins
ws_indices = np.digitize(ws, wind_speed_bins)
wd_indices = np.digitize(wd, wind_direction_bins)
# Create a 2D histogram to count occurrences
hist, ws_edges, wd_edges = np.histogram2d(ws,wd, bins=[wind_speed_bins, wind_direction_bins])
probabilities = hist / hist.sum() # Normalize the histogram to get probabilities
probabilities=probabilities.T

# Calculate turbulence intensity (TI)
ti=0.1
# Create the dataset for XRSite
site = XRSite(
    ds=xr.Dataset(
        data_vars={
          'P': (('wd', 'ws'), probabilities),
          'TI': (('wd', 'ws'), np.full(probabilities.shape, ti))
        },
        coords={
          'wd': wind_direction_bins[:-1],
          'ws': wind_speed_bins[:-1]})
    #shear=PowerShear(h_ref=10, alpha=0.2)
)
#%% Define windfarm model
from py_wake.wind_farm_models import All2AllIterative
from py_wake.deficit_models import NOJDeficit, SelfSimilarityDeficit
from py_wake.superposition_models import LinearSum
#wf_model= NOJ(site,windTurbines,superpositionModel=LinearSum())
wf_model=All2AllIterative(site,windTurbines,
                wake_deficitModel=NOJDeficit(),
                superpositionModel=LinearSum(),
                blockage_deficitModel=SelfSimilarityDeficit())
#wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)   #PyWake's wind farm model


##Cost model
def aep_func(x,y,wd=wd):
    sim_res = wf_model(x,y, wd=wd)
    aep = sim_res.aep().sum()
    return aep
aep_comp = CostModelComponent(input_keys=['x','y'],
                              n_wt=n_wt,
                              cost_function=aep_func,
                              output_keys=[('AEP', 0)],
                              output_unit="GWh",
                              objective=True,
                              maximize=True
                             )


#Driver type
#driver = EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=100)
driver=EasyScipyOptimizeDriver(disp=False)


wt_location= np.load('./Predefined_data/initail_layout.npy')
wt_y=(wt_location[:,1]) * 1e3
wt_x=(wt_location[:,0]) * 1e3

#print(XYBoundaryConstraint(boundary, 'convex_hull'))
tf_problem = TopFarmProblem(
            design_vars={'x': wt_x, 'y': wt_y},
            #design_vars=design_vars,
            n_wt=n_wt,
            cost_comp=aep_comp,
            constraints=[SpacingConstraint(wt_space),XYBoundaryConstraint(boundary, 'convex_hull')],
            driver=driver,
            plot_comp=XYPlotComp(save_plot_per_iteration=True))
_, state, _ = tf_problem.optimize()

