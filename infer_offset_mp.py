"""
Tom Dixon
26/07/18
thomas.dixon@unsw.edu.au

This is a function (and implementation for testing
purposes) that computes the offset from 0,0 of a particle trajectory.
In my data collection, I have made assumptions about the position-of-mean for
trajectories collected. These are not necessarily valid, but it is not possible
to calculate the mean for such small datasets.
It uses the known brownian motion behviour of the particle (characterised
earlier) to infer the likely position of the centre of a trajectory. It
takes an input offset (+/-x and y deviation between the estimated mean
and the true mean) and optimises to find the location of the true mean.

These optimisations will be integrated in to my visualise script.

THIS RUNS IN PYTHON 2! Use python2-nonda to run...
"""

import pandas as pd 
import numpy as np
import logging
import math
import scipy.optimize as opt
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import multiprocessing as mp
import itertools
import matplotlib.patches as patches
from scipy.interpolate import NearestNDInterpolator
import scipy.interpolate as interp
import scipy as sp
import random
#%%
def find_offset(data, smesh_df, drag_coeff, init_guess, logger, inc_size, inc_range):
    """
    find_offset finds the offsets in x and y that give the highest
    cumulative probability of trajectory data_df. It uses a bayesian
    inference scheme I stole from InferenceMap

    data_df is a pandas dataFrame with the trajectory data inside. 
    it should have the following columns (order not important):
        x_pos (m)
        y_pos (m)
        time (seconds)

    smesh_df is the dataFrame that contains the smesh output of an
    InferenceMAP fit to the whole trajectory. This contains the
    diffusion and force parameters at each point in the 2d space that the
    trajectory spans (split up into segments). You should generate this with a
    high number of points.

    drag_coeff is the drag coefficient of the trapped object

    init_guess are initial guesses of the x and y offset.

    It is assumed that you have separated data_df into a single trajectory!

    You are trying to maximise the result of optimise wrapper

    """

    # I have written my own minimisation function because the other ones are taking
    # weird jumps.

    # Should probably look for a more optimal boi...

    #Constuct a guess matrix:
    x_guesses = np.linspace(init_guess[0]-inc_range*inc_size, init_guess[0]+inc_range*inc_size, 2*inc_range+1)
    y_guesses = np.linspace(init_guess[1]-inc_range*inc_size, init_guess[1]+inc_range*inc_size, 2*inc_range+1)
    print(len(x_guesses))
    print(len(y_guesses))
    final_resid = -100000000000
    final_out = [-1000000,-1000000]
    resid_list = list() 
    Xl = list()
    Yl = list()
    num_pass_list = list()
    non_zero_list = list()
    for idx, x in enumerate(x_guesses):
        for y in y_guesses:
            resid, num_pass, non_zero = optimise_wrapper([x, y], data, smesh_df, drag_coeff, logger)
            resid_list.append(resid)
            Xl.append(x)
            Yl.append(y)
            num_pass_list.append(num_pass)
            non_zero_list.append(non_zero)
            if resid > final_resid:
                final_resid = np.copy(resid)
                final_out = [x,y]

    logger.debug(final_resid)
    
    #Plot a 3D figure of this trajectory
    # Convert to a mesh format (needed for surface plot)
    
    cols = np.unique(Xl).shape[0]
    X = np.array(Xl).reshape(-1, cols)
    Y = np.array(Yl).reshape(-1, cols)
    Z = np.array(resid_list).reshape(-1, cols)
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    # plot the trajectory
    z_offset = [final_resid + 10 for i in range(data.x_pos.size)]
    ax.plot(data.x_pos, data.y_pos, z_offset)
    
    #ax = fig.add_subplot(122, projection='3d')
    #Z = np.array(non_zero_list).reshape(-1, cols)
    #ax.plot_surface(X, Y, Z)
    
    plt.show()
    
    return final_out
"""
    # Plot the smesh mesh and the trajectory
     # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data.x_pos, data.y_pos, z_offset)
    # Convert smesh to a mesh format (needed for surface plot)
    Z = np.array(np.linalg.norm([smesh_df['Fx'],smesh_df['Fy']],axis=0))
    scale = (z_offset[0]-10)/max(Z)
    ax.plot_trisurf(smesh_df['x-Center']*10**-6, smesh_df['y-Center']*10**-6, Z)
    plt.show()
"""


def optimise_wrapper(guess, data, smesh_df, drag_coeff, logger):
    """
    Just a wrapper for find_cum_prob that fits scipy_optimize

    """

    # Apply our guesses to the offset (create a copy first!!)
    temp_df = data.copy()
    temp_df.x_pos = data.x_pos + guess[0]
    temp_df.y_pos = data.y_pos + guess[1]

    result, num_pass, non_zero = find_cum_prob_compact(temp_df, smesh_df, drag_coeff, logger)
    logger.debug('Guess of ' + str(guess) + ' returns ' + str(result))

    return result, num_pass, non_zero



def find_cum_prob(data, smesh_df, drag_coeff, logger):
    """
    find_offset is the core function to evaluate the offset of
    a trajectory from the known brownian motion.
        data_df is the trajectory data file, containing:
            x_pos, x position trajectory, in nanometres
            y_pos, position trajectory, in nanometres
            time, time of each position sample, in seconds
        smesh_df is a pandas data file with the brownian motion data
            it is generated by the inferenceMAP program.

    Probabilites are returned such that a low number constitutes a high probability
    (because scipy optimize likes to minim ise things, not maximise)
    """
    num_pass = 0
    del_x_list = list()
    del_y_list = list()
    total_list = list()
    for i in range(len(data.x_pos)):
    # for i in range(200):
        # Don't do this for the last value
        if i != len(data.x_pos)-1:
            logger.debug("Step" + str(i))
            del_x = (data.x_pos.iloc[i+1] - data.x_pos.iloc[i])
            del_y = (data.y_pos.iloc[i+1] - data.y_pos.iloc[i])
            del_t = data.time.iloc[i+1] - data.time.iloc[i]
            D, Fx, Fy = find_forces(x=data.x_pos.iloc[i],
                                    y=data.y_pos.iloc[i],
                                    smesh_df=smesh_df,
                                    logger=logger)
            #logger.debug('D is ' + str(D) +
            #            ' Fx is ' + str(Fx) + ' Fy is ' + str(Fy))
            if D != 0:
                # If D = 0, then our point is out-of-bounds of our setup
                new_prob = calc_prob(del_x, del_y, del_t, D, Fx, Fy, drag_coeff, logger)
                #logger.debug('Del_x is ' + str(del_x) + ' Del_Y is ' + str(del_y) +
                #             ' Prob is ' + str(new_prob))
                #logger.debug(' ')
                del_x_list.append(del_x)
                del_y_list.append(del_y)
                total_list.append(np.linalg.norm([del_x, del_y]))
                try:
                    cum_prob = cum_prob + new_prob
                    if new_prob != 0:
                        num_pass += 1
                        # We need to keep track of how many successful trajectories
                        # we got
                except NameError:
                    cum_prob = new_prob
                except UnboundLocalError:
                    cum_prob = new_prob

    try:
        result = cum_prob/num_pass
    except NameError:
        result = 0
    except ZeroDivisionError: 
        result = 0

    return result

def find_cum_prob_compact(data, smesh_df, drag_coeff, logger):
    """
    This finds the cumulative probability of a trajectory but in a quicker way.
    Essentially a vectorised / somewhat optimised version of the find_cum_prob function
    """

    del_t = 5*10**-6 # Seconds, need to remove this hard code later

    # Cast x and y positions into numpy arrays:
    x_pos = np.array(data.x_pos)*10**6 #Cast as um from m
    y_pos = np.array(data.y_pos)*10**6 #Cast as um

    # Form a positions array
    positions = [(x,y) for (x,y) in zip(x_pos,y_pos)]

    # Compute the differences (jumps!)
    del_x = np.diff(x_pos)*10**-6 # Cast back to m
    del_y = np.diff(y_pos)*10**-6 # Cast back to m

    ### Find the Force X, Force Y and Diffusion at each point:
    # Firstly, find our nearest grid points:
    # Parse smesh locations as ND arrays

    # Extract values from the smesh df to be used in scipy's interpolator
    X = smesh_df['x-Center'].values
    Y = smesh_df['y-Center'].values
    Fx =smesh_df['Fx'].values*10**-12
    Fy =smesh_df['Fy'].values*10**-12
    D = smesh_df['D'].values*10**-12

    # Interpolate the Fx and Fy positions
    Fx_int = interp.griddata((X, Y), Fx, positions, method='nearest')
    Fy_int = interp.griddata((X, Y), Fy, positions, method='nearest')
    D_int = interp.griddata((X, Y), D, positions, method='nearest')

    # Finally, set the Forces and Diffusion constants of any point outside the smesh
    x_max = np.max(np.unique(X))
    y_max = np.max(np.unique(Y))
    x_min = np.min(np.unique(X))
    y_min = np.min(np.unique(Y))


    sigma = 30*10**-9 # Uncertainty in position, nanometre

    # Compute probability set
    """
    prob_x = np.exp(-((del_x-Fx*del_t/drag_coeff)**2)/(4*(D+(sigma**2)/del_t)*del_t))/(4*(D+(sigma**2)/del_t)*del_t)
    prob_y = np.exp(-((del_y-Fy*del_t/drag_coeff)**2)/(4*(D+(sigma**2)/del_t)*del_t))/(4*(D+(sigma**2)/del_t)*del_t)

    prob_combined = np.log(prob_x*prob_y)
    print(del_x[0],Fx[0],del_t,D[0],prob_x[0])
    """
    prob_combined = list()

    # Some debug parameters to see how many successful inferences we get in a given area
    num_pass = 0        # How many jumps fit in the smesh?
    non_zero = 0         # How many jumps returned a non-zero probability?
    for i in range(len(x_pos)-1):
        # Test the old function
        if ((x_pos[i] < x_max) & (x_pos[i] > x_min) & (y_pos[i] < y_max) & (y_pos[i] > y_min)):
            A = calc_prob(del_x[i], del_y[i], del_t, D_int[i], Fx_int[i], Fy_int[i], drag_coeff, logger)
            num_pass += 1
            prob_combined.append(A)
            if A != 0:
                non_zero += 1
 #       else:
 #           prob_combined.append(-10)

    cum_prob = np.sum(prob_combined)
    logger.debug(num_pass)
    return cum_prob, num_pass, non_zero




def calc_prob(del_x, del_y, del_t, D, Fx, Fy, drag_coeff, logger):
    """
    Calculates the probability of a transition based on
    the fokker plank equation.

    Does it one step at a time because that's a little easier

    Also returns logarithms because I don't want to underflow my life away

    del_x is in metres
    del_y is in metres
    del_t is in seconds
    D is in m^2/s
    Fx is in N
    Fy is in N
    drag is in N/(m/s)
    """
    sigma = 30*10**-9 # Uncertainty in position, nanometres
    prob_x = np.exp(-((del_x-Fx*del_t/drag_coeff)**2)/(4*(D+(sigma**2)/del_t)*del_t))/(4*(D+(sigma**2)/del_t)*del_t)
    prob_y = np.exp(-((del_y-Fy*del_t/drag_coeff)**2)/(4*(D+(sigma**2)/del_t)*del_t))/(4*(D+(sigma**2)/del_t)*del_t)
    #print(prob_y)
    logger.debug('Del_x is' + str(del_x) + 'F_x norm is ' + str(Fx*del_t/drag_coeff))
    try:
        new_prob = math.log(prob_y*prob_x)
    except ValueError:
        new_prob = 0
    #logger.debug(new_prob)
    #print(del_x,Fx,del_t,D,new_prob)
    return new_prob


def find_forces(x, y, smesh_df, logger):
    """
    A function to find the forces from the appropriate force constants
    given an x and y position, reading from the smesh data file. 
    It simply finds the closest data points to the x and y co-rds and returns
    the diffusion constant and force constants.


    I am working on the FOLLOWING assumptions about the smesh file:
        Diffusion constant is in micrometers squared per second
        Forces are in pico newtons
    """
    x_um = x*10**6   # convert from metres to micrometres
    y_um = y*10**6
    side_length = 50*10**-3 # in micrometers
    best_id = np.argmin(np.abs(smesh_df['x-Center']-x_um) +
                        np.abs(smesh_df['y-Center']-y_um))
    if (np.abs(x_um - smesh_df['x-Center'].loc[best_id]) > side_length)\
        or (np.abs(y_um - smesh_df['y-Center'].loc[best_id]) > side_length):
        logger.debug('SKIPPED x, y of ' + str(x_um) + ', ' + str(y_um)
            + 'um map to centres of ' +
            str(smesh_df['x-Center'].loc[best_id]) + ' ' +
            str(smesh_df['y-Center'].loc[best_id]))
        diffusion = 0
        force_x = 0
        force_y = 0
    else:
        logger.debug('x, y of ' + str(x_um) + ', ' + str(y_um)
                     + 'um map to centres of ' +
                     str(smesh_df['x-Center'].loc[best_id]) + ' ' +
                     str(smesh_df['y-Center'].loc[best_id]))

        diffusion = smesh_df['D'].loc[best_id]*10**-12  # Convert to um
        force_x = smesh_df['Fx'].loc[best_id]*10**-12  # Convert to N
        force_y = smesh_df['Fy'].loc[best_id]*10**-12   # Convert to N

    return diffusion, force_x, force_y


def find_forces_spring(x, y, smesh_df, logger):
    # Finds the forces assuming just a spring system, hard coding some values
    # This is just for testing
    stiffness_x = 2.31*10**-6    # N/m
    stiffness_y = 2.31*10**-6
    diffusion = 4.392*10**-12  # m^2/s

    force_x = -stiffness_x*x
    force_y = -stiffness_y*y

    #logger.debug('x, y of ' + str(x) + ', ' + str(y))
    return diffusion, force_x, force_y


def mp_wrapper(traj_IDs, data_df, smesh_df, drag_coeff, range_points, mo_out, am_out, inc_size, inc_range):
    """A wrapper for multiprocessing - automatically extracts the 
    required data and calles the find_offset function."""

    # Create a logging object for this specific MP process
    logger = setup_logger('LogfileCore%s' %traj_IDs[0],str(os.getcwd())+'/temp/LogfileCore%s.log' %traj_IDs[0])
    logger.info('Logger Test')
    logger.setLevel('DEBUG')
    mean_offset=list()
    arithmatic_mean=list()
    for i in traj_IDs:
        if type(i) == int:
            # This is the case where each ID indicates what subset of the giant trajectory to run.
            print("Running trajectory " + str(i))
            temp_df = data_df[i*range_points:(i+1)*range_points-1]
            mean_offset.append(find_offset(temp_df, smesh_df, drag_coeff, [0,0], logger, inc_size, inc_range))
            x_mean = (np.mean(data_df[i*range_points:(i+1)*range_points-1].x_pos))
            y_mean = (np.mean(data_df[i*range_points:(i+1)*range_points-1].y_pos))
            arithmatic_mean.append([x_mean, y_mean])
        else:
            # This is the case where each ID inidicates what trap ID to run.
            x_goal = data_df.x_centre.unique()[i[0]]
            y_goal = data_df.y_centre.unique()[i[1]]
            print("Running trajectory centered at " + str(x_goal) + str(y_goal))
            temp_df = data_df[((data_df.x_centre == x_goal) & ((data_df.y_centre == y_goal)))]
            mean_offset.append(find_offset(temp_df, smesh_df, drag_coeff, [0,0], logger, inc_size, inc_range))
            x_mean = (np.mean(data_df[i*range_points:(i+1)*range_points-1].x_pos))
            y_mean = (np.mean(data_df[i*range_points:(i+1)*range_points-1].y_pos))
            arithmatic_mean.append([x_mean, y_mean])
    
    mo_out.put(mean_offset)
    am_out.put(arithmatic_mean)

def setup_logger(name_logfile, path_logfile):
        # Stolen from https://stackoverflow.com/questions/28478424/multiprocessing-how-to-write-separate-log-files-for-each-instance-while-using-p
        logger = logging.getLogger(name_logfile)
        formatter = logging.Formatter('%(asctime)s:   %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        fileHandler = logging.FileHandler(path_logfile, mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        logger.setLevel(logging.INFO)
        logger.addHandler(fileHandler)
        return logger

#################################
# IN-SCRIPT IMPLEMENTATION
#################################
#logging.basicConfig(level=logging.INFO,
#                    filename=str(os.getcwd())+'/temp/infer_offset.log',
#                    filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')
logger.info('Beginning of logging')
logger.debug('Debug mode')

# Getting smesh'd (lol)
#smesh_loc = 'iMAPBayes_Test_4.smesh'
smesh_loc = 'iMAPAug27-180-Second.smesh'
#smesh_loc = 'second_iMAPAUg9-083.smesh'
""" IMPORTANT:
You need to modify the smesh output to only have ONE TAB between
column headings
"""
logger.info('Reading smesh file')
df = pd.read_table(smesh_loc, skiprows=26)
smesh_df = df.drop(df[df.Active == 0].index)
# filter out the inactive rows.

# You need to find the drag coefficent of the probe
# It's output in my simulations
drag_coeff = 9.4247*10**-10  # kg/s

# Getting input data
#input_loc = 'Bayes_Test_4_CIRCLEcombo_out.txt'
input_loc = '../NSM/data/OLD DATA/Aug27-18/Exp0.dat'
#input_loc = '../NSM/data/AUg9-08/Exp3.dat'
# IF you are using an experimental data set, it doesn't save with a time vector
# so you need to specify what time step you are using.
time_step = 5*10**-6 

"""
The column format of experiments is a little different ot that of simulations
"""
experimental = True

""" Do you want to drop the x_centre, y_centre columns? 
This allows you to pick a subset of the trajectory to run, instead of 
needing to run each trap position as a full trajectory"""
manual_drop = True

# Some search parameters
inc_size = 5*10**-9 # increment size, metres.
inc_range = 40


# Depending on if you are doing a Minor Verlet simulation (one trap pos)
# or a major / experiment (many trap positions), you have a different number of columns
logger.info('Reading Data File')
data_df = pd.read_table(input_loc, header=0)
if len(data_df.columns) == 6:
    if experimental:
        data_df.columns = ['x_pos', 'y_pos', 'z_pos',
                           'intensity', 'x_centre', 'y_centre']
        time_vector = np.linspace(0, time_step*(len(data_df.index)-1), len(data_df.index))
        data_df['time'] = pd.Series(time_vector)
        data_df.x_pos = data_df.x_pos*10**-9 # Convert into metres
        data_df.y_pos = data_df.y_pos*10**-9 # Convert into metres
        logger.debug(data_df)
        logger.debug(data_df.x_pos.mean())     
    else:
        data_df.columns = ['time', 'x_pos', 'y_pos',
                           'intensity', 'x_centre', 'y_centre']        
elif len(data_df.columns) == 4:
    data_df.columns = ['time', 'x_pos', 'y_pos', 'intensity']
elif len(data_df.columns) == 7:
    if experimental:
        data_df.columns = ['x_pos', 'y_pos', 'z_pos',
                           'intensity', 'x_centre', 'y_centre', 'write_index']
        time_vector = np.linspace(0, time_step*(len(data_df.index)-1), len(data_df.index))
        data_df['time'] = pd.Series(time_vector)
        data_df.x_pos = data_df.x_pos*10**-9 # Convert into metres
        data_df.y_pos = data_df.y_pos*10**-9 # Convert into metres
        logger.debug(data_df)
        logger.debug(data_df.x_pos.mean())     

else:
    sys.exit('DATA FILE WITH INCORRECT NUMBER OF COLUMNS')

if manual_drop:
    data_df = data_df.drop(['x_centre', 'y_centre'], axis=1)


#################
# TESTING
#################
# I am going to compare the geometric mean vs inferred centre for a
# 200,000 point trajectory in 200 point increments. 
if __name__ == "__main__":
# Parse the data into separate sections based on x_centre and y_centre.
    mo_out = mp.Queue()
    am_out = mp.Queue()
    i=0 # Counter for outputing run ID
    range_points = 200  # Points per trajectory
    max_traj = 20# Maximum number of trajectories to try
    max_cores = 4 # Number of cores to use in parallel processing
    rand_traj = True # Output random trajectories
    random.seed(25)
    if 'x_centre' in data_df.columns:
        # STILL NEED TO DEBUG THIS PART
        # Generate sets of trajectories for each process
        proc_list = [list() for i in range(max_cores)]
        # Generate a list of trap positions
        trap_sets = [(x, y) for x in data_df.x_centre.unique() for y in data_df.y_centre.unique()]
        # Assign trajectories to each core
        for i in range(min(len(trap_sets),max_traj)):
            proc_list[i%max_cores].append(trap_sets[i])
        print(proc_list)

        # Farm out the processing to each core based on the posible trajectories
        processes = [mp.Process(target=mp_wrapper,
                                args=(x, data_df, smesh_df, drag_coeff, range_points, mo_out, am_out, inc_size, inc_range))
                    for x in proc_list]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        # Extract our output
        mean_offset_temp = [mo_out.get() for p in processes]
        arithmatic_mean_temp = [am_out.get() for p in processes]
        
        # Flatten the output lists
        mean_offset = list(itertools.chain.from_iterable(mean_offset_temp))
        arithmatic_mean = list(itertools.chain.from_iterable(arithmatic_mean_temp))
    else:
        # Generate sets of trajectories for each multiprocess
        proc_list = [list() for i in range(max_cores)]
        for j in range(min(data_df.index.size//range_points,max_traj)):
            if rand_traj:
                proc_list[j%max_cores].append(random.randint(0,data_df.index.size//range_points-1))
            else:
                proc_list[j%max_cores].append(j)
        print(proc_list)
        # Farm out the processing to each core based on the posible trajectories
        processes = [mp.Process(target=mp_wrapper,
                                args=(x, data_df, smesh_df, drag_coeff, range_points, mo_out, am_out, inc_size, inc_range))
                     for x in proc_list]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

        # Extract our output
        mean_offset_temp = [mo_out.get() for p in processes]
        arithmatic_mean_temp = [am_out.get() for p in processes]
        
        # Flatten the output lists
        mean_offset = list(itertools.chain.from_iterable(mean_offset_temp))
        arithmatic_mean = list(itertools.chain.from_iterable(arithmatic_mean_temp))


    
    #%%
    # Printing Results
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    x_set = [i[0]*10**9 for i in arithmatic_mean]
    y_set = [i[1]*10**9 for i in arithmatic_mean]
    x_guess_set = [i[0]*10**9 for i in mean_offset]
    y_guess_set = [i[1]*10**9 for i in mean_offset]
    x_derived = np.array(x_set)+np.array(x_guess_set)
    y_derived = np.array(y_set)+np.array(y_guess_set)
    ax.scatter(x_set, y_set)
    plt.xlabel('X mean, nanometres')
    plt.ylabel('Y mean, nanometres')
    plt.title([str(range_points) + ' sample subset of ND'])
    ax.scatter(data_df.x_pos.mean(), data_df.y_pos.mean())
    ax.scatter(x_guess_set, y_guess_set)
    ax.legend(['Geometric Means', 'Trap Centre', 'Derived Position'])
    x_mean = data_df.x_pos.mean()
    y_mean = data_df.y_pos.mean()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    x_diff = np.abs(np.array(x_set)-x_mean) - np.abs(np.array(x_guess_set)-x_mean)
    y_diff = np.abs(np.array(y_set)-y_mean) - np.abs(np.array(y_guess_set)-y_mean)
    ax2.scatter(x_diff,y_diff)
    ax2.axhline()
    ax2.axvline()
    plt.title('Mean Vs Bayes, (+ve = Bayes Better)')
    plt.xlabel('X Difference,')
    plt.ylabel('Y Difference, metres')
    plt.show()


    