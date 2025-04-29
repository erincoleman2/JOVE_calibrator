# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import datetime
from uncertainties import ufloat as u


# tell the code where to locate the file
file_path = "250317171747-684-GAC.csv"

# read it in as a pandas dataframe
cal_run = pd.read_csv(file_path)

# turn the time and date stamps into a list of times in seconds since 
# the beginning of the run
dates = cal_run["Date"]
times = cal_run["Time"]
del cal_run["Date"]
del cal_run["Time"]
for i in range(len(times)):
    x = datetime.time.fromisoformat(times[i])
    y = datetime.datetime.strptime(dates[i], '%Y/%m/%d')
    x= datetime.datetime.combine(y, x)
    times[i] = x.timestamp()
times = np.array(times-times[0])

# flatten by using mean of all frequency channels at each timestamp
means = cal_run.aggregate('mean',axis = 1)

# get a rolling median of the aggregated intensity data, (only over 5 samples or ~0.3 seconds)
# this smooths electrical noise
s = pd.Series(means)
rolling_median = s.rolling(5).median()
rolling_median = np.nan_to_num(rolling_median)


# define t=0 at the timestamp where the calibration begins
t0 = times[np.argmax(np.array(means[1:])-np.array(means[:-1]))]

# define length of each calibration step in seconds
step_length = 5

def step(x, loc, height):
    """Given an x value, the location of a step transition, and a step height,
    return a Heaviside-like sigmoid function."""
    return_list = []
    for x0 in x:
        return_list.append(height/2* (erf((x0-loc)*30)+1))
    return return_list

def multistep(x, t0, step_length, a127, a0, a3, a6, a9, a12, a15, a18, a21, a24, a27, a30, a33, a36, a39, a42):
    """Use a sum of step functions to model the calibration sequence of this calibrator."""
    params = [a127, a0, a3, a6, a9, a12, a15, a18, a21, a24, a27, a30, a33, a36, a39, a42]
    y  = np.zeros_like(x)

    # sum over each step
    for i, A in enumerate(params):
        y+= step(x, t0+i*step_length, A)

    return list(y)

def dB_to_power_ratio(dB):
    """Converts the decibel value of attenuation to the power ratio."""
    return 10 **(dB/10)

# provide initial guesses for height parameters
p0 = [30, 5, 0, 8050, 7600, 6900, 6300, 5500, 4700, 4000, 3300, 2700, 2000, 1300, 900, 500, 200, 70]

# perform fit and get the fit values and covariance matrix
vals, cov = curve_fit(multistep, np.array(times), rolling_median, p0)
# get standard deviation in each parameter from covariance matrix
stdevs = np.sqrt(cov.diagonal())

# plot the time series data and fitted values
plt.plot(times, means, label = 'data')
plt.plot(times,multistep(times,*vals),label = 'fit')
plt.xlabel("Time (sec)")
plt.ylabel("Intensity")
plt.grid(visible =True, which = 'major')
plt.legend()
plt.show()

# calculate the ADC values for each attenuation value and print them
labels = ["Calibration start time", "Step duration", "127 dB", "0 dB", "3 dB", "6 dB", "9 dB", "12 dB", "15 dB", "18 dB", "21 dB", "24 dB", "27 dB", "30 dB", "33 dB", "36 dB", "39 dB", "42 dB"]
attenuation_vals = [-127, 0, -3, -6, -9,-12, -15, -18, -21, -24, -27, -30, -33, -36, -39, -42]
cumulative_amp = u(0,0)
params = []
params_uncertainty = []
for label, val, stdev in zip(labels, vals, stdevs):
    if label not in labels[0:2]:
        cumulative_amp+=u(val,stdev)
        print_val = u(cumulative_amp.n, stdev)

        # get the parameter value into a list
        params.append(print_val.n)
        params_uncertainty.append(print_val.s)

        print(label, "\t", print_val)
    else:
        print(label, "\t", u(val,stdev) )

[a127, a0, a3, a6, a9, a12, a15, a18, a21, a24, a27, a30, a33, a36, a39, a42] = params


# plot ADC value vs attenuation
fig, ax = plt.subplots()
plt.sca(ax)
plt.errorbar(np.array(attenuation_vals[1:],dtype=float), params[1:], yerr=params_uncertainty[1:], capsize=5, marker='o',linestyle='none')
plt.xlabel("Attenuation value (dB)")
plt.ylabel("ADC value")
plt.show()
