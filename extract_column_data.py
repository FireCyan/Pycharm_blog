
import pandas as pd
import gpxpy
import gpxpy.gpx

def get_run_data(run_dataframe, gpx, col_name):

    import gpxpy
    import gpxpy.gpx
    import matplotlib.pyplot as plt
    from math import radians, cos, sin, asin, sqrt
    import numpy as np
    from datetime import datetime
    import itertools # for using compress function
    import math
    import xlwt
    import pandas as pd


    def haversine(lon1, lat1, lon2, lat2):
        """
            Calculate the great circle distance between two points
            on the earth (specified in decimal degrees)
            """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2*asin(sqrt(a))
        km = 6367*c
        return km



    ## Extracting data
    lat = [] # latitute
    lon = [] # longitude
    time_datetime = [] # for storing time in date time format
    time = [] # for calculating time difference and running rate
    time_diff = [] # for storing time difference
    time_relative = [] # relative time to the start time
    rate = [] # running rate
    dist_diff = [] # distance between each subsequent GPS point
    elev = [] # elevation
    km_per_min = [] # minute for 1 km
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                lon_prev = []
                lat_prev = []
                time_prev = []
                if lat: # To check if it is the first entry, for getting latitude difference
                    lat_prev = lat[-1] # get previous latitude
                lat.append(point.latitude)
                lat_curr = lat[-1] # get current latitude
                if lon: # To check if it is the first entry, for getting longitude difference
                    lon_prev = lon[-1] # get previous longitude
                lon.append(point.longitude)
                lon_curr = lon[-1] # get current longitude
                if time: # To check if it is the first entry, for getting time difference
                    time_prev = time[-1]
                time.append(point.time.timestamp())
                time_curr = time[-1]
                if lon_prev:
                    dist_diff.append(haversine(lon_prev, lat_prev, lon_curr, lat_curr)) # in km
                if time_prev:
                    time_diff.append(time_curr - time_prev) # in s
                    rate_temp = dist_diff[-1]/time_diff[-1]*1000 # in m/s
                    rate.append(rate_temp)
                    if rate[-1] != 0:
                        km_per_min.append(1/(rate[-1]/1000*60)) # km/min
                    else:
                        km_per_min.append(0)
                        #print(rate[-1])
                        #print(km_per_min[-1])
                time_datetime.append(point.time)
                elev.append(point.elevation)
                # print(point)

    time_relative = [(t - time[0]) / 60 for t in time[1:]]
    elev_diff = np.diff(elev)

    #print(time_datetime[0].date())

    # Separate pause and run
    # numpy method
    '''
    time_diff = np.array(time_diff)
    km_per_min = np.array(km_per_min)

    time_c = (time_diff < 20)
    kmpm_slow_c = (km_per_min < 10)
    kmpm_fast_c = (km_per_min > 2.5)
    '''

    # List method
    time_c = [t < 20 for t in time_diff] # Satisfy the criteria that time difference is less than 20s
    kmpm_slow_c = [k < 8 for k in km_per_min] # Satisfy the criteria that km per minute is less than 10 min per km (othersie = walking)
    kmpm_fast_c = [k > 2.5 for k in km_per_min] # Satisfy the criteria that km per minute is more than 2.5 min per km (WD = 2'11)
    # dist_upper_c = [k > 2.5 for k in km_per_min] # Satisfy the criteria that distance is less than

    mask = [all(tup) for tup in zip(time_c, kmpm_slow_c, kmpm_fast_c)]

    '''getting all the column data to write into .csv file'''

    run_date = time_datetime[0].date()
    run_year = str(time_datetime[0].year)
    run_month = str(time_datetime[0].month)
    start_time = time_datetime[0].time()
    end_time = time_datetime[-1].time()

    # print(run_year)
    # print(run_month)

    # Remove pause from run time and distance
    time_relative_adj = list(itertools.compress(time_relative,mask))
    km_per_min_adj = list(itertools.compress(km_per_min, mask))
    time_diff_adj = list(itertools.compress(time_diff,mask))
    elev_diff_adj = list(itertools.compress(elev_diff,mask))
    elev_adj = list(itertools.compress(elev[1:],mask))
    dist_diff_adj = list(itertools.compress(dist_diff,mask))

    dist_acc = np.cumsum(dist_diff) # accummulated distance

    time_diff_pause = list(itertools.compress(time_diff,np.invert(mask)))
    dist_diff_pause = list(itertools.compress(dist_diff,np.invert(mask)))
    dist_pause = list(itertools.compress(dist_acc,np.invert(mask))) # distance at which pauses occurred

    total_time = time_relative_adj[-1] - time_relative_adj[0] # in mins
    total_distance = np.sum(dist_diff) # in kn

    total_run_time = np.sum(time_diff_adj)/60 # in mins
    total_run_distance = np.sum(dist_diff_adj)

    #total_pause_time = total_time - total_run_time

    total_pause_time = np.sum(time_diff_pause)/60 # in min
    total_pause_distance = np.sum(dist_diff_pause) # in km
    avg_run_rate = np.mean(km_per_min_adj) # min/km, only run time
    avg_total_rate = np.mean(km_per_min)

    elev_diff_adj = np.array(elev_diff_adj)

    total_uphill = np.sum(elev_diff_adj[elev_diff_adj > 0])
    total_downhill = np.sum(elev_diff_adj[elev_diff_adj < 0])


    # print(total_run_distance)
    # print(total_run_time)
    # print(total_pause_time)
    # print(total_pause_distance)
    # print(total_time)
    # print(avg_run_rate)
    # print(avg_total_rate)
    # print(total_uphill)
    # print(total_downhill)


    def get_run_rate(total_dist, dist_interval):
        n_group = math.ceil(total_dist[-1]/dist_interval)
        group = []
        avg_run_rate = []
        avg_elev = []
        for i in range(n_group):
            group.append(i)
            lower_lim = i*dist_interval
            upper_lim = (i + 1)*dist_interval
            total_dist_lower_c = [t >= lower_lim for t in total_dist]
            total_dist_upper_c = [t < upper_lim for t in total_dist]
            mask = [all(tup) for tup in zip(total_dist_upper_c, total_dist_lower_c)]
            temp_run_rate = list(itertools.compress(km_per_min_adj, mask))
            temp_elev = list(itertools.compress(elev_adj, mask))
            avg_run_rate.append(np.mean(temp_run_rate))
            avg_elev.append(temp_elev[-1] - temp_elev[0]) # Get the relative elevation within the distance run
        return group, avg_run_rate, avg_elev

    total_dist_adj = np.cumsum(dist_diff_adj)*1000 # convert to meters
    group, avg_run_rate, avg_elev = get_run_rate(total_dist_adj, 100) # Calculate run rate for each 100m interval
    group = np.array(group)

    tag_100m = group
    ind_100m_rate = avg_run_rate
    ind_100m_elev = avg_elev
    ind_percent_run = group/10/total_run_distance


    # print(tag_100m)
    # print(ind_100m_rate)
    # print(ind_100m_elev)
    # print(ind_percent_run)


    valid_pause = [t > 15 for t in time_diff_pause]
    ind_pause_time = list(itertools.compress(time_diff_pause, valid_pause))
    dist_diff_pause_acc = np.cumsum(dist_diff_pause) # accumulating the distance during pause
    dist_pause_valid = list(itertools.compress(dist_pause, valid_pause)) # only include pause that has long enough pause time, checked by the variable 'valid-pause'
    dist_diff_pause_acc_valid = list(itertools.compress(dist_diff_pause_acc, valid_pause))


    dist_pause_adj = np.subtract(dist_pause_valid, dist_diff_pause_acc_valid) # get the distance at which pause occurred

    # print(ind_pause_time)
    # print(dist_diff_pause_acc)
    # print(dist_pause_adj)

    dist_pause_adj = np.array(dist_pause_adj)
    tag_pause_distance = np.floor(dist_pause_adj*10)

    # print(tag_pause_distance)




    run_overview = xlwt.Workbook()
    sh = run_overview.add_sheet('1')


    temp_series = []
    for c in col_name:
        temp_val = eval(c.lower())
        temp_series.append(temp_val)

    run_dataframe.loc[0] = temp_series


    return run_dataframe

col_name = ['Run_date', 'Run_year', 'Run_month', 'Total_time', 'Total_distance', 'Total_run_time', 'Total_run_distance', 'Start_time', 'End_time', 'Total_pause_time',
                'Total_pause_distance', 'Total_uphill', 'Total_downhill']
run_dataframe = pd.DataFrame(columns=col_name)

gpx_file = open('20170216-213821-Run.gpx', 'r', encoding='utf-8')
gpx = gpxpy.parse(gpx_file)
# print(gpx)

run_dataframe = get_run_data(run_dataframe, gpx, col_name)
print(run_dataframe)

## Open file
