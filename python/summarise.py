import numpy as np
import os
import sys
import csv
from tabulate import tabulate
        
def match_result_to_str(match_result):
    result = ""
    for i in range(0, len(match_result)):
        result = result + "{:.2f}".format(match_result[i]) + ","
    return result

csv_file = sys.argv[1]

results = np.zeros([3, 10, 6], dtype=float)
level_finish_time_sum = np.zeros([3], dtype=float)
level_finish_count = np.zeros([3], dtype=float)

with open(csv_file) as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
    
        if line_count > 0:
        
            if int(row[4]) > 0:
                level_finish_time_sum[int((line_count - 1) / 10)] += float(row[8])
                level_finish_count[int((line_count - 1) / 10)] += 1
                
            results[int((line_count - 1) / 10)][(line_count - 1) % 10][0] = row[4]
            results[int((line_count - 1) / 10)][(line_count - 1) % 10][1] = row[5]
            results[int((line_count - 1) / 10)][(line_count - 1) % 10][2] = row[6]
            results[int((line_count - 1) / 10)][(line_count - 1) % 10][3] = row[7]
            results[int((line_count - 1) / 10)][(line_count - 1) % 10][4] = row[12]
            results[int((line_count - 1) / 10)][(line_count - 1) % 10][5] = row[13]
            
        line_count += 1

mean = np.mean(results, axis=1)
mean_finish_time = level_finish_time_sum / level_finish_count

print()

for idx2 in range(1, 3):
    print(tabulate([['Initial', 100 * mean[0][0], int(100 * mean_finish_time[0]) if (not np.isnan(mean_finish_time[0])) else 'N/A', 100 * mean[0][1], 100 * mean[0][2], 100 * mean[0][3], mean[0][4], mean[0][5]], ['After ' + str(idx2) + ' round(s)', 100 * mean[idx2][0], int(100 * mean_finish_time[idx2]) if (not np.isnan(mean_finish_time[idx2])) else 'N/A', 100 * mean[idx2][1], 100 * mean[idx2][2], 100 * mean[idx2][3], mean[idx2][4], mean[idx2][5]]], headers=['Run', 'Finish lvl %', 'Finish time (s)', 'Mushroom %', 'Yoshi %', 'Secret %', 'Coins', 'Enemies']))
    print()
    if idx2 == 1:
        input("Press ENTER to compare next round...")
        print("")
    
    #, 'Finish time (s)', 'Mushroom?', 'Yoshi?', 'Secret?', 'Coins', 'Enemies'
