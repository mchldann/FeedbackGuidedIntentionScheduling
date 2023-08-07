import numpy as np
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import csv
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument("-i", "--input_dir", type=str, nargs='?', const=True, required=True, help="Root directory for the log files.")
parser.add_argument("-n", "--noise", type=str2bool, nargs='?', const=True, default=False, help="Indicates second type of experiment (varying noise level).")

trimmed_args = sys.argv.copy()
del trimmed_args[0]
parsed_args = parser.parse_args(trimmed_args)

parsed_args.input_dir = parsed_args.input_dir + '/'
#parsed_args.input_dir = '/home/mickey/SchedulerCode/InferPayoffMatrix/log/GOAL_ACHIEVEMENT_SIMPLE_2021-08-19_22-44-23/'

if parsed_args.noise:
    feedback_interval = 5
    SUB_DIRS = ['noise_0_25', 'noise_0_5', 'noise_1_0', 'noise_2_0']
    SCHED_NAMES = ['Noise 0.25', 'Noise 0.5', 'Noise 1.0', 'Noise 2.0']
    FIRST_EPOCH_LEN = [feedback_interval, feedback_interval, feedback_interval, feedback_interval]
    EPOCH_LEN = [feedback_interval, feedback_interval, feedback_interval, feedback_interval]
    TRAINING_SAMPLES_PER_EPOCH = [feedback_interval, feedback_interval, feedback_interval, feedback_interval]
else:
    SUB_DIRS = ['feedback_freq_100', 'feedback_freq_25', 'feedback_freq_5']
    SCHED_NAMES = ['Feedback after 100', 'Feedback after 25', 'Feedback after 5']
    FIRST_EPOCH_LEN = [100, 25, 5]
    EPOCH_LEN = [100, 25, 5]
    TRAINING_SAMPLES_PER_EPOCH = [100, 25, 5]
    
RESULTS_FILENAME = 'match_results_for_python.csv'
MAX_X = 300

# Read baseline score
baseline_score = None
with open(parsed_args.input_dir + 'baseline/stats.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 1:
            baseline_score = float(row[0])
        line_count += 1
        
# Read score with oracle
score_with_oracle = None
with open(parsed_args.input_dir + 'oracle/stats.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 1:
            score_with_oracle = float(row[0])
        line_count += 1
                            
x_vals = {}
y_vals = {}
for i in range(0, len(SUB_DIRS)):
    x_vals[SUB_DIRS[i]] = []
    y_vals[SUB_DIRS[i]] = []
    for subdir, dirs, files in os.walk(parsed_args.input_dir + 'learner/' + SUB_DIRS[i]):
        for file in files:
            if file == RESULTS_FILENAME:
                with open(os.path.join(subdir, file)) as csv_file:
            
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    epoch_num = 0
                    lines_this_epoch = 0
                    epoch_len = FIRST_EPOCH_LEN[i]
                    epoch_results = np.zeros([epoch_len], dtype = float)
                
                    for row in csv_reader:
                
                        if line_count > 0:
                            epoch_results[lines_this_epoch] = row[len(row) - 1]
                            lines_this_epoch += 1

                        if lines_this_epoch >= epoch_len:
                            x_vals[SUB_DIRS[i]].append(epoch_num * TRAINING_SAMPLES_PER_EPOCH[i])
                            y_vals[SUB_DIRS[i]].append(np.mean(epoch_results))
                            epoch_num += 1
                            lines_this_epoch = 0
                            epoch_len = EPOCH_LEN[i]
                            epoch_results = np.zeros([epoch_len], dtype = float)
                        
                        line_count += 1

for i in range(0, len(SUB_DIRS)):                 
    df = pd.DataFrame({'Training samples': x_vals[SUB_DIRS[i]], SUB_DIRS[i]: y_vals[SUB_DIRS[i]]})
    graph = seaborn.lineplot(x='Training samples', y=SUB_DIRS[i], data=df, label=SCHED_NAMES[i])

graph.set(ylabel='Oracle score')

graph.axhline(y=baseline_score, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][len(SUB_DIRS)], linestyle='--', label='MCTS maximise goals')
graph.axhline(y=score_with_oracle, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][len(SUB_DIRS) + 1], linestyle='--', label='MCTS maximise oracle')

graph.set(xlim=(0, MAX_X))

graph.legend(loc='lower right')

plt.savefig(parsed_args.input_dir + '/plot.png')           

