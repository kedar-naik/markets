# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 23:37:54 2018

@author: Kedar
"""
import csv
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import gridspec
import webbrowser
import numpy as np
from scipy.special import gammaln
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import itertools

plt.ioff()

#-----------------------------------------------------------------------------#
class PriceHistory:
    '''
    class for reading in a ticker's price history and computing features for 
    machine learning
    '''
    # constructor, parses csv file and creates dictionary ---------------------
    def __init__(self, ticker_symbol, label):
        '''
        takes in a csv file name where the first line contains column names and 
        all subsequent lines are data points. returns a dictionary in which the 
        keys are the lowercase column names and the values are lists containing 
        the corresponding column's data.
        '''
        # record the ticker symbol and the label
        self.ticker = ticker_symbol
        self.label = label
        # parse the csv corresponding to this ticker symbol
        with open(ticker_symbol+'.csv') as csv_file:
            reader_object = csv.reader(csv_file)
            raw_data = [line for line in reader_object]
        # pull out the header as list
        header = raw_data.pop(0)
        # initialize the dictionary, where the keys are the lowercase column
        # names and the values are empty lists
        self.data = {}
        for column_name in header:
            self.data[column_name.lower()] = []
        # run through the rows of data extracted from the file
        for line in raw_data:
            # set the column index to zero
            column_index = 0
            # run through the entries in this row
            for entry in line:
                # extract the name of this column (i.e. the key)
                column_name = header[column_index].lower()
                # depending on the column, process accordingly
                if column_name == 'date':
                    # try processing the date in two different ways
                    try:
                        # try the YYYY-MM-DD format
                        self.data[column_name].append(datetime.strptime(entry, '%Y-%m-%d'))
                    except ValueError:
                        # try the MM/DD/YYYY format
                        self.data[column_name].append(datetime.strptime(entry, '%m/%d/%Y'))
                else:
                    # if this isn't the date column, try converting the value
                    # to a float and store it
                    try:
                        self.data[column_name].append(float(entry))
                    except ValueError:
                        # if a certain point can't be processed, then it is 
                        # probably 'null' (or something like that). replace the
                        # missing/bad value with the corresponding value from 
                        # the previous row (assuming there is a previous row)
                        if self.data[column_name]:
                            previous_entry = self.data[column_name][-1]
                            self.data[column_name].append(previous_entry)
                        else:
                            # if there is no previous value, i.e. the first row
                            # has invalid values, check the file for fishiness
                            raise IOError('\n\t Check the ' + ticker_symbol + \
                                  ' file -- something is wrong here...\n')                        
                # increment the column index
                column_index += 1
        # count the number of records
        self.days_recorded = len(self.data['date'])
    # compute features of the time history ------------------------------------
    def compute_features(self):
        '''
        takes the data dictionary created by the constructor and computes the
        quantities defined here. the new values are recorded as their own keys.
        new user-defined values can be computed here
        '''
        # find the daily percentage change, the first value is defined as zero
        self.data['daily percent change'] = [0]
        # run through the records, skipping day zero
        for day in range(1,self.days_recorded):
            todays_close = self.data['close'][day]
            yesterdays_close = self.data['close'][day-1]
            percent_change = 100.0*(todays_close-yesterdays_close)/yesterdays_close
            self.data['daily percent change'].append(percent_change)
    # fit a gaussian to the given features and place each point in a bin ------
    def assign_gaussian_bins(self, features_to_bin, bins_to_use):
        '''
        for each of the given feature names, compute the mean and standard
        deviation (std) of the feature time history. also compute the gaussian
        distribution associated with this mean-std pair. using this 
        distribution, create an even number of bins in which to place the data 
        based on the number provided by the user. for example, if the number of 
        bins to use for the given feature is 6, then the data points for the 
        feature will each be categorized based on which of the following six 
        ranges they fall into and by the following labels:
                            range                           label
            (1) point in (-infinity, mean-2*std]             -3
            (2) point in (mean-2*std, mean-std]              -2
            (3) point in (mean-std, mean]                    -1
            (4) point in (mean, mean+std]                     1
            (5) point in (mean+std, mean+2*std]               2
            (6) point in (mean+2*std, infinity]               3
        '''
        # check to make sure bin numbers have been given for each feature
        n_features_to_bin = len(features_to_bin)
        n_bins_to_use = len(bins_to_use)
        assert n_bins_to_use==n_features_to_bin, '\n\t '+str(n_bins_to_use) + \
               ' bin values given for '+str(n_features_to_bin) + ' features!\n'
        # check to make sure that the given number of bins is an even number
        for n_bins in bins_to_use:
            assert n_bins%2==0, '\n\t The desired number of gaussian bins ' + \
                                'needs to be an even number! (' + str(n_bins)+\
                                ' is odd!)\n'
        # run through the features
        for index in range(n_features_to_bin):
            # extract the feature name and the corresponding number of bins
            quantity_name = features_to_bin[index]
            n_bins = bins_to_use[index]
            # create the bin labels (-n_bins/2, ..., -1, 1, ..., n_bins/2)
            bin_labels = list(range(-int(n_bins/2.0), int(n_bins/2.0)+1))
            bin_labels.remove(0)
            # for this feature, compute the mean and standard deviation
            mu = np.mean(self.data[quantity_name])
            sigma = np.std(self.data[quantity_name])
            # create the bins as a list of small dictionaries
            bin_ranges = []
            # define the ranges for each bin
            for i in range(n_bins):
                bin_range = {}
                # set the lower bound of the range
                if i==0:
                    start = -np.inf
                else:
                    start = mu + (i-n_bins/2.0)*sigma
                # set the upper bound of the range
                if i==n_bins-1:
                    end = np.inf
                else:
                    end = mu + (i+1-n_bins/2.0)*sigma
                # create the range dictionary
                bin_range['start'] = start
                bin_range['end'] = end
                bin_range['label'] = bin_labels[i]
                # record the range tuple
                bin_ranges.append(bin_range)
            # run through the points in this feature, figure out the right bin
            # and assign the correct label in a newly created list
            new_key = quantity_name + ' (' + str(n_bins) + ' bins)'
            self.data[new_key] = {}
            self.data[new_key]['values'] = []
            for point in self.data[quantity_name]:
                for bin_range in bin_ranges:
                    if point > bin_range['start'] and point <= bin_range['end']:
                        self.data[new_key]['values'].append(bin_range['label'])
            # record the dictionary of bin ranges for this features
            self.data[new_key]['bins'] = bin_ranges
    # plot the time history of a given feature --------------------------------
    def plot_quantities(self, quantity_names):
        '''
        this subroutine plots the time history of the given feature.
        '''
        # create a list of plottable dates
        dates_to_plot = mdates.date2num(self.data['date'])
        for quantity_name in quantity_names:
            # preliminaries
            plot_name = self.label + ' (' + quantity_name + ')'
            auto_open = True
            the_fontsize = 14
            fig = plt.figure(plot_name)
            grid_specs = gridspec.GridSpec(1,2)
            # stretch the plotting window
            width, height = fig.get_size_inches()
            fig.set_size_inches(2.5*width, 1.0*height, forward=True)
            # plot the title
            fig.suptitle('$' + self.label.replace(' ','\;') + '\;\\left( ' + \
                         self.ticker.replace('^','^\\wedge ')+'\\right)'+'$', 
                         fontsize=the_fontsize+2)
            # plot the time history
            ax = fig.add_subplot(grid_specs[0])
            if ' bins)' in quantity_name:
                feature_time_history = self.data[quantity_name]['values']
                max_bin_number = max(feature_time_history)
                ax.set_yticks(list(range(-max_bin_number,max_bin_number+1)))
            else:
                feature_time_history = self.data[quantity_name]
            ax.plot_date(dates_to_plot, feature_time_history, 'k.-')
            # set the title and axis labels
            ax.set_xlabel('$t$', fontsize=the_fontsize)
            feature_latex = quantity_name.replace(' ', '\quad') 
            ax.set_ylabel('$' + feature_latex + '$', fontsize=the_fontsize)
            # rotate the dates
            for tick in ax.get_xticklabels():
                tick.set_rotation(55)
            # plot the histogram and overlay model
            ax = fig.add_subplot(grid_specs[1])
            n_bins_to_use = int(self.days_recorded/10.0)
            n, bins, patches = ax.hist(feature_time_history, n_bins_to_use, label='data')
            if ' bins)' in quantity_name:
                ax.set_xticks(list(range(-max_bin_number,max_bin_number+1)))
                ax.set_xlabel('$' + feature_latex + '\; (label)$', fontsize=the_fontsize)
            else:
                ax.set_xlabel('$' + feature_latex + '$', fontsize=the_fontsize)
            ax.set_ylabel('$frequency$', fontsize=the_fontsize)
            # create a new set of axes that shares the same x-axis
            ax2 = ax.twinx()
            # compute and plot gaussian (or, for trading volume, a possion) 
            # distribution based on the mean and standard deviation values
            mean =  np.mean(feature_time_history)
            x_distribution = np.linspace(min(feature_time_history), max(feature_time_history), 200)
            if quantity_name == 'volume':
                # compute the poisson distribution for trading volume, but use 
                # the alternate formula, which uses the natural logarithm of 
                # the gamma function, since the standard form can blown up
                # standard formula:
                # p_distribution = np.exp(-mean)*(mean**x_distribution)/factorial(x_distribution)
                # alternate formula:
                p_distribution = np.exp(x_distribution*np.log(mean)-mean-gammaln(x_distribution+1.0))
                legend_suffix = 'Poisson \; PMF'
                marker = '-.'
            else:
                # compute the gaussian distribution for this feature
                std = np.std(feature_time_history)
                variance = std**2
                p_distribution = np.exp(-(x_distribution-mean)**2/(2.0*variance))/np.sqrt(2.0*np.pi*variance)
                legend_suffix = 'Gaussian \; PDF'
                marker = '-'
            ax2.plot(x_distribution, p_distribution, 'r'+marker, label='$'+legend_suffix+'$')
            ax2.set_ylabel('$p \\left( x \\right)$', fontsize=the_fontsize)
            ax2.legend(loc='best')
            # set the tight_layout options to take into account the title
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            # save plot and close
            print('\n\t'+'saving final image...', end='')
            file_name = plot_name+'.png'
            plt.savefig(file_name, dpi=300)
            print('figure saved: '+plot_name)
            plt.close(plot_name)
            # open the saved image, if desired
            if auto_open:
                webbrowser.open(file_name)
#-----------------------------------------------------------------------------#
def create_dataset(tickers_to_process):
    '''
    given the list of ticker symbols and names to process, create a dictionary
    of price-history objects. for each object, automatically compute all 
    derived features, assign the desired gaussian bins, and check to make sure 
    the date ranges over which the price histories are defined are identical.
    '''
    # initialize the dataset dictionary
    dataset = {}
    ticker_counter = 0
    # run through the list of tuples
    for ticker_tuple in tickers_to_process:
        # extract the symbol and ticker name
        ticker_symbol = ticker_tuple[0]
        ticker_label = ticker_tuple[1]
        # create a new key for this ticker and create a price-history object
        dataset[ticker_label] = PriceHistory(ticker_symbol, ticker_label)
        dataset[ticker_label].compute_features()
        
        # [user input] define a set of features to place in gaussian bins. n.b. 
        # this section can be changed/expanded, if desired, to produce unique
        # sets of bins for specific features
        features_to_bin = ['volume', 'daily percent change', 'daily percent change', 'daily percent change']
        # [user input] specify how many bins to use for each of those features
        bins_to_use = [4, 6, 2, 4]
        
        # compute the gaussian bins and assign the correct labels to each point
        dataset[ticker_label].assign_gaussian_bins(features_to_bin, bins_to_use)        
        # record the dates over which this ticker's history is available
        if ticker_counter==0:
            previous_ticker_dates = dataset[ticker_label].data['date']
            previous_ticker_label = ticker_label
        else:
            current_ticker_dates = dataset[ticker_label].data['date']
            # check if this date range is the same as the previous one
            assert current_ticker_dates == previous_ticker_dates, \
                   '\n\t Uh oh! The dates at which ' + ticker_label+' has ' + \
                   'been recorded are not the same as the dates at which ' + \
                   previous_ticker_label + ' has been recorded at! Please ' + \
                   'check!\n'
            # reassign the current ticker dates and label as the previous ones
            previous_ticker_dates = current_ticker_dates
            previous_ticker_label = ticker_label
    # return the dataset dictionary
    return dataset
#-----------------------------------------------------------------------------#
def collate_and_plot_features(features_to_consider, make_plots=True):
    '''
    given a list of tuples of features to consider, plot the feature time
    histories and flatten the list
    '''
    # flatten the list of tuples of features to consider
    features_list = []
    for feature_tuple in features_to_consider:
        ticker_label = feature_tuple[0]
        feature_list = feature_tuple[1]
        # make plots for these features and this ticker, if desired
        if make_plots:
            dataset[ticker_label].plot_quantities(feature_list)
        # flatten the user-defined list
        for feature_name in feature_list:
            features_list.append((ticker_label, feature_name))
    # return the flattened list
    return features_list
#-----------------------------------------------------------------------------#
def print_data_summary(dataset):
    '''
    print a summary of the loaded ticker symbols and derived features
    '''
    # print a short heading
    print('\n\n  SUMMARY OF DATASET')
    # print the ticker symbols and their common names
    n_tickers = len(dataset.keys())
    print('\n\n\t- tickers loaded:\n')
    for ticker_name in dataset.keys():
        ticker_symbol = dataset[ticker_name].ticker
        print('\t - '+(ticker_symbol+':').ljust(8)+ticker_name)
    print('\n\t\t- number of tickers loaded: %d' % n_tickers)
    # print information about the time histories loaded
    first_ticker = list(dataset.keys())[0]
    start_date = min(dataset[first_ticker].data['date'])
    end_date = max(dataset[first_ticker].data['date'])
    print('\n\n\t- information about ticker time histories:\n')
    print('\t  - starting date:\t\t' + start_date.strftime('%m/%d/%Y'))
    print('\t  - ending date:\t\t' + end_date.strftime('%m/%d/%Y'))
    time_spanned = end_date - start_date    
    years_spanned = (time_spanned/365).days
    days_spanned = time_spanned.days%365
    print('\t  - time spanned:\t\t', end='')
    if years_spanned:
        print('%d years' % years_spanned, end='')
        if days_spanned:
            print(', ', end='')
    if days_spanned:
        print('%d days' % days_spanned)
    print('\t  - number of dates recorded:\t%d' % dataset[first_ticker].days_recorded)
    # print to the screen the features that are available for each ticker symbol
    features_available = list(dataset[first_ticker].data.keys())
    n_features_ticker = len(features_available)
    print('\n\n\t- available features for each ticker:\n')
    for feature in features_available:
        print('\t  - ' + feature)
    print('\n\t\t- number of features available per ticker: %d' %n_features_ticker)
    print('\n\t\t- total number of features available:\t   %d' 
          % (n_features_ticker*n_tickers))
    # print the features that will be used to build the machine-learning model
    print('\n\n\t- features that will be used for machine learning:\n')
    for feature_tuple in enumerate(features_list):
        feature_number = feature_tuple[0]+1
        ticker_name = feature_tuple[1][0]
        quantity_considered = feature_tuple[1][1]
        print('\t  - feature #%2d: ' % feature_number + quantity_considered + \
              ' of ' + ticker_name)
    print('\n\t\t- total number of features considered: %d' % feature_number)
#-----------------------------------------------------------------------------#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# [user input] specify which ticker symbols to process and their names
tickers_to_process = [('^GSPC',     'S&P 500'),
                      ('^RUT',      'Russell 2000'),
                      ('^TNX',      '10-year yield'),
                      ('GLD',       'Gold ETF'),
                      ('USO',       'Oil ETF'),
                      ('UUP',       'Dollar Index ETF'),
                      
                      ('HD',        'Home Depot'),
                      ('KO',        'Coke'),
                      ('XOM',       'Exxon')]

# [user input] select features to consider in the machine-learning model
features_to_consider = [('S&P 500',             ['daily percent change',
                                                 'volume'
                                                ]),
                        ('Russell 2000',        ['daily percent change',
                                                 'volume'
                                                ]),
                        ('10-year yield',       ['daily percent change'
                                                ]),
                        ('Gold ETF',            ['daily percent change',
                                                 'volume'
                                                ]),
                        ('Oil ETF',             ['daily percent change',
                                                 'volume'
                                                ]),
                        ('Dollar Index ETF',    ['daily percent change',
                                                 'volume'
                                                ])
                       ]

features_to_consider = [('S&P 500',             ['daily percent change (6 bins)'
                                                ]),
                        ('Russell 2000',        ['daily percent change (6 bins)'
                                                ]),
                        ('10-year yield',       ['daily percent change (6 bins)'
                                                ]),
                        ('Gold ETF',            ['daily percent change (6 bins)'
                                                ]),
                        ('Oil ETF',             ['daily percent change (6 bins)'
                                                ]),
                        ('Dollar Index ETF',    ['daily percent change (6 bins)'
                                                ])
                       ]
'''                    
features_to_consider = [('S&P 500',             ['daily percent change'
                                                ]),
                        ('Russell 2000',        ['daily percent change'
                                                ]),
                        ('10-year yield',       ['daily percent change'
                                                ]),
                        ('Gold ETF',            ['daily percent change'
                                                ]),
                        ('Oil ETF',             ['daily percent change'
                                                ]),
                        ('Dollar Index ETF',    ['daily percent change'
                                                ])
                       ]

features_to_consider = [('S&P 500',             ['volume',
                                                ]),
                        ('Russell 2000',        ['daily percent change',
                                                 'volume'
                                                ]),
                        ('10-year yield',       ['daily percent change'
                                                ]),
                        ('Gold ETF',            ['volume'
                                                ]),
                        ('Oil ETF',             ['volume'
                                                ]),
                        ('Dollar Index ETF',    ['volume'
                                                ])
                       ]
'''
           
# [user input] select the quantity to predict
quantity_to_predict = ('Coke', 'daily percent change (6 bins)')
quantity_to_predict = ('Home Depot', 'daily percent change (2 bins)')
#quantity_to_predict = ('Exxon', 'daily percent change (6 bins)')
#quantity_to_predict = ('Russell 2000', 'daily percent change (4 bins)')

class_names = list(np.array(list(range(int(quantity_to_predict[1].split()[-2][-1])+1)))-int(int(quantity_to_predict[1].split()[-2][-1])/2))
class_names.remove(0)

# create a dataset consisting of these tickers
dataset = create_dataset(tickers_to_process)

# plot the features of interest and flatten the given list of tuples
features_list = collate_and_plot_features(features_to_consider, make_plots=False)

# print a summary of the dataset to the screen
print_data_summary(dataset)


# create X and y for predicting tomorrow's closing gaussian bin based on 
# today's daily percentage changes
first_ticker_name = features_to_consider[0][0]
# the number of samples here is two less than the total. why? because the first
# date has an undefined change from the previous date and the last date's 
# values are not useful, since there is no value to predict on the day after
#   known: change from day 1 to day 2 (recorded on day 2)     
#   predict: change on day 3 (recorded on day 3)
n_samples = dataset[first_ticker_name].days_recorded - 2
n_features = len(features_list)
# initialize an X matrix and a y vector of the right shapes, filled with nans
X = np.nan*np.zeros((n_samples,n_features))
y = np.nan*np.zeros((n_samples,1))
# fill in the X matrix, column-by-column
feature_counter = 0
for feature_tuple in features_list:
    ticker_key = feature_tuple[0]
    ticker_feature = feature_tuple[1]
    if ' bins)' in ticker_feature:
        # if the feature is in gaussian bins, go one level deeper
        X[:,feature_counter] = dataset[ticker_key].data[ticker_feature]['values'][1:-1]
    else:
        X[:,feature_counter] = dataset[ticker_key].data[ticker_feature][1:-1]
    feature_counter += 1
# fill in the y vector
y_key = quantity_to_predict[0]
y_feature = quantity_to_predict[1]
if ' bins)' in y_feature:
    # if the feature is in gaussian bins, go one level deeper
    y[:,0] = dataset[y_key].data[y_feature]['values'][2:]
else:
    y[:,0] = dataset[y_key].data[y_feature][2:]

# compute the mean for each feature
feature_means = np.mean(X, axis=0)

# write the means to a pickle file
with open('feature_means.pkl', 'wb') as pickle_file:
    pickle.dump(feature_means, pickle_file)

# center the data
X = X - feature_means

# compute the standard deviation for each feature
feature_stds = np.std(X, axis=0)

# write the standard deviations to a pickle file
with open('feature_stds.pkl', 'wb') as pickle_file:
    pickle.dump(feature_stds, pickle_file)
    
# normalize the data
X = X/feature_stds

# split the data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# make a t-SNE plot of the training set
X_embedded = TSNE(n_components=2).fit_transform(X)


# instantiate the model
classifier = svm.SVC(kernel='rbf')
classifier = tree.DecisionTreeClassifier()
#classifier = RandomForestClassifier(n_estimators=10, max_depth=9)
#classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60,160,160,140,80))

# train the model
classifier.fit(X_train, y_train)

# evaluate accuracy on the test set
y_pred = classifier.predict(X_test)

accuracy = classifier.score(X_test, y_test)
print('\n\taccuracy =', accuracy)


# print confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# pickle the model

# accept new feature vector

# process new feature vector

# predict new value
    
    



