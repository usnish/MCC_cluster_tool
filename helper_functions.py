# import required libraries
import math
import pandas as pd
import scipy.stats as stats

# HELPER FUNCTIONS

def prop_error(n, p, alpha = 0.05):
    """
    Return the confidence interval for a proportion at the given alpha level
    """
    z_critical = stats.norm.ppf(1-alpha/2.0)

    returnstr = str(round(p, 2))
    margin = z_critical * math.sqrt(p*(1-p)/n)
    returnstr += ' (' + str(round(p - margin,2)) + '-' + str(round(p + margin,2)) + ')'
    
    return(returnstr)

def prop_error_pct(n, p, alpha = 0.05):
    """
    Return the confidence interval for a proportion at the given alpha level
    """
    z_critical = stats.norm.ppf(1-alpha/2.0)

    returnstr = str(round(p * 100, 1))
    margin = (z_critical * math.sqrt(p*(1-p)/n)) * 100
    returnstr += ' (' + str(round(p * 100 - margin,1)) + '-' + str(round(p * 100 + margin,1)) + ')'
    
    return(returnstr)

#def find_quintiles(df,cost_col):
   
# def cost_bins(df_col, n):
#     #pd.qcut(df_col, n).value_counts()
#     pass

def epi_rows(df, col, year = None):
    """
    Pass: dataframe and category column
    Return: temporary dataframe with broken down multimorbidity
        #things for each category
    """
    categories = df[col].unique().sort_values().tolist()
    
    temp = pd.DataFrame(index = categories)
    
    gt_1_col = []
    gt_2_col = []
    total_col = []
    
    # assemble dictionary of columns
    col_data = {}
    
    # set up
    cols = ['n', '>1 chronic condition', '>1 %', '>2 chronic condition', '>2 %', 'Total Spending ($)', 'Yearly Spending %']
    for c in cols:
        col_data[c] = []
        
    if year == None:
        for category in categories:
            print(category)
            gt_1 = df[(df[col] == category) & \
                      (df.no_comorbidities > 1)].shape[0]
            gt_2 = df[(df[col] == category) & \
                      (df.no_comorbidities > 2)].shape[0]
            total = df[(df[col] == category)].shape[0]

            cost = int(df[(df[col] == category)]['sum_costs'].sum())

            total_cost = df['sum_costs'].sum()

            col_data['n'].append(total)
            col_data['>1 chronic condition'].append(gt_1)
            col_data['>1 %'].append(round(gt_1/total * 100, 1))
            col_data['>2 chronic condition'].append(gt_2)
            col_data['>2 %'].append(round(gt_2/total * 100, 1))
            col_data['Total Spending ($)'].append(cost)
            col_data['Yearly Spending %'].append(round(cost / total_cost * 100,2))
            
    else:
        for category in categories:
            gt_1 = df[(df.year == year) & \
                      (df[col] == category) & \
                      (df.no_comorbidities > 1)].shape[0]
            gt_2 = df[(df.year == year) & \
                      (df[col] == category) & \
                      (df.no_comorbidities > 2)].shape[0]
            total = df[(df.year == year) & \
                       (df[col] == category)].shape[0]

            cost = int(df[(df.year == year) & \
                       (df[col] == category)]['sum_costs'].sum())

            total_cost = df[(df.year == year)]['sum_costs'].sum()

            col_data['n'].append(total)
            col_data['>1 chronic condition'].append(gt_1)
            col_data['>1 %'].append(round(gt_1/total * 100, 1))
            col_data['>2 chronic condition'].append(gt_2)
            col_data['>2 %'].append(round(gt_2/total * 100, 1))
            col_data['Total Spending ($)'].append(cost)
            col_data['Yearly Spending %'].append(round(cost / total_cost * 100,2))
    
    temp = temp.assign(**col_data) #FIXXXXX
    temp = temp[cols]
    
    return temp


def age_bin(age, labels, bins):
    """ Return a label for a given age and bin.
    
    Argument notes:
    age -- int
    labels -- list of strings
    bins -- list of tuples, with the first tuple value being 
            the inclusive lower limit, and the higher tuple 
            value being the exclusive upper limit
    """
    for x in range(len(bins)):
        if age < bins[x][1] and age >= bins[x][0]:
            return labels[x]
        
def comorbidity_indicator(data, cc):
    """Takes a list of comorbidities and returns a column indicator.
    
    Arguments:
    data
    cc
    """
    return data.classes.apply(lambda x: set(x).issuperset(set(cc)))

def cost_center_indicator(data, cc): 
    """Takes a list of cost center indices and returns a column indicator.
    Unique to the agg_indices column name
    Arguments:
    data
    cc
    """
    return data.agg_indices.apply(lambda x: set(x).issuperset(set(cc)))

def prevalence(data, indicator):
    """
    return prevalence in a given year's worth of data in df
    """
    return data[indicator].sum() / data.shape[0]
    

def age_adjusted_Frequency(data, indicator=None, comorbidities=None, weights = pd.Series([]), yr = None):
    """
    parameters --
        data : dataframe, rows are patient-years
        cc : list of chronic conditions in that given patient-year
        yr : year of prevalence determination
        aw : age weights (pandas Series with bins as index)
        
    description --
        Given a dataframe, compute the age-adjusted prevalence. 
        Need "age_bin" column
        
    returns --
        tuple containing unadjusted, then adjusted prevalence
    """
    if yr == None:
        temp = data
    else:
        temp = data[data.year == yr]
    
    # table to perform calculation
    aw_table = pd.DataFrame(index=weights.index)        
    #each bin's prevalence in a list to add as column
    binned_prevalences = [] 
    for x in weights.index.tolist(): # add prevalence for each age bin
        if comorbidities != None:
            binned_prevalences.append(temp[temp.age_bin == x].classes.apply(lambda x: set(x).issuperset(set(comorbidities))).sum() / temp[temp.age_bin == x].shape[0])
            
        else:
            binned_prevalences.append(prevalence(temp[temp.age_bin == x], indicator))
            
    # multiple weights in parallel
    aw_table['raw'] = binned_prevalences
    aw_table['weights'] = weights
    aw_table['adjusted'] = aw_table['raw'] * aw_table['weights']
        
    #return adjusted prevalence
    if comorbidities != None:
        return (temp.classes.apply(lambda x: set(x).issuperset(set(comorbidities))).sum() / temp.shape[0], aw_table['adjusted'].sum())
    
    else:
        return (prevalence(temp,indicator), aw_table['adjusted'].sum())

