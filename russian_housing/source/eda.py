#!/usr/bin/env python3


"""Exploratory Data Analysis"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

processed_dir = '/home/edward/work/projects/kaggle/russian_housing/data/processed/'

# processed_dir = '../data/processed/'
df_train = pd.read_pickle(processed_dir + 'macro_train.pkl')



def distribution_report(df, var_list, file):
    """
    Creates a pdf report with a distribution plot for each
    variable specified in var_list.
    """
    page = PdfPages(file)
    for var in var_list:
        try:
            sns.distplot(df[var], rug = False)
            page.savefig()
        except ValueError:
            continue
    page.close()


plot_file = '/home/edward/work/projects/kaggle/russian_housing/output/figures/dist_plots.pdf'
vars_list = ["price_doc", "provision_doctors", "salary","full_sq"]
automatic_vars_list = list(df_train)[:10]
distribution_report(df_train, automatic_vars_list, plot_file)

fig,axarray = plt.subplots(1,1,figsize=[15,10],sharey="row",sharex="col") 

sns.distplot(df_train["full_sq"],rug = False)

sns.jointplot(x="rent_price_2room_bus", y="provision_doctors", 
	data=df_train)

# sns.boxplot(x="state", y="rent_price_2room_bus", data=df_train)

# sns.pointplot(x="state", y="rent_price_2room_bus", data=df_train)
            
# sns.factorplot(x="day", y="total_bill", hue="smoker", data=df_train)

plt.show()
