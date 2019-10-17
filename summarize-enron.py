from collections import Counter
from pathlib import Path
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys


def main(): 
    """
    Main function is to create a three columns csv file sorted by 
    the the number of emails sent. Addtionally two images are created
    and saved in the image folder of this directory. 
    """
    print('Calculations are being created')
    filename = sys.argv[1]
    df = pd.read_csv(f'data/{filename}', low_memory=False,\
        names=["Time", "MessageId", "Sent", "Recipients", "Topic", "Mode"], usecols=["Time", "MessageId", "Sent", "Recipients"])

    cols = ['Sent', 'Recipients']

    #Pre-Processing Sent
    for i in df.columns: 
        df[i] = df[i].apply(lambda x:str(x).split("@")[0].split("/")[0].split("AT")[0].split(" - ")[0].lower())
    for i in df.columns: 
        df[i] = df[i].apply(lambda x:str(x).replace(".", " ").replace("_", " ").replace('"', "").replace(":", ""))

    df['Date'] = pd.to_datetime(df['Time'], unit='ms')
    df['DATE'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    drop = ['Time', 'Date'] 
    df.drop(drop, axis=1, inplace=True)
    df_plot = df.copy()

    #Number of emails sent by each user
    df_sent = pd.DataFrame(pd.value_counts(df.Sent)).reset_index().rename(columns={'index':'Name'})

    #Expand series into column recipients and do preprocessing
    df = df.Recipients.str.split("|",expand=True)

    #Concat all values of the expand and flatten and create a create collection.Counter
    cols = df.columns.values
    df['Names'] = df[cols].apply(lambda row: ",".join(row.values.astype(str)), axis=1)
    #Create DataFrame to get number of recieved emails by date 
    df_rec = df.copy()
    #Get Count of Recieved Total by Person
    df_split = df['Names'].values
    z = [sent.split(",") for sent in df_split]
    z = [y for t in z for y in t]
    counter_dic = dict(Counter(z))

    #Create a DataFrame to write out to
    df_final_sent = pd.DataFrame.from_dict(counter_dic, orient='index', columns=['Recieved']).reset_index()\
                                           .rename(columns={'index':'Name', 'Sent':'Sent', 'Recieved':'Recieved'})
    indexNames = df_final_sent[df_final_sent['Name'] == 'none'].index
    df_final_sent.drop(indexNames, inplace=True)

    #Create merged DataFrame
    df_merge = pd.merge(df_final_sent, df_sent, on='Name', how='outer', indicator=False)
    df_merge.sort_values("Sent", ascending=False, inplace=True)
    df_merge.reset_index(drop=True, inplace=True)
    df_merge = df_merge[['Name', 'Sent', 'Recieved']]
    print(f'Calculations are complete and have been stored')
    df_merge.to_csv(f"{os.path.join(os.getcwd(), filename.split('.')[0] + '_clean.csv')}", index=False)
    print('Top 5 Results')
    print(df_merge.head())

    #Create Top Five Names for below
    top_5_names = list(df_merge[:5].Name)

    #Concat all values 
    cols = df_rec.columns.values
    df_rec['Names'] = df_rec[cols].apply(lambda row: ",".join(row.values.astype(str)), axis=1)
    names = df.Names
    names = names.to_dict()
    names = {k:list(filter(lambda x: x != "None", v.split(','))) for k,v in names.items()}
    frequencies = {k:[] for k in names.keys()}

    #Count loop 
    for top5 in top_5_names: 
        out = [top5]
        for k,v in names.items():
            if top5 in v:
                frequencies[k].append(1)
            else:
                frequencies[k].append(0)

    name_in_list = pd.DataFrame(list(frequencies.items()), columns=['index', 'Count'])

    merge_plot = pd.DataFrame(name_in_list.Count.tolist(), columns= ['jeff', 'sara', 'pete', 'chris', 'notes'], index=name_in_list.index)

    plot_rec = pd.merge(df_plot, merge_plot, left_index=True, right_index=True)
    plot_rec = plot_rec.sort_values('DATE', ascending=True)

    # df_5 = df_plot[df_plot['Sent'].isin(top_5_names)]
    #plot sent another dry conspect needs to be cleaned up 
    df_1_rec = plot_rec.groupby(['jeff', 'DATE']).size().drop(0, level='jeff').reset_index(level='jeff', drop=True)
    df_1_rec = pd.DataFrame(df_1_rec, columns={'Count'}).sort_index()
    df_2_rec = plot_rec.groupby(['sara', 'DATE']).size().drop(0, level='sara').reset_index(level='sara', drop=True)
    df_2_rec = pd.DataFrame(df_2_rec, columns={'Count'}).sort_index()
    df_3_rec = plot_rec.groupby(['pete', 'DATE']).size().drop(0, level='pete').reset_index(level='pete', drop=True)
    df_3_rec = pd.DataFrame(df_3_rec, columns={'Count'}).sort_index()
    df_4_rec = plot_rec.groupby(['chris', 'DATE']).size().drop(0, level='chris').reset_index(level='chris', drop=True)
    df_4_rec = pd.DataFrame(df_4_rec, columns={'Count'}).sort_index()
    df_5_rec = plot_rec.groupby(['notes', 'DATE']).size().drop(0, level='notes').reset_index(level='notes', drop=True)
    df_5_rec = pd.DataFrame(df_5_rec, columns={'Count'}).sort_index()

    #Create the plot space upon which to plot the data
    fig, ax = plt.subplots(ncols=1, nrows=10, figsize=(15,50))
    fig.subplots_adjust(hspace=.5)

    #Add the x-axis and the y-axis to the plot
    ax[0].plot(df_1_rec.index.values, df_1_rec['Count'], '-o', color='blue')
    sns.distplot(df_1_rec['Count'], color = "b", label = "Employeed", bins = 40, ax=ax[1], kde = True)
    ax[2].plot(df_2_rec.index.values, df_2_rec['Count'], '-o', color='green')
    sns.distplot(df_2_rec['Count'], color = "g", label = "Employeed", bins = 40, ax=ax[3], kde = True)
    ax[4].plot(df_3_rec.index.values, df_3_rec['Count'], '-o', color='red')
    sns.distplot(df_2_rec['Count'], color = "r", label = "Employeed", bins = 40, ax=ax[5], kde = True)
    ax[6].plot(df_4_rec.index.values, df_4_rec['Count'], '-o', color='purple')
    sns.distplot(df_5_rec['Count'], color="purple", label = "Employeed", bins = 40, ax=ax[7], kde = True)
    ax[8].plot(df_5_rec.index.values, df_5_rec['Count'], '-o', color='orange')
    sns.distplot(df_5_rec['Count'], color = "orange", label = "Employeed", bins = 40, ax=ax[9], kde = True)

    #Set Title and Labels for axes
    ax[0].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Jeff Dasovich")
    ax[1].set(xlabel="Count of Emails (bin size = 40)",
              ylabel="Density Percentile of Daily Emails",
              title="Distribution of Emaild Recieved Daily for Jeff Dasovich")
    ax[2].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Sara Shackleton")
    ax[3].set(xlabel="Count of Emails (bin size = 40)",
              ylabel="Density Percentile of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Sara Shackleton")
    ax[4].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Pete Davis")
    ax[5].set(xlabel="Count of Emails (bin size = 40)",
              ylabel="Density Percentile of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Pete Davis")
    ax[6].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Chris Germany")
    ax[7].set(xlabel="Count of Emails (bin size = 40)",
              ylabel="Density Percentile of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Chris Germany")
    ax[8].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Notes")
    ax[9].set(xlabel="Count of Emails (bin size = 40)",
              ylabel="Density Percentile of Daily Emails",
              title="Enron Data Challenge Emails Recieved for Notes")
    #Clean up the x axis dates
    ax[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=15))
    ax[0].xaxis.set_major_formatter(DateFormatter("%m-%d-%Y"))
    ax[2].xaxis.set_major_locator(mdates.WeekdayLocator(interval=15))
    ax[2].xaxis.set_major_formatter(DateFormatter("%m-%d-%Y"))
    ax[4].xaxis.set_major_locator(mdates.WeekdayLocator(interval=15))
    ax[4].xaxis.set_major_formatter(DateFormatter("%m-%d-%Y"))
    ax[6].xaxis.set_major_locator(mdates.WeekdayLocator(interval=15))
    ax[6].xaxis.set_major_formatter(DateFormatter("%m-%d-%Y"))
    ax[8].xaxis.set_major_locator(mdates.WeekdayLocator(interval=15))
    ax[8].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    #Create folder and save recieved
    SUBM = Path(f'{os.getcwd()}/images/')
    recieved_image = 'recieved_image.pdf'
    os.makedirs(SUBM, exist_ok=True)
    plt.savefig(Path(f'{SUBM}/{recieved_image}'))
    print(f'Image has been saved in /images/{recieved_image}')
    ## Plot Sent 

    plot_sent = df_plot[df_plot['Sent'].isin(top_5_names)]
    plot_sent = plot_sent.groupby(['DATE', 'Sent']).size()
    plot_sent = pd.DataFrame(plot_sent, columns={'Count'}).reset_index(1)

    df_1=plot_sent[plot_sent.Sent == 'jeff dasovich']
    df_2=plot_sent[plot_sent.Sent == 'sara shackleton']
    df_3=plot_sent[plot_sent.Sent == 'pete davis']
    df_4=plot_sent[plot_sent.Sent == 'chris germany']
    df_5=plot_sent[plot_sent.Sent == 'notes'] 

    df_1=plot_sent[plot_sent.Sent == 'jeff dasovich']
    df_2=plot_sent[plot_sent.Sent == 'sara shackleton']
    df_3=plot_sent[plot_sent.Sent == 'pete davis']
    df_4=plot_sent[plot_sent.Sent == 'chris germany']
    df_5=plot_sent[plot_sent.Sent == 'notes'] 

    #Create the plot space upon which to plot the data
    fig, ax = plt.subplots(ncols=1, nrows=5, figsize=(15,30))
    fig.subplots_adjust(hspace=.5)

    #Add the x-axis and the y-axis to the plot
    ax[0].plot(df_1.index.values, df_1['Count'], '-o', color='blue')
    ax[1].plot(df_2.index.values, df_2['Count'], '-o', color='green')
    ax[2].plot(df_3.index.values, df_3['Count'], '-o', color='red')
    ax[3].plot(df_4.index.values, df_4['Count'], '-o', color='purple')
    ax[4].plot(df_5.index.values, df_5['Count'], '-o', color='orange')

    #Set Title and Labels for axes
    ax[0].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge\n Emails for Jeff Dasovich")
    ax[1].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge\n Emails for Sara Shackleton")
    ax[2].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge\n Emails for Pete Davis")
    ax[3].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge\n Emails for Chris Germany")
    ax[4].set(xlabel="Date",
              ylabel="Number of Daily Emails",
              title="Enron Data Challenge\n Emails for Notes")

    #Clean up the x axis dates
    ax[0].xaxis.set_major_locator(mdates.WeekdayLocator(interval=10))
    ax[0].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=15))
    ax[1].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    ax[2].xaxis.set_major_locator(mdates.WeekdayLocator(interval=10))
    ax[2].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    ax[3].xaxis.set_major_locator(mdates.WeekdayLocator(interval=15))
    ax[3].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    ax[4].xaxis.set_major_locator(mdates.WeekdayLocator(interval=10))
    ax[4].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    #Save Sent
    sent_emails = 'sent_image.pdf'
    fig.savefig(Path(f'{SUBM}/{sent_emails}'))
    print(f'Image has been saved in /images/{sent_emails}')

if __name__ == '__main__':
    main()