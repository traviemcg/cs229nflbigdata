import csv
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def preprocess(input_df, init_tables=None):
    """ 
    Function to preprocess dataframe
    Returns preprocessed dataframes, and tables used to process data frame as tuple
    When training, do not provide init_tables.
    When validating, provide init_tables from training.
    """
    
    df = input_df.copy()
    if init_tables is None:
        tables = {}
    else:
        tables = init_tables

    # Downsample, keep entries where player is rusher 
    #df = df[df['NflIdRusher']==df['NflId']]
    #df.reset_index(drop=True, inplace=True)

    # Handle 50 YardLine by filling in the null 'FieldPositions' with the value in 'PossessionTeam'
    df.FieldPosition.fillna(df.PossessionTeam, inplace=True)

    # Clean defenders in box - fill nan with median (i.e. 7), and bump 1 or 2 (few samples) up to 3
    df.DefendersInTheBox.fillna(7, inplace=True)
    df.DefendersInTheBox.replace(to_replace=[1, 2], value=3, inplace=True)

    # Group rare position values - change 'CB', 'DE', 'DT' (few samples) to 'Other'
    df.Position.replace(to_replace=['CB', 'DE', 'DT'], value='Other', inplace=True)

    # Fix inconsistent naming
    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'} 
    for abb in df['PossessionTeam'].unique():
        map_abbr[abb] = abb
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)

    # Find which team is on defense
    df['DefenseTeam'] = np.where(df['HomeTeamAbbr'] == df['PossessionTeam'], df['VisitorTeamAbbr'], df['HomeTeamAbbr'])

    # Calculate team's average offensive yards
    if init_tables is None:
        yards_avg_offense = df[['PossessionTeam','Yards']].groupby(['PossessionTeam']).mean()
        yards_avg_offense = yards_avg_offense.rename(columns={"Yards": "YardsAvgOffense"}).reset_index()
        tables['yards_avg_offense'] = yards_avg_offense
    else:
        yards_avg_offense = tables['yards_avg_offense']
    df = pd.merge(df, yards_avg_offense, how='left', on='PossessionTeam')
    df.YardsAvgOffense.fillna(4, inplace=True)

    # Calculate team's average yards allowed
    if init_tables is None:
        yards_avg_defense = df[['DefenseTeam','Yards']].groupby(['DefenseTeam']).mean()
        yards_avg_defense = yards_avg_defense.rename(columns={"Yards": "YardsAvgDefense"}).reset_index()
        tables['yards_avg_defense'] = yards_avg_defense
    else:
        yards_avg_defense = tables['yards_avg_defense']
    df = pd.merge(df, yards_avg_defense, how='left', on='DefenseTeam')
    df.YardsAvgDefense.fillna(4, inplace=True)

    # Calculate yards remaining to touchdown
    df['YardsRemaining'] = 100 - df.YardLine[df.FieldPosition == df.PossessionTeam]
    df.YardsRemaining.fillna(df.YardLine, inplace=True)

    # Calculate rusher carries
    if init_tables is None:
        carries = df[['PlayId', 'NflIdRusher', 'DisplayName']].groupby(['DisplayName', 'NflIdRusher']).agg('count').reset_index()
        carries.rename(columns={'PlayId':'Carries'}, inplace=True)
        tables['carries'] = carries
    else:
        carries = tables['carries']
    df = df.merge(carries[['NflIdRusher', 'Carries']], how='left', on='NflIdRusher')
    df.Carries.fillna(0, inplace=True)

    # Calculate rusher mean, max, min yards
    if init_tables is None:
        player_yards = df[['Yards', 'NflIdRusher', 'DisplayName']].groupby(['DisplayName', 'NflIdRusher']).agg(['mean', 'max', 'min'])['Yards'].reset_index()
        player_yards.rename(columns={'mean':'RusherMeanYards', 'max':'RusherMaxYards', 'min':'RusherMinYards'}, inplace=True)
        tables['player_yards'] = player_yards
    else:
        player_yards = tables['player_yards']
    df = df.merge(player_yards[['NflIdRusher', 'RusherMeanYards', 'RusherMaxYards', 'RusherMinYards']], how='left', on='NflIdRusher')
    df.RusherMeanYards.fillna(4, inplace=True)
    df.RusherMaxYards.fillna(99, inplace=True)
    df.RusherMinYards.fillna(-15, inplace=True)

    return (df, tables)

# Load and process data
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')
train = train[train['NflIdRusher']==train['NflId']]
train.reset_index(drop=True, inplace=True)
train, tables = preprocess(train)

# Fit model
features = ['Distance','Down','DefendersInTheBox', 'A']
le = preprocessing.LabelEncoder()
le.fit(train['Yards'])
clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0, min_samples_split=20, min_samples_leaf=10)
clf.fit(train[features], le.transform(train['Yards']))

def squash(yard_cdf, yard_remaining):
    squash_cdf = yard_cdf.copy()
    squash_cdf[199-(100 - yard_remaining):] = 1
    squash_cdf[0:yard_remaining-1] = 0 
    return squash_cdf

def predsToPMF(preds):
    pmf = np.zeros((1,199))
    for i in range(0, preds.shape[1]):
        pmf[0,int(le.inverse_transform([i]))+99] = preds[0,i]
    return pmf

from kaggle.competitions import nflrush
env = nflrush.make_env()
iter_test = env.iter_test()

for (test, sample_test_pred) in iter_test:
    # Load and preprocess test sample
    test = test[test['NflIdRusher']==test['NflId']]
    test.reset_index(drop=True, inplace=True)
    test, _ = preprocess(test, tables)
    
    # Use classifier to make predictions
    pred = clf.predict_proba(test[features])
    pred_pmf = predsToPMF(pred)
    pred_cdf = np.cumsum(pred_pmf, axis=1)
    try:
        squashed_pred_cdf = squash(pred_cdf, int(test['YardsRemaining']))
    except: # sometimes YardRemaining is missing, handle that case ...
        squashed_pred_cdf = pred_cdf

    # Write predictions to output df
    preds_df = pd.DataFrame(data=squashed_pred_cdf, columns=sample_test_pred.columns)
    env.predict(preds_df)

env.write_submission_file()