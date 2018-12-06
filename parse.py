import zipfile
import numpy as np
import pandas as pd
import sklearn as sk
import io

n_heroes = 82
n_heroes_per_game = 10
n_features = n_heroes_per_game + 1

def process_data(replays):
    # hero winrate table
    wins_by_hero = replays.groupby(['HeroID'])['Is Winner'].value_counts(normalize=True, sort=False).iloc[1::2].values
    assert len(wins_by_hero) == n_heroes

    def synth_features(replay):
        pass

    # generate feature vector for each replay
    games = replays.groupby(['ReplayID'])
    features = np.array([1 for i in range(n_features)], dtype=np.float64)
    for _, group in games:
        # we need to put the winner/loser in place randomly
        g_features = []
        group = group.copy()
        win_pos = True
        if np.random.randint(1, 3) % 2 == 0:
            win_pos = False
        group.sort_values('Is Winner', inplace=True, ascending=win_pos)
        winrate = []
        for hero in group['HeroID']:
            winrate.append(wins_by_hero[hero - 1])
        winrate.append(group['Is Winner'].iloc[0])
        features = np.vstack((features, winrate))
    features = features[1:]
    return features
    
if __name__ == '__main__':
    replays = None
    n = None #number of rows pandas will read
    if n is None:
        print('consuming all replays...', end='')
    else:
        print('consuming {} replays...'.format(n // 10), end='')
    with zipfile.ZipFile('data.zip') as z:
        with io.TextIOWrapper(z.open('ReplayCharacters.csv', 'r')) as rc:
            replays = pd.read_csv(rc, nrows=n, engine='c')
            print('done')
    print('synthesizing features...', end='')
    features = process_data(replays)
    print('done')
    np.savetxt('features.txt', features)
    print('{} features written to features.txt'.format(n_features))
