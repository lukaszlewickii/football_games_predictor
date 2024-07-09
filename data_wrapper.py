import pandas as pd
import numpy as np
import argparse

class FootballPredictorDataWrapper:
    def __init__(self, data):
        self.data = data
    
    def clean_data(self, df):
        #transforming date variable
        df['date_GMT'] = pd.to_datetime(df['date_GMT'])
        df['date'] = pd.to_datetime(df['date_GMT'].dt.date)
        df['time'] = df['date_GMT'].dt.time
        
        #adding aggregated variables
        df['corners_total'] = df['home_team_corner_count'] + df['away_team_corner_count']
        df['yellow_cards_total'] = df['home_team_yellow_cards'] + df['away_team_yellow_cards']
        df['red_cards_total'] = df['home_team_red_cards'] + df['away_team_red_cards']
        df['cards_total'] = df['yellow_cards_total'] + df['red_cards_total']
        df['shots_total'] = df['home_team_shots'] + df['away_team_shots']
        df['shots_on_target_total'] = df['home_team_shots_on_target'] + df['away_team_shots_on_target']
        df['shots_off_target_total'] = df['home_team_shots_off_target'] + df['away_team_shots_off_target']
        df['fouls_total'] = df['home_team_fouls'] + df['away_team_fouls']
        
        #getting stadium names without city in brackets
        # self.data['base_name'] = self.data['stadium_name'].str.replace(r" \(.*\)$", "", regex=True)

        # city_map = self.data[self.data['stadium_name'].str.contains(r"\(.*\)")].copy()
        # city_map['city'] = city_map['stadium_name'].str.extract(r"\((.*?)\)")[0]
        # city_map = city_map.groupby('base_name')['city'].agg(pd.Series.mode).to_dict()

        # #filling stadium names without city in brackets
        # self.data['normalized_stadium'] = self.data.apply(lambda row: f"{row['base_name']} ({city_map.get(row['base_name'], 'Unknown')})" if '(' not in row['stadium_name'] else row['stadium_name'], axis=1)

        #dropping unnecessary features
        df.drop(['timestamp', 'status', 'home_team_goal_timings', 'away_team_goal_timings', 'date_GMT'], axis=1, inplace=True)
        
        #setting one of the target variable - result of the game
        df['result'] = np.where(df['home_team_goal_count'] == df['away_team_goal_count'], 0, np.where(df['home_team_goal_count'] > df['away_team_goal_count'], 1, 2))
        
        def assign_season(date):
            year = date.year
            if date.month >= 8:  #season starts in august
                return f'{str(year)[2:]}/{str(year+1)[2:]}'
            else:
                return f'{str(year-1)[2:]}/{str(year)[2:]}'
        
        df['season'] = df['date'].apply(assign_season)
        
        return df
    
    def add_cumulative_goals_scored_before_game(self, df):

        #creating working dataframe with goals scored as well as a home team as an away team
        goals = pd.concat([
            df[['date', 'home_team_name', 'home_team_goal_count']].rename(columns={'home_team_name': 'team', 'home_team_goal_count': 'goals'}),
            df[['date', 'away_team_name', 'away_team_goal_count']].rename(columns={'away_team_name': 'team', 'away_team_goal_count': 'goals'})
        ])

        #sorting data by date
        goals.sort_values('date', inplace=True)

        #calculating cumulative goals for every team
        goals['cumulative_goals'] = goals.groupby('team')['goals'].cumsum()

        #substracting cumulative goals by goals scored in specific game to count goals only before game
        goals['cumulative_goals'] -= goals['goals']

        #merging back to original datafame
        df = df.merge(goals[['date', 'team', 'cumulative_goals']], left_on=['date', 'home_team_name'], 
                      right_on=['date', 'team'], how='left').rename(columns={'cumulative_goals': 'home_team_cumulative_goals_scored_pre_game'}).drop('team', axis=1)
        
        df = df.merge(goals[['date', 'team', 'cumulative_goals']], left_on=['date', 'away_team_name'], 
                      right_on=['date', 'team'], how='left').rename(columns={'cumulative_goals': 'away_team_cumulative_goals_scored_pre_game'}).drop('team', axis=1)
        
        return df
        
    def add_cumulative_goals_conceded_before_game(self, df):
        
        #creating working dataframe with goals conceded as well as a home team as an away team
        goals_conceded = pd.concat([
            df[['date', 'home_team_name', 'away_team_goal_count']].rename(columns={'home_team_name': 'team', 'away_team_goal_count': 'goals_conceded'}),
            df[['date', 'away_team_name', 'home_team_goal_count']].rename(columns={'away_team_name': 'team', 'home_team_goal_count': 'goals_conceded'})
        ])

        #sorting data by date
        goals_conceded.sort_values('date', inplace=True)

        #calculating cumulative goals conceded for every team
        goals_conceded['cumulative_goals_conceded'] = goals_conceded.groupby('team')['goals_conceded'].cumsum()

        #substracting cumulative goals by goals conceded in specific game to count goals only before game
        goals_conceded['cumulative_goals_conceded'] -= goals_conceded['goals_conceded']

        #merging back to original dataframe
        df = df.merge(goals_conceded[['date', 'team', 'cumulative_goals_conceded']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_goals_conceded': 'home_team_cumulative_goals_conceded_pre_game'}).drop('team', axis=1)
        df = df.merge(goals_conceded[['date', 'team', 'cumulative_goals_conceded']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_goals_conceded': 'away_team_cumulative_goals_conceded_pre_game'}).drop('team', axis=1)
        
        return df
        
    def add_average_goals_scored_and_conceded_per_game_before_game(self, df):

        #adding working features with goals scored by hosts and guests
        df['home_goals_scored'] = df['home_team_goal_count']
        df['away_goals_scored'] = df['away_team_goal_count']

        #adding working features with goals conceded by hosts and guests
        df['home_goals_conceded'] = df['away_team_goal_count']
        df['away_goals_conceded'] = df['home_team_goal_count']
        
        #creating working dataframe with goals scored and conceded as well as a home team as an away team
        goals_data = pd.concat([
            df[['date', 'home_team_name', 'home_goals_scored', 'home_goals_conceded']].rename(columns={'home_team_name': 'team', 'home_goals_scored': 'goals_scored', 'home_goals_conceded': 'goals_conceded'}),
            df[['date', 'away_team_name', 'away_goals_scored', 'away_goals_conceded']].rename(columns={'away_team_name': 'team', 'away_goals_scored': 'goals_scored', 'away_goals_conceded': 'goals_conceded'})
        ])

        #sorting data by date
        goals_data.sort_values('date', inplace=True)
        
        #calculating cumulative goals scored and conceded
        goals_data['cumulative_goals_scored'] = goals_data.groupby('team')['goals_scored'].cumsum()
        goals_data['cumulative_goals_conceded'] = goals_data.groupby('team')['goals_conceded'].cumsum()

        #counting number of games played
        goals_data['games_played'] = goals_data.groupby('team').cumcount() + 1

        #calculating average goals scored and conceded per game
        goals_data['average_goals_scored_per_game'] = goals_data['cumulative_goals_scored'] / goals_data['games_played']
        goals_data['average_goals_conceded_per_game'] = goals_data['cumulative_goals_conceded'] / goals_data['games_played']

        #removing last game
        goals_data['average_goals_scored_per_game_pre_game'] = goals_data.groupby('team')['average_goals_scored_per_game'].shift().fillna(0)
        goals_data['average_goals_conceded_per_game_pre_game'] = goals_data.groupby('team')['average_goals_conceded_per_game'].shift().fillna(0)
        
        #merging back to original dataframe
        df = df.merge(goals_data[['date', 'team', 'average_goals_scored_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_goals_scored_per_game_pre_game': 'home_team_average_goals_scored_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(goals_data[['date', 'team', 'average_goals_scored_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_goals_scored_per_game_pre_game': 'away_team_average_goals_scored_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(goals_data[['date', 'team', 'average_goals_conceded_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_goals_conceded_per_game_pre_game': 'home_team_average_goals_conceded_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(goals_data[['date', 'team', 'average_goals_conceded_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_goals_conceded_per_game_pre_game': 'away_team_average_goals_conceded_per_game_pre_game'}).drop('team', axis=1)
        
        return df
        
    def add_average_goals_scored_in_first_and_second_half_per_game_before_game(self, df):
        
        #creating working features
        df['home_team_second_half_goals'] = df['home_team_goal_count'] - df['home_team_goal_count_half_time']
        df['away_team_second_half_goals'] = df['away_team_goal_count'] - df['away_team_goal_count_half_time']
        
        #creating working dataframe with goals scored per half
        goals_data_half = pd.concat([
            df[['date', 'home_team_name', 'home_team_goal_count_half_time', 'home_team_second_half_goals']].rename(columns={'home_team_name': 'team', 'home_team_goal_count_half_time': 'first_half_goals', 'home_team_second_half_goals': 'second_half_goals'}),
            df[['date', 'away_team_name', 'away_team_goal_count_half_time', 'away_team_second_half_goals']].rename(columns={'away_team_name': 'team', 'away_team_goal_count_half_time': 'first_half_goals', 'away_team_second_half_goals': 'second_half_goals'})
        ])

        #sorting data by date
        goals_data_half.sort_values('date', inplace=True)
        
        #calculating cumulative goals scored in first and second half
        goals_data_half['cumulative_first_half_goals'] = goals_data_half.groupby('team')['first_half_goals'].cumsum()
        goals_data_half['cumulative_second_half_goals'] = goals_data_half.groupby('team')['second_half_goals'].cumsum()

        #counting number of games
        goals_data_half['games_played'] = goals_data_half.groupby('team').cumcount() + 1

        #calculating average goals per half
        goals_data_half['average_first_half_goals_per_game'] = goals_data_half['cumulative_first_half_goals'] / goals_data_half['games_played']
        goals_data_half['average_second_half_goals_per_game'] = goals_data_half['cumulative_second_half_goals'] / goals_data_half['games_played']

        #removing last game
        goals_data_half['average_first_half_goals_per_game_pre_game'] = goals_data_half.groupby('team')['average_first_half_goals_per_game'].shift().fillna(0)
        goals_data_half['average_second_half_goals_per_game_pre_game'] = goals_data_half.groupby('team')['average_second_half_goals_per_game'].shift().fillna(0)

        #merging back to original dataframe
        df = df.merge(goals_data_half[['date', 'team', 'average_first_half_goals_per_game_pre_game', 'average_second_half_goals_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_first_half_goals_per_game_pre_game': 'home_team_average_first_half_goals_scored_pre_game', 'average_second_half_goals_per_game_pre_game': 'home_team_average_second_half_goals_scored_pre_game'}).drop('team', axis=1)
        df = df.merge(goals_data_half[['date', 'team', 'average_first_half_goals_per_game_pre_game', 'average_second_half_goals_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_first_half_goals_per_game_pre_game': 'away_team_average_first_half_goals_scored_pre_game', 'average_second_half_goals_per_game_pre_game': 'away_team_average_second_half_goals_scored_pre_game'}).drop('team', axis=1)
        
        return df
        
    def add_average_goals_conceded_in_first_and_second_half_per_game_before_game(self, df):    

        #creating working features
        df['home_goals_conceded_first_half'] = df['away_team_goal_count_half_time']
        df['away_goals_conceded_first_half'] = df['home_team_goal_count_half_time']
        df['home_goals_conceded_second_half'] = df['away_goals_conceded'] - df['away_goals_conceded_first_half']
        df['away_goals_conceded_second_half'] = df['home_goals_conceded'] - df['home_goals_conceded_first_half']
        
        #creating working dataframe with goals conceded
        conceded_goals_data = pd.concat([
            df[['date', 'home_team_name', 'home_goals_conceded_first_half', 'home_goals_conceded_second_half']].rename(columns={'home_team_name': 'team', 'home_goals_conceded_first_half': 'first_half_conceded', 'home_goals_conceded_second_half': 'second_half_conceded'}),
            df[['date', 'away_team_name', 'away_goals_conceded_first_half', 'away_goals_conceded_second_half']].rename(columns={'away_team_name': 'team', 'away_goals_conceded_first_half': 'first_half_conceded', 'away_goals_conceded_second_half': 'second_half_conceded'})
        ])

        #sorting data by date
        conceded_goals_data.sort_values('date', inplace=True)
        
        #calculating cumulative goals conceded
        conceded_goals_data['cumulative_first_half_conceded'] = conceded_goals_data.groupby('team')['first_half_conceded'].cumsum()
        conceded_goals_data['cumulative_second_half_conceded'] = conceded_goals_data.groupby('team')['second_half_conceded'].cumsum()

        #counting number of games played
        conceded_goals_data['games_played'] = conceded_goals_data.groupby('team').cumcount() + 1

        #calculating average goals scored in both halfs
        conceded_goals_data['average_first_half_conceded_per_game'] = conceded_goals_data['cumulative_first_half_conceded'] / conceded_goals_data['games_played']
        conceded_goals_data['average_second_half_conceded_per_game'] = conceded_goals_data['cumulative_second_half_conceded'] / conceded_goals_data['games_played']

        #removing last game
        conceded_goals_data['average_first_half_conceded_per_game_pre_game'] = conceded_goals_data.groupby('team')['average_first_half_conceded_per_game'].shift().fillna(0)
        conceded_goals_data['average_second_half_conceded_per_game_pre_game'] = conceded_goals_data.groupby('team')['average_second_half_conceded_per_game'].shift().fillna(0)
    
        #merging back to original dataframe
        df = df.merge(conceded_goals_data[['date', 'team', 'average_first_half_conceded_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_first_half_conceded_per_game_pre_game': 'home_team_average_first_half_goals_conceded_pre_game'}).drop('team', axis=1)
        df = df.merge(conceded_goals_data[['date', 'team', 'average_second_half_conceded_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_second_half_conceded_per_game_pre_game': 'home_team_average_second_half_goals_conceded_pre_game'}).drop('team', axis=1)
        df = df.merge(conceded_goals_data[['date', 'team', 'average_first_half_conceded_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_first_half_conceded_per_game_pre_game': 'away_team_average_first_half_goals_conceded_pre_game'}). drop('team', axis=1)
        df = df.merge(conceded_goals_data[['date', 'team', 'average_second_half_conceded_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_second_half_conceded_per_game_pre_game': 'away_team_average_second_half_goals_conceded_pre_game'}).drop('team', axis=1)
    
        return df
    
    def add_average_goals_total_in_first_and_second_half_per_game_before_game(self, df):
        
        #creating working features
        df['home_first_half_goals'] = df['home_team_goal_count_half_time']
        df['home_second_half_goals'] = df['home_team_goal_count'] - df['home_team_goal_count_half_time']
        df['away_first_half_goals'] = df['away_team_goal_count_half_time']
        df['away_second_half_goals'] = df['away_team_goal_count'] - df['away_team_goal_count_half_time']
        
        #creating working dataframe with total goals in both halfs
        goals_data_half = pd.concat([
            df[['date', 'home_team_name', 'home_first_half_goals', 'home_second_half_goals']].rename(columns={'home_team_name': 'team', 'home_first_half_goals': 'first_half_goals', 'home_second_half_goals': 'second_half_goals'}),
            df[['date', 'away_team_name', 'away_first_half_goals', 'away_second_half_goals']].rename(columns={'away_team_name': 'team', 'away_first_half_goals': 'first_half_goals', 'away_second_half_goals': 'second_half_goals'})
        ])

        #sorting data by date
        goals_data_half.sort_values('date', inplace=True)
    
        goals_data_half['cumulative_first_half_goals'] = goals_data_half.groupby('team')['first_half_goals'].cumsum()
        goals_data_half['cumulative_second_half_goals'] = goals_data_half.groupby('team')['second_half_goals'].cumsum()

        #counting number of games played
        goals_data_half['games_played'] = goals_data_half.groupby('team').cumcount() + 1

        #calculating average goals scores in both halfs
        goals_data_half['average_first_half_goals_per_game'] = goals_data_half['cumulative_first_half_goals'] / goals_data_half['games_played']
        goals_data_half['average_second_half_goals_per_game'] = goals_data_half['cumulative_second_half_goals'] / goals_data_half['games_played']
        
        goals_data_half['average_first_half_goals_per_game_pre_game'] = goals_data_half.groupby('team')['average_first_half_goals_per_game'].shift().fillna(0)
        goals_data_half['average_second_half_goals_per_game_pre_game'] = goals_data_half.groupby('team')['average_second_half_goals_per_game'].shift().fillna(0)

        #merging back to original dataframe
        df = df.merge(goals_data_half[['date', 'team', 'average_first_half_goals_per_game_pre_game', 'average_second_half_goals_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_first_half_goals_per_game_pre_game': 'home_team_average_first_half_goals_total_pre_game', 'average_second_half_goals_per_game_pre_game': 'home_team_average_second_half_goals_total_pre_game'}).drop('team', axis=1)
        df = df.merge(goals_data_half[['date', 'team', 'average_first_half_goals_per_game_pre_game', 'average_second_half_goals_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_first_half_goals_per_game_pre_game': 'away_team_average_first_half_goals_total_pre_game', 'average_second_half_goals_per_game_pre_game': 'away_team_average_second_half_goals_total_pre_game'}).drop('team', axis=1)

        return df

    def add_average_total_corners_per_game_before_game(self, df):
        
        #creating working dataframe with number of corners
        corners = pd.concat([
            df[['date', 'home_team_name', 'corners_total']].rename(columns={'home_team_name': 'team', 'corners_total': 'corners'}),
            df[['date', 'away_team_name', 'corners_total']].rename(columns={'away_team_name': 'team', 'corners_total': 'corners'})
        ])

        #sorting data by date
        corners.sort_values('date', inplace=True)
    
        #counting number of games played
        corners['cumulative_corners'] = corners.groupby('team')['corners'].cumsum()
        corners['games_played'] = corners.groupby('team').cumcount() + 1  # Dodajemy 1, bo cumcount zaczyna od 0

        #calculating average cumulative corners
        corners['average_corners'] = corners['cumulative_corners'] / corners['games_played']

        #removing last game
        corners['average_corners_pre_game'] = corners.groupby('team')['average_corners'].shift().fillna(0)
        
        #merging back to original dataframe
        df = df.merge(corners[['date', 'team', 'average_corners_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_corners_pre_game': 'home_team_average_corners_total_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(corners[['date', 'team', 'average_corners_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_corners_pre_game': 'away_team_average_corners_total_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_corners_by_team_per_game_before_game(self, df):

        #creating working dataframe with corners by team
        corners = pd.concat([
            df[['date', 'home_team_name', 'home_team_corner_count']].rename(columns={'home_team_name': 'team', 'home_team_corner_count': 'corners'}),
            df[['date', 'away_team_name', 'away_team_corner_count']].rename(columns={'away_team_name': 'team', 'away_team_corner_count': 'corners'})
        ])

        #sorting data by date
        corners.sort_values('date', inplace=True)   
        
        #counting number of games
        corners['cumulative_corners'] = corners.groupby('team')['corners'].cumsum()
        corners['games_played'] = corners.groupby('team').cumcount() + 1  # Dodajemy 1, bo cumcount zaczyna od 0

        #calculating average cumulative
        corners['average_corners'] = corners['cumulative_corners'] / corners['games_played']

        #removing last game
        corners['average_corners_pre_game'] = corners.groupby('team')['average_corners'].shift().fillna(0)
        
        #merging back to original dataframe
        df = df.merge(corners[['date', 'team', 'average_corners_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_corners_pre_game': 'home_team_average_corners_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(corners[['date', 'team', 'average_corners_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_corners_pre_game': 'away_team_average_corners_per_game_pre_game'}).drop('team', axis=1)
    
        return df
    
    def add_average_yellow_cards_total_per_game_before_game(self, df):

        # Tworzenie DataFrame z łączną liczbą żółtych kartek zarówno w domu, jak i na wyjeździe
        cards = pd.concat([
            df[['date', 'home_team_name', 'yellow_cards_total']].rename(columns={'home_team_name': 'team', 'yellow_cards_total': 'yellow_cards'}),
            df[['date', 'away_team_name', 'yellow_cards_total']].rename(columns={'away_team_name': 'team', 'yellow_cards_total': 'yellow_cards'})
        ])

        # Sortowanie danych według daty, aby kumulatywne średnie były poprawne
        cards.sort_values('date', inplace=True)
        
        # Obliczanie sumy i liczby meczów
        cards['cumulative_yellow_cards'] = cards.groupby('team')['yellow_cards'].cumsum()
        cards['games_played'] = cards.groupby('team').cumcount() + 1  # Dodajemy 1, bo cumcount zaczyna od 0

        # Obliczanie średniej kumulatywnej
        cards['average_yellow_cards'] = cards['cumulative_yellow_cards'] / cards['games_played']

        # Usuwamy ostatni mecz, aby uzyskać średnią przed bieżącym meczem
        cards['average_yellow_cards_pre_game'] = cards.groupby('team')['average_yellow_cards'].shift().fillna(0)
        
        # Dodanie średniej liczby żółtych kartek do oryginalnego DataFrame
        df = df.merge(cards[['date', 'team', 'average_yellow_cards_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_yellow_cards_pre_game': 'home_team_average_yellow_cards_total_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(cards[['date', 'team', 'average_yellow_cards_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_yellow_cards_pre_game': 'away_team_average_yellow_cards_total_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_yellow_cards_by_team_per_game_before_game(self, df):
    
        # Tworzenie DataFrame z liczbą żółtych kartek dla gospodarzy i gości
        cards = pd.concat([
            df[['date', 'home_team_name', 'home_team_yellow_cards']].rename(columns={'home_team_name': 'team', 'home_team_yellow_cards': 'yellow_cards'}),
            df[['date', 'away_team_name', 'away_team_yellow_cards']].rename(columns={'away_team_name': 'team', 'away_team_yellow_cards': 'yellow_cards'})
        ])

        # Sortowanie danych według daty, aby kumulatywne średnie były poprawne
        cards.sort_values('date', inplace=True)
        
        # Obliczanie sumy i liczby meczów
        cards['cumulative_yellow_cards'] = cards.groupby('team')['yellow_cards'].cumsum()
        cards['games_played'] = cards.groupby('team').cumcount() + 1  # Dodajemy 1, bo cumcount zaczyna od 0

        # Obliczanie średniej kumulatywnej
        cards['average_yellow_cards'] = cards['cumulative_yellow_cards'] / cards['games_played']

        # Usuwamy ostatni mecz, aby uzyskać średnią przed bieżącym meczem
        cards['average_yellow_cards_pre_game'] = cards.groupby('team')['average_yellow_cards'].shift().fillna(0)
        
        # Dodanie średniej liczby żółtych kartek do oryginalnego DataFrame
        df = df.merge(cards[['date', 'team', 'average_yellow_cards_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_yellow_cards_pre_game': 'home_team_average_yellow_cards_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(cards[['date', 'team', 'average_yellow_cards_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_yellow_cards_pre_game': 'away_team_average_yellow_cards_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_cumulative_red_cards_by_team_before_game(self, df):
        df['home_team_received_red_card'] = df['home_team_red_cards'] > 0
        df['away_team_received_red_card'] = df['away_team_red_cards'] > 0
        
        # Tworzenie DataFrame z danymi dotyczącymi czerwonych kartek
        red_cards_data = pd.concat([
            df[['date', 'home_team_name', 'home_team_received_red_card']].rename(columns={'home_team_name': 'team', 'home_team_received_red_card': 'received_red_card'}),
            df[['date', 'away_team_name', 'away_team_received_red_card']].rename(columns={'away_team_name': 'team', 'away_team_received_red_card': 'received_red_card'})
        ])

        # Sortowanie danych według daty
        red_cards_data.sort_values('date', inplace=True)

        # Obliczanie kumulatywnych sum meczów z czerwoną kartką
        red_cards_data['cumulative_games_with_red_cards'] = red_cards_data.groupby('team')['received_red_card'].cumsum()

        # Usuwamy ostatni mecz, aby uzyskać dane przed bieżącym meczem
        red_cards_data['games_with_red_cards_pre_game'] = red_cards_data.groupby('team')['cumulative_games_with_red_cards'].shift().fillna(0)
        
        df = df.merge(red_cards_data[['date', 'team', 'games_with_red_cards_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'games_with_red_cards_pre_game': 'home_team_cumulative_red_cards_pre_game'}).drop('team', axis=1)
        df = df.merge(red_cards_data[['date', 'team', 'games_with_red_cards_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'games_with_red_cards_pre_game': 'away_team_cumulative_red_cards_pre_game'}).drop('team', axis=1)

        return df

    def add_average_total_red_cards_per_game_before_game(self, df):
        df['total_red_cards_in_match'] = df['home_team_red_cards'] + df['away_team_red_cards']
        
        # Agregacja danych o łącznych czerwonych kartkach dla drużyny gospodarzy i gości
        total_red_cards_data = pd.concat([
            df[['date', 'home_team_name', 'total_red_cards_in_match']].rename(columns={'home_team_name': 'team'}),
            df[['date', 'away_team_name', 'total_red_cards_in_match']].rename(columns={'away_team_name': 'team'})
        ])

        # Sortowanie danych według daty
        total_red_cards_data.sort_values('date', inplace=True)
        
        # Obliczanie kumulatywnych sum łącznych czerwonych kartek
        total_red_cards_data['cumulative_total_red_cards'] = total_red_cards_data.groupby('team')['total_red_cards_in_match'].cumsum()

        # Liczenie liczby meczów
        total_red_cards_data['games_played'] = total_red_cards_data.groupby('team').cumcount() + 1

        # Obliczanie średniej liczby łącznych czerwonych kartek na mecz
        total_red_cards_data['average_total_red_cards_per_game'] = total_red_cards_data['cumulative_total_red_cards'] / total_red_cards_data['games_played']

        # Przesunięcie o jeden wiersz, aby nie uwzględniać bieżącego meczu
        total_red_cards_data['average_total_red_cards_per_game_pre_game'] = total_red_cards_data.groupby('team')['average_total_red_cards_per_game'].shift().fillna(0)
        
        df = df.merge(total_red_cards_data[['date', 'team', 'average_total_red_cards_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_total_red_cards_per_game_pre_game': 'home_team_average_red_cards_total_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(total_red_cards_data[['date', 'team', 'average_total_red_cards_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_total_red_cards_per_game_pre_game': 'away_team_average_red_cards_total_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_red_cards_by_team_per_game_before_game(self, df):

        # Tworzenie DataFrame z danymi dotyczącymi czerwonych kartek
        red_cards_data = pd.concat([
            df[['date', 'home_team_name', 'home_team_red_cards']].rename(columns={'home_team_name': 'team', 'home_team_red_cards': 'red_cards'}),
            df[['date', 'away_team_name', 'away_team_red_cards']].rename(columns={'away_team_name': 'team', 'away_team_red_cards': 'red_cards'})
        ])

        # Sortowanie danych według daty
        red_cards_data.sort_values('date', inplace=True)
        
        # Obliczanie kumulatywnych sum czerwonych kartek
        red_cards_data['cumulative_red_cards'] = red_cards_data.groupby('team')['red_cards'].cumsum()

        # Liczenie liczby meczów
        red_cards_data['games_played'] = red_cards_data.groupby('team').cumcount() + 1

        # Obliczanie średniej liczby czerwonych kartek na mecz
        red_cards_data['average_red_cards_per_game'] = red_cards_data['cumulative_red_cards'] / red_cards_data['games_played']
        
        red_cards_data['average_red_cards_per_game_pre_game'] = red_cards_data.groupby('team')['average_red_cards_per_game'].shift().fillna(0)
        
        df = df.merge(red_cards_data[['date', 'team', 'average_red_cards_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_red_cards_per_game_pre_game': 'home_team_average_red_cards_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(red_cards_data[['date', 'team', 'average_red_cards_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_red_cards_per_game_pre_game': 'away_team_average_red_cards_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_shots_per_game_before_game(self, df):

        # Załóżmy, że `df` to Twój DataFrame z danymi
        # Tworzenie DataFrame z danymi dotyczącymi strzałów
        shots_data = pd.concat([
            df[['date', 'home_team_name', 'home_team_shots', 'home_team_shots_on_target']].rename(columns={'home_team_name': 'team', 'home_team_shots': 'shots', 'home_team_shots_on_target': 'shots_on_target'}),
            df[['date', 'away_team_name', 'away_team_shots', 'away_team_shots_on_target']].rename(columns={'away_team_name': 'team', 'away_team_shots': 'shots', 'away_team_shots_on_target': 'shots_on_target'})
        ])

        # Sortowanie danych według daty
        shots_data.sort_values('date', inplace=True)
        
        # Obliczanie kumulatywnych sum strzałów i strzałów celnym
        shots_data['cumulative_shots'] = shots_data.groupby('team')['shots'].cumsum()
        shots_data['cumulative_shots_on_target'] = shots_data.groupby('team')['shots_on_target'].cumsum()

        # Liczenie liczby meczów
        shots_data['games_played'] = shots_data.groupby('team').cumcount() + 1

        # Obliczanie średnich kumulatywnych na mecz
        shots_data['average_shots_per_game'] = shots_data['cumulative_shots'] / shots_data['games_played']
        shots_data['average_shots_on_target_per_game'] = shots_data['cumulative_shots_on_target'] / shots_data['games_played']
        
        shots_data['average_shots_per_game_pre_game'] = shots_data.groupby('team')['average_shots_per_game'].shift().fillna(0)
        shots_data['average_shots_on_target_per_game_pre_game'] = shots_data.groupby('team')['average_shots_on_target_per_game'].shift().fillna(0)
        
        df = df.merge(shots_data[['date', 'team', 'average_shots_per_game_pre_game', 'average_shots_on_target_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_shots_per_game_pre_game': 'home_team_average_shots_per_game_pre_game', 'average_shots_on_target_per_game_pre_game': 'home_team_average_shots_on_target_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(shots_data[['date', 'team', 'average_shots_per_game_pre_game', 'average_shots_on_target_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_shots_per_game_pre_game': 'away_team_average_shots_per_game_pre_game', 'average_shots_on_target_per_game_pre_game': 'away_team_average_shots_on_target_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_fouls_by_team_per_game_before_game(self, df):
        # Tworzenie DataFrame z danymi dotyczącymi fauli
        fouls_data = pd.concat([
            df[['date', 'home_team_name', 'home_team_fouls']].rename(columns={'home_team_name': 'team', 'home_team_fouls': 'fouls'}),
            df[['date', 'away_team_name', 'away_team_fouls']].rename(columns={'away_team_name': 'team', 'away_team_fouls': 'fouls'})
        ])

        # Sortowanie danych według daty
        fouls_data.sort_values('date', inplace=True)
        
        # Obliczanie kumulatywnych sum fauli
        fouls_data['cumulative_fouls'] = fouls_data.groupby('team')['fouls'].cumsum()

        # Liczenie liczby meczów
        fouls_data['games_played'] = fouls_data.groupby('team').cumcount() + 1

        # Obliczanie średnich kumulatywnych na mecz
        fouls_data['average_fouls_per_game'] = fouls_data['cumulative_fouls'] / fouls_data['games_played']
        
        fouls_data['average_fouls_per_game_pre_game'] = fouls_data.groupby('team')['average_fouls_per_game'].shift().fillna(0)
        
        df = df.merge(fouls_data[['date', 'team', 'average_fouls_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_fouls_per_game_pre_game': 'home_team_average_fouls_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(fouls_data[['date', 'team', 'average_fouls_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_fouls_per_game_pre_game': 'away_team_average_fouls_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_fouls_total_per_game_before_game(self, df):
        df['total_fouls'] = df['home_team_fouls'] + df['away_team_fouls']
        
        # Tworzenie DataFrame z danymi dotyczącymi łącznych fauli dla obu drużyn w meczu
        fouls_data = pd.concat([
            df[['date', 'home_team_name', 'total_fouls']].rename(columns={'home_team_name': 'team'}),
            df[['date', 'away_team_name', 'total_fouls']].rename(columns={'away_team_name': 'team'})
        ])

        # Sortowanie danych według daty
        fouls_data.sort_values('date', inplace=True)

        # Obliczanie kumulatywnej sumy łącznych fauli
        fouls_data['cumulative_total_fouls'] = fouls_data.groupby('team')['total_fouls'].cumsum()

        # Liczenie liczby meczów
        fouls_data['games_played'] = fouls_data.groupby('team').cumcount() + 1

        # Obliczanie średniej łącznej liczby fauli na mecz
        fouls_data['average_total_fouls_per_game'] = fouls_data['cumulative_total_fouls'] / fouls_data['games_played']

        # Usuwamy ostatni mecz, aby uzyskać dane przed bieżącym meczem
        fouls_data['average_total_fouls_per_game_pre_game'] = fouls_data.groupby('team')['average_total_fouls_per_game'].shift().fillna(0)
        
        df = df.merge(fouls_data[['date', 'team', 'average_total_fouls_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_total_fouls_per_game_pre_game': 'home_team_average_fouls_total_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(fouls_data[['date', 'team', 'average_total_fouls_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_total_fouls_per_game_pre_game': 'away_team_average_fouls_total_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_ball_possession_per_game_before_game(self, df):
        # Tworzenie DataFrame z danymi dotyczącymi posiadania piłki
        possession_data = pd.concat([
            df[['date', 'home_team_name', 'home_team_possession']].rename(columns={'home_team_name': 'team', 'home_team_possession': 'possession'}),
            df[['date', 'away_team_name', 'away_team_possession']].rename(columns={'away_team_name': 'team', 'away_team_possession': 'possession'})
        ])

        # Sortowanie danych według daty
        possession_data.sort_values('date', inplace=True)
        
        # Obliczanie kumulatywnych sum posiadania piłki
        possession_data['cumulative_possession'] = possession_data.groupby('team')['possession'].cumsum()

        # Liczenie liczby meczów
        possession_data['games_played'] = possession_data.groupby('team').cumcount() + 1

        # Obliczanie średniego posiadania piłki na mecz
        possession_data['average_possession_per_game'] = possession_data['cumulative_possession'] / possession_data['games_played']
        
        possession_data['average_possession_per_game_pre_game'] = possession_data.groupby('team')['average_possession_per_game'].shift().fillna(0)
        
        df = df.merge(possession_data[['date', 'team', 'average_possession_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_possession_per_game_pre_game': 'home_team_average_possession_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(possession_data[['date', 'team', 'average_possession_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_possession_per_game_pre_game': 'away_team_average_possession_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_average_xg_per_game_before_game(self, df):
        # Tworzenie DataFrame z danymi dotyczącymi xG
        xg_data = pd.concat([
            df[['date', 'home_team_name', 'team_a_xg']].rename(columns={'home_team_name': 'team', 'team_a_xg': 'xg'}),
            df[['date', 'away_team_name', 'team_b_xg']].rename(columns={'away_team_name': 'team', 'team_b_xg': 'xg'})
        ])

        # Sortowanie danych według daty
        xg_data.sort_values('date', inplace=True)
        
        # Obliczanie kumulatywnych sum xG
        xg_data['cumulative_xg'] = xg_data.groupby('team')['xg'].cumsum()

        # Liczenie liczby meczów
        xg_data['games_played'] = xg_data.groupby('team').cumcount() + 1

        # Obliczanie średniej xG na mecz
        xg_data['average_xg_per_game'] = xg_data['cumulative_xg'] / xg_data['games_played']
        
        xg_data['average_xg_per_game_pre_game'] = xg_data.groupby('team')['average_xg_per_game'].shift().fillna(0)

        df = df.merge(xg_data[['date', 'team', 'average_xg_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_xg_per_game_pre_game': 'home_team_average_xg_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(xg_data[['date', 'team', 'average_xg_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_xg_per_game_pre_game': 'away_team_average_xg_per_game_pre_game'}).drop('team', axis=1)
    
        return df
    
    def add_no_goals_scored_cumulative_before_game(self, df):

        df['home_team_failed_to_score'] = df['home_team_goal_count'] == 0
        df['away_team_failed_to_score'] = df['away_team_goal_count'] == 0
        
        # Tworzenie DataFrame z danymi dotyczącymi braku strzelonych bramek
        no_goals_data = pd.concat([
            df[['date', 'home_team_name', 'home_team_failed_to_score']].rename(columns={'home_team_name': 'team', 'home_team_failed_to_score': 'failed_to_score'}),
            df[['date', 'away_team_name', 'away_team_failed_to_score']].rename(columns={'away_team_name': 'team', 'away_team_failed_to_score': 'failed_to_score'})
        ])

        # Sortowanie danych według daty
        no_goals_data.sort_values('date', inplace=True)

        # Obliczanie kumulatywnych sum meczów bez strzelonych bramek
        no_goals_data['cumulative_games_without_goals'] = no_goals_data.groupby('team')['failed_to_score'].cumsum()

        # Usuwamy ostatni mecz, aby uzyskać dane przed bieżącym meczem
        no_goals_data['games_without_goals_pre_game'] = no_goals_data.groupby('team')['cumulative_games_without_goals'].shift().fillna(0)
        
        df = df.merge(no_goals_data[['date', 'team', 'games_without_goals_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'games_without_goals_pre_game': 'home_team_games_without_goals_pre_game'}).drop('team', axis=1)
        df = df.merge(no_goals_data[['date', 'team', 'games_without_goals_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'games_without_goals_pre_game': 'away_team_games_without_goals_pre_game'}).drop('team', axis=1)
        
        return df
        
    def add_btts_cumulative_before_game(self, df):
        df['both_teams_scored'] = (df['home_team_goal_count'] > 0) & (df['away_team_goal_count'] > 0)
        
        # Tworzenie DataFrame z danymi dotyczącymi, czy obie drużyny strzeliły
        scored_data = pd.concat([
            df[['date', 'home_team_name', 'both_teams_scored']].rename(columns={'home_team_name': 'team'}),
            df[['date', 'away_team_name', 'both_teams_scored']].rename(columns={'away_team_name': 'team'})
        ])

        # Sortowanie danych według daty
        scored_data.sort_values('date', inplace=True)

        # Obliczanie kumulatywnej sumy meczów, gdzie obie drużyny strzeliły
        scored_data['cumulative_both_scored'] = scored_data.groupby('team')['both_teams_scored'].cumsum()

        # Usuwamy ostatni mecz, aby uzyskać dane przed bieżącym meczem
        scored_data['cumulative_both_scored_pre_game'] = scored_data.groupby('team')['cumulative_both_scored'].shift().fillna(0)
        
        df = df.merge(scored_data[['date', 'team', 'cumulative_both_scored_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_both_scored_pre_game': 'home_team_cumulative_btts_pre_game'}).drop('team', axis=1)
        df = df.merge(scored_data[['date', 'team', 'cumulative_both_scored_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_both_scored_pre_game': 'away_team_cumulative_btts_pre_game'}).drop('team', axis=1)

        return df

    def add_clean_sheets_cumulative_before_game(self, df):
        df['home_clean_sheet'] = df['away_team_goal_count'] == 0
        df['away_clean_sheet'] = df['home_team_goal_count'] == 0
        
        # Tworzenie DataFrame z danymi dotyczącymi czystych kont
        clean_sheets_data = pd.concat([
            df[['date', 'home_team_name', 'home_clean_sheet']].rename(columns={'home_team_name': 'team', 'home_clean_sheet': 'clean_sheet'}),
            df[['date', 'away_team_name', 'away_clean_sheet']].rename(columns={'away_team_name': 'team', 'away_clean_sheet': 'clean_sheet'})
        ])

        # Sortowanie danych według daty
        clean_sheets_data.sort_values('date', inplace=True)

        # Obliczanie kumulatywnej sumy czystych kont
        clean_sheets_data['cumulative_clean_sheets'] = clean_sheets_data.groupby('team')['clean_sheet'].cumsum()

        # Usuwamy ostatni mecz, aby uzyskać dane przed bieżącym meczem
        clean_sheets_data['cumulative_clean_sheets_pre_game'] = clean_sheets_data.groupby('team')['cumulative_clean_sheets'].shift().fillna(0)
        
        df = df.merge(clean_sheets_data[['date', 'team', 'cumulative_clean_sheets_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_clean_sheets_pre_game': 'home_team_cumulative_clean_sheets_pre_game'}).drop('team', axis=1)
        df = df.merge(clean_sheets_data[['date', 'team', 'cumulative_clean_sheets_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_clean_sheets_pre_game': 'away_team_cumulative_clean_sheets_pre_game'}).drop('team', axis=1)

        return df

    def add_wins_losses_draws_before_game(self, df):
        
        df['home_win'] = df['home_team_goal_count'] > df['away_team_goal_count']
        df['away_win'] = df['home_team_goal_count'] < df['away_team_goal_count']
        df['draw'] = df['home_team_goal_count'] == df['away_team_goal_count']
        
        # Tworzenie DataFrame z danymi dotyczącymi wyników meczów
        results_data = pd.concat([
            df[['date', 'home_team_name', 'home_win', 'draw', 'away_win']].rename(columns={'home_team_name': 'team', 'home_win': 'win', 'away_win': 'loss'}),
            df[['date', 'away_team_name', 'away_win', 'draw', 'home_win']].rename(columns={'away_team_name': 'team', 'away_win': 'win', 'home_win': 'loss'})
        ])

        # Sortowanie danych według daty
        results_data.sort_values('date', inplace=True)

        # Obliczanie kumulatywnych sum zwycięstw, remisów i porażek
        results_data['cumulative_wins'] = results_data.groupby('team')['win'].cumsum()
        results_data['cumulative_draws'] = results_data.groupby('team')['draw'].cumsum()
        results_data['cumulative_losses'] = results_data.groupby('team')['loss'].cumsum()

        # Usuwamy ostatni mecz, aby uzyskać dane przed bieżącym meczem
        results_data['cumulative_wins_pre_game'] = results_data.groupby('team')['cumulative_wins'].shift().fillna(0)
        results_data['cumulative_draws_pre_game'] = results_data.groupby('team')['cumulative_draws'].shift().fillna(0)
        results_data['cumulative_losses_pre_game'] = results_data.groupby('team')['cumulative_losses'].shift().fillna(0)
        
        df = df.merge(results_data[['date', 'team', 'cumulative_wins_pre_game', 'cumulative_draws_pre_game', 'cumulative_losses_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_wins_pre_game': 'home_team_cumulative_wins_pre_game', 'cumulative_draws_pre_game': 'home_team_cumulative_draws_pre_game', 'cumulative_losses_pre_game': 'home_team_cumulative_losses_pre_game'}).drop('team', axis=1)
        df = df.merge(results_data[['date', 'team', 'cumulative_wins_pre_game', 'cumulative_draws_pre_game', 'cumulative_losses_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'cumulative_wins_pre_game': 'away_team_cumulative_wins_pre_game', 'cumulative_draws_pre_game': 'away_team_cumulative_draws_pre_game', 'cumulative_losses_pre_game': 'away_team_cumulative_losses_pre_game'}).drop('team', axis=1)

        return df

    def add_points_per_game_before_game(self, df):
        
        df['home_win'] = df['home_team_goal_count'] > df['away_team_goal_count']
        df['away_win'] = df['home_team_goal_count'] < df['away_team_goal_count']
        df['draw'] = df['home_team_goal_count'] == df['away_team_goal_count']

        df['home_points'] = df['home_win'] * 3 + df['draw'] * 1
        df['away_points'] = df['away_win'] * 3 + df['draw'] * 1
        
        points_data = pd.concat([
            df[['date', 'home_team_name', 'home_points']].rename(columns={'home_team_name': 'team', 'home_points': 'points'}),
            df[['date', 'away_team_name', 'away_points']].rename(columns={'away_team_name': 'team', 'away_points': 'points'})
        ])

        # Sortowanie danych według daty
        points_data.sort_values('date', inplace=True)
        
        # Obliczanie kumulatywnych sum punktów
        points_data['cumulative_points'] = points_data.groupby('team')['points'].cumsum()

        # Liczenie liczby meczów
        points_data['games_played'] = points_data.groupby('team').cumcount() + 1

        # Obliczanie średniej liczby punktów na mecz
        points_data['average_points_per_game'] = points_data['cumulative_points'] / points_data['games_played']

        # Usuwamy ostatni mecz, aby uzyskać dane przed bieżącym meczem
        points_data['average_points_per_game_pre_game'] = points_data.groupby('team')['average_points_per_game'].shift().fillna(0)
        
        df = df.merge(points_data[['date', 'team', 'average_points_per_game_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_points_per_game_pre_game': 'home_team_average_points_per_game_pre_game'}).drop('team', axis=1)
        df = df.merge(points_data[['date', 'team', 'average_points_per_game_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'average_points_per_game_pre_game': 'away_team_average_points_per_game_pre_game'}).drop('team', axis=1)

        return df

    def add_form_in_last_5_games(self, df):
        # Tworzenie flag dla wyników meczu
        df['home_win'] = df['home_team_goal_count'] > df['away_team_goal_count']
        df['away_win'] = df['home_team_goal_count'] < df['away_team_goal_count']
        df['draw'] = df['home_team_goal_count'] == df['away_team_goal_count']
        
        # Przygotowanie ramki danych dla każdej drużyny, gospodarzy i gości
        home_form = df[['date', 'home_team_name', 'home_win', 'draw', 'away_win']].rename(columns={'home_team_name': 'team', 'home_win': 'win', 'away_win': 'loss'})
        away_form = df[['date', 'away_team_name', 'away_win', 'draw', 'home_win']].rename(columns={'away_team_name': 'team', 'away_win': 'win', 'home_win': 'loss'})

        # Łączenie danych dla gospodarzy i gości
        all_form = pd.concat([home_form, away_form], axis=0).sort_values(by=['team', 'date'])

        # Grupowanie po drużynach i datach, i obliczenie sumy kroczącej dla 5 ostatnich meczów
        all_form['last_5_wins'] = all_form.groupby('team')['win'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        all_form['last_5_draws'] = all_form.groupby('team')['draw'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        all_form['last_5_losses'] = all_form.groupby('team')['loss'].transform(lambda x: x.rolling(5, min_periods=1).sum().shift(1))
        
        # Przyłączanie danych o formie do głównego DataFrame
        df = df.merge(all_form[['date', 'team', 'last_5_wins', 'last_5_draws', 'last_5_losses']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left', suffixes=('', '_home'))
        df = df.merge(all_form[['date', 'team', 'last_5_wins', 'last_5_draws', 'last_5_losses']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left', suffixes=('', '_away'))

        df.rename(columns={
            'last_5_wins': 'home_team_wins_in_last_5_games',
            'last_5_draws': 'home_team_draws_in_last_5_games',
            'last_5_losses': 'home_team_losses_in_last_5_games',
            'last_5_wins_away': 'away_team_wins_in_last_5_games',
            'last_5_draws_away': 'away_team_draws_in_last_5_games',
            'last_5_losses_away': 'away_team_losses_in_last_5_games'
        }, inplace=True)
        
        df[['home_team_wins_in_last_5_games', 'home_team_draws_in_last_5_games', 'home_team_losses_in_last_5_games', 
            'away_team_wins_in_last_5_games', 'away_team_draws_in_last_5_games', 'away_team_losses_in_last_5_games']].fillna(0, inplace=True)
        
        return df
        
    def add_rolling_goals_scored_and_conceded_in_last_5_games(self, df):
    
        # Dodanie kolumn z goli zdobytymi przez gospodarzy i gości oraz straconymi
        df['home_goals_scored'] = df['home_team_goal_count']
        df['away_goals_scored'] = df['away_team_goal_count']
        df['home_goals_conceded'] = df['away_team_goal_count']
        df['away_goals_conceded'] = df['home_team_goal_count']

        # Tworzenie DataFrame z danymi dotyczącymi zdobytych i straconych goli
        goals_data = pd.concat([
            df[['date', 'home_team_name', 'home_goals_scored', 'home_goals_conceded']].rename(columns={'home_team_name': 'team', 'home_goals_scored': 'goals_scored', 'home_goals_conceded': 'goals_conceded'}),
            df[['date', 'away_team_name', 'away_goals_scored', 'away_goals_conceded']].rename(columns={'away_team_name': 'team', 'away_goals_scored': 'goals_scored', 'away_goals_conceded': 'goals_conceded'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów
        goals_data['rolling_goals_scored'] = goals_data.groupby('team')['goals_scored'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        goals_data['rolling_goals_conceded'] = goals_data.groupby('team')['goals_conceded'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        goals_data['rolling_goals_scored_pre_game'] = goals_data.groupby('team')['rolling_goals_scored'].shift().fillna(0)
        goals_data['rolling_goals_conceded_pre_game'] = goals_data.groupby('team')['rolling_goals_conceded'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(goals_data[['date', 'team', 'rolling_goals_scored_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_goals_scored_pre_game': 'home_team_average_goals_scored_in_last_5_games'})
        df = df.merge(goals_data[['date', 'team', 'rolling_goals_scored_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_goals_scored_pre_game': 'away_team_average_goals_scored_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(goals_data[['date', 'team', 'rolling_goals_conceded_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_goals_conceded_pre_game': 'home_team_average_goals_conceded_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(goals_data[['date', 'team', 'rolling_goals_conceded_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_goals_conceded_pre_game': 'away_team_average_goals_conceded_in_last_5_games'}).drop('team', axis=1)

        return df
    
    def add_rolling_corners_total_in_last_5_games(self, df):
        # Dodanie kolumn z rzutami rożnymi wykonanymi przez gospodarzy i gości
        df['home_corners'] = df['home_team_corner_count']
        df['away_corners'] = df['away_team_corner_count']

        # Tworzenie DataFrame z danymi dotyczącymi rzutów rożnych
        corners_data = pd.concat([
            df[['date', 'home_team_name', 'home_corners']].rename(columns={'home_team_name': 'team', 'home_corners': 'corners'}),
            df[['date', 'away_team_name', 'away_corners']].rename(columns={'away_team_name': 'team', 'away_corners': 'corners'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów
        corners_data['rolling_corners'] = corners_data.groupby('team')['corners'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        corners_data['rolling_corners_pre_game'] = corners_data.groupby('team')['rolling_corners'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(corners_data[['date', 'team', 'rolling_corners_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_corners_pre_game': 'home_team_rolling_corners_pre_game'}).drop('team', axis=1)
        df = df.merge(corners_data[['date', 'team', 'rolling_corners_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_corners_pre_game': 'away_team_rolling_corners_pre_game'}).drop('team', axis=1)

        # Obliczanie całkowitej liczby rzutów rożnych w meczach danej drużyny
        df['total_corners'] = df['home_corners'] + df['away_corners']
        total_corners_data = df[['date', 'home_team_name', 'away_team_name', 'total_corners']].melt(id_vars=['date', 'total_corners'], value_vars=['home_team_name', 'away_team_name'], var_name='role', value_name='team')
        total_corners_data['rolling_total_corners'] = total_corners_data.groupby('team')['total_corners'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        total_corners_data['rolling_total_corners_pre_game'] = total_corners_data.groupby('team')['rolling_total_corners'].shift().fillna(0)

        df = df.merge(total_corners_data[['date', 'team', 'rolling_total_corners_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_total_corners_pre_game': 'average_total_corners_in_home_team_games_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(total_corners_data[['date', 'team', 'rolling_total_corners_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_total_corners_pre_game': 'average_total_corners_in_away_team_games_in_last_5_games'}).drop('team', axis=1)

        return df
    
    def add_rolling_corners_by_team_in_last_5_games(self, df):
        # Przygotowanie danych z rzutami rożnymi dla gospodarzy i gości
        df['home_corners'] = df['home_team_corner_count']
        df['away_corners'] = df['away_team_corner_count']
        
        # Tworzenie DataFrame z danymi dotyczącymi rzutów rożnych
        corners_data = pd.concat([
            df[['date', 'home_team_name', 'home_corners']].rename(columns={'home_team_name': 'team', 'home_corners': 'corners'}),
            df[['date', 'away_team_name', 'away_corners']].rename(columns={'away_team_name': 'team', 'away_corners': 'corners'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów dla rzutów rożnych
        corners_data['rolling_corners'] = corners_data.groupby('team')['corners'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        corners_data['rolling_corners_pre_game'] = corners_data.groupby('team')['rolling_corners'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym dla rzutów rożnych jako gospodarz i gość
        df = df.merge(corners_data[['date', 'team', 'rolling_corners_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_corners_pre_game': 'average_corners_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(corners_data[['date', 'team', 'rolling_corners_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_corners_pre_game': 'average_corners_by_away_team_in_last_5_games'}).drop('team', axis=1)

        return df
            
    def add_rolling_cards_by_team_in_last_5_games(self, df):
        # Dodanie kolumn z kartkami zdobytymi przez gospodarzy i gości
        df['home_yellow_cards'] = df['home_team_yellow_cards']
        df['away_yellow_cards'] = df['away_team_yellow_cards']
        df['home_red_cards'] = df['home_team_red_cards']
        df['away_red_cards'] = df['away_team_red_cards']

        # Tworzenie DataFrame z danymi dotyczącymi kartek
        cards_data = pd.concat([
            df[['date', 'home_team_name', 'home_yellow_cards', 'home_red_cards']].rename(columns={'home_team_name': 'team', 'home_yellow_cards': 'yellow_cards', 'home_red_cards': 'red_cards'}),
            df[['date', 'away_team_name', 'away_yellow_cards', 'away_red_cards']].rename(columns={'away_team_name': 'team', 'away_yellow_cards': 'yellow_cards', 'away_red_cards': 'red_cards'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów
        cards_data['rolling_yellow_cards'] = cards_data.groupby('team')['yellow_cards'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        cards_data['rolling_red_cards'] = cards_data.groupby('team')['red_cards'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        cards_data['rolling_yellow_cards_pre_game'] = cards_data.groupby('team')['rolling_yellow_cards'].shift().fillna(0)
        cards_data['rolling_red_cards_pre_game'] = cards_data.groupby('team')['rolling_red_cards'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(cards_data[['date', 'team', 'rolling_yellow_cards_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_yellow_cards_pre_game': 'average_yellow_cards_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(cards_data[['date', 'team', 'rolling_yellow_cards_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_yellow_cards_pre_game': 'average_yellow_cards_by_away_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(cards_data[['date', 'team', 'rolling_red_cards_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_red_cards_pre_game': 'average_red_cards_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(cards_data[['date', 'team', 'rolling_red_cards_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_red_cards_pre_game': 'average_red_cards_by_away_team_in_last_5_games'}).drop('team', axis=1)
    
        return df
    
    def add_rolling_ball_possession_by_team_in_last_5_games(self, df):
        # Dodanie kolumn z posiadaniem piłki przez gospodarzy i gości
        df['home_possession'] = df['home_team_possession']
        df['away_possession'] = df['away_team_possession']

        # Tworzenie DataFrame z danymi dotyczącymi posiadania piłki
        possession_data = pd.concat([
            df[['date', 'home_team_name', 'home_possession']].rename(columns={'home_team_name': 'team', 'home_possession': 'possession'}),
            df[['date', 'away_team_name', 'away_possession']].rename(columns={'away_team_name': 'team', 'away_possession': 'possession'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów
        possession_data['rolling_possession'] = possession_data.groupby('team')['possession'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        possession_data['rolling_possession_pre_game'] = possession_data.groupby('team')['rolling_possession'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(possession_data[['date', 'team', 'rolling_possession_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_possession_pre_game': 'average_ball_possession_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(possession_data[['date', 'team', 'rolling_possession_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_possession_pre_game': 'average_ball_possession_by_away_team_in_last_5_games'}).drop('team', axis=1)
    
        return df
    
    def add_rolling_xg_by_team_in_last_5_games(self, df):
        # Dodanie kolumn z expected goals (xG) zdobytymi przez gospodarzy i gości
        df['home_xg'] = df['team_a_xg']
        df['away_xg'] = df['team_b_xg']

        # Tworzenie DataFrame z danymi dotyczącymi expected goals
        xg_data = pd.concat([
            df[['date', 'home_team_name', 'home_xg']].rename(columns={'home_team_name': 'team', 'home_xg': 'xg'}),
            df[['date', 'away_team_name', 'away_xg']].rename(columns={'away_team_name': 'team', 'away_xg': 'xg'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów dla xG
        xg_data['rolling_xg'] = xg_data.groupby('team')['xg'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        xg_data['rolling_xg_pre_game'] = xg_data.groupby('team')['rolling_xg'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(xg_data[['date', 'team', 'rolling_xg_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_xg_pre_game': 'average_xg_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(xg_data[['date', 'team', 'rolling_xg_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_xg_pre_game': 'average_xg_by_away_team_in_last_5_games'}).drop('team', axis=1)

        return df
    
    def add_rolling_shots_by_team_in_last_5_games(self, df):
        # Dodanie kolumn ze strzałami i strzałami celnymi wykonanymi przez gospodarzy i gości
        df['home_shots'] = df['home_team_shots']
        df['away_shots'] = df['away_team_shots']
        df['home_shots_on_target'] = df['home_team_shots_on_target']
        df['away_shots_on_target'] = df['away_team_shots_on_target']

        # Tworzenie DataFrame z danymi dotyczącymi strzałów
        shots_data = pd.concat([
            df[['date', 'home_team_name', 'home_shots', 'home_shots_on_target']].rename(columns={'home_team_name': 'team', 'home_shots': 'shots', 'home_shots_on_target': 'shots_on_target'}),
            df[['date', 'away_team_name', 'away_shots', 'away_shots_on_target']].rename(columns={'away_team_name': 'team', 'away_shots': 'shots', 'away_shots_on_target': 'shots_on_target'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów dla strzałów i strzałów celnych
        shots_data['rolling_shots'] = shots_data.groupby('team')['shots'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        shots_data['rolling_shots_on_target'] = shots_data.groupby('team')['shots_on_target'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        shots_data['rolling_shots_pre_game'] = shots_data.groupby('team')['rolling_shots'].shift().fillna(0)
        shots_data['rolling_shots_on_target_pre_game'] = shots_data.groupby('team')['rolling_shots_on_target'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(shots_data[['date', 'team', 'rolling_shots_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_shots_pre_game': 'average_shots_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(shots_data[['date', 'team', 'rolling_shots_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_shots_pre_game': 'average_shots_by_away_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(shots_data[['date', 'team', 'rolling_shots_on_target_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_shots_on_target_pre_game': 'average_shots_on_target_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(shots_data[['date', 'team', 'rolling_shots_on_target_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_shots_on_target_pre_game': 'average_shots_on_target_by_away_team_in_last_5_games'}).drop('team', axis=1)

        return df
    
    def add_rolling_fouls_total_in_last_5_games(self, df):
        # Dodanie kolumn z faulami wykonanymi przez gospodarzy i gości
        df['home_fouls'] = df['home_team_fouls']
        df['away_fouls'] = df['away_team_fouls']

        # Tworzenie DataFrame z danymi dotyczącymi fauli
        fouls_data = pd.concat([
            df[['date', 'home_team_name', 'home_fouls']].rename(columns={'home_team_name': 'team', 'home_fouls': 'fouls'}),
            df[['date', 'away_team_name', 'away_fouls']].rename(columns={'away_team_name': 'team', 'away_fouls': 'fouls'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów dla fauli
        fouls_data['rolling_fouls'] = fouls_data.groupby('team')['fouls'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        fouls_data['rolling_fouls_pre_game'] = fouls_data.groupby('team')['rolling_fouls'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(fouls_data[['date', 'team', 'rolling_fouls_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_fouls_pre_game': 'home_team_rolling_fouls_pre_game'}).drop('team', axis=1)
        df = df.merge(fouls_data[['date', 'team', 'rolling_fouls_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_fouls_pre_game': 'away_team_rolling_fouls_pre_game'}).drop('team', axis=1)

        # Obliczanie całkowitej liczby fauli w meczach danej drużyny
        df['total_fouls'] = df['home_fouls'] + df['away_fouls']
        total_fouls_data = df[['date', 'home_team_name', 'away_team_name', 'total_fouls']].melt(id_vars=['date', 'total_fouls'], value_vars=['home_team_name', 'away_team_name'], var_name='role', value_name='team')
        total_fouls_data['rolling_total_fouls'] = total_fouls_data.groupby('team')['total_fouls'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        total_fouls_data['rolling_total_fouls_pre_game'] = total_fouls_data.groupby('team')['rolling_total_fouls'].shift().fillna(0)

        df = df.merge(total_fouls_data[['date', 'team', 'rolling_total_fouls_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_total_fouls_pre_game': 'average_fouls_total_in_home_team_games_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(total_fouls_data[['date', 'team', 'rolling_total_fouls_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_total_fouls_pre_game': 'average_fouls_total_in_away_team_games_in_last_5_games'}).drop('team', axis=1)

        return df

    def add_rolling_fouls_by_team_in_last_5_games(self, df):
        # Dodanie kolumn z faulami wykonanymi przez gospodarzy i gości
        df['home_fouls'] = df['home_team_fouls']
        df['away_fouls'] = df['away_team_fouls']

        # Tworzenie DataFrame z danymi dotyczącymi fauli
        fouls_data = pd.concat([
            df[['date', 'home_team_name', 'home_fouls']].rename(columns={'home_team_name': 'team', 'home_fouls': 'fouls'}),
            df[['date', 'away_team_name', 'away_fouls']].rename(columns={'away_team_name': 'team', 'away_fouls': 'fouls'})
        ]).sort_values('date')

        # Obliczanie średniej kroczącej z 5 ostatnich meczów dla fauli
        fouls_data['rolling_fouls'] = fouls_data.groupby('team')['fouls'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Przesunięcie wyników o jeden wiersz wstecz, aby nie uwzględniać bieżącego meczu
        fouls_data['rolling_fouls_pre_game'] = fouls_data.groupby('team')['rolling_fouls'].shift().fillna(0)

        # Mergowanie wyników z DataFrame głównym
        df = df.merge(fouls_data[['date', 'team', 'rolling_fouls_pre_game']], left_on=['date', 'home_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_fouls_pre_game': 'average_fouls_by_home_team_in_last_5_games'}).drop('team', axis=1)
        df = df.merge(fouls_data[['date', 'team', 'rolling_fouls_pre_game']], left_on=['date', 'away_team_name'], right_on=['date', 'team'], how='left').rename(columns={'rolling_fouls_pre_game': 'average_fouls_by_away_team_in_last_5_games'}).drop('team', axis=1)

        return df

    def remove_working_variables(self):
        self.data.drop(['Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'Home Team Pre-Match xG', 'Away Team Pre-Match xG',
                        'average_goals_per_match_pre_match', 'home_goals_scored', 'away_goals_scored', 'home_goals_conceded',
                        'away_goals_conceded', 'home_team_second_half_goals', 'away_team_second_half_goals', 'home_goals_conceded_first_half',
                        'away_goals_conceded_first_half', 'home_goals_conceded_second_half', 'away_goals_conceded_second_half',
                        'home_first_half_goals', 'home_second_half_goals', 'away_first_half_goals', 'away_second_half_goals',
                        'home_team_received_red_card', 'away_team_received_red_card', 'total_red_cards_in_match', 'total_fouls',
                        'home_team_failed_to_score', 'away_team_failed_to_score', 'both_teams_scored', 'home_clean_sheet',
                        'away_clean_sheet', 'home_win', 'away_win', 'draw', 'home_points', 'away_points', 'team_x', 'team_away',
                        'team_y', 'home_corners', 'away_corners', 'total_corners', 'home_yellow_cards', 'away_yellow_cards',
                        'home_red_cards', 'away_red_cards', 'home_possession', 'away_possession', 'home_xg', 'away_xg', 'home_shots',
                        'away_shots', 'home_shots_on_target', 'away_shots_on_target', 'home_fouls', 'away_fouls', 'home_team_rolling_fouls_pre_game',
                        'away_team_rolling_fouls_pre_game'], 
                       axis=1, inplace=True)
        
        return self.data

    def run(self):
        self.data = self.clean_data(self.data)
        self.data = self.add_cumulative_goals_scored_before_game(self.data)
        self.data = self.add_cumulative_goals_conceded_before_game(self.data)
        self.data = self.add_average_goals_scored_and_conceded_per_game_before_game(self.data)
        self.data = self.add_average_goals_scored_in_first_and_second_half_per_game_before_game(self.data)
        self.data = self.add_average_goals_conceded_in_first_and_second_half_per_game_before_game(self.data)
        self.data = self.add_average_goals_total_in_first_and_second_half_per_game_before_game(self.data)
        self.data = self.add_average_total_corners_per_game_before_game(self.data)
        self.data = self.add_average_corners_by_team_per_game_before_game(self.data)
        self.data = self.add_average_yellow_cards_total_per_game_before_game(self.data)
        self.data = self.add_average_yellow_cards_by_team_per_game_before_game(self.data)
        self.data = self.add_cumulative_red_cards_by_team_before_game(self.data)
        self.data = self.add_average_total_red_cards_per_game_before_game(self.data)
        self.data = self.add_average_red_cards_by_team_per_game_before_game(self.data)
        self.data = self.add_average_shots_per_game_before_game(self.data)
        self.data = self.add_average_fouls_by_team_per_game_before_game(self.data)
        self.data = self.add_average_fouls_total_per_game_before_game(self.data)
        self.data = self.add_average_ball_possession_per_game_before_game(self.data)
        self.data = self.add_average_xg_per_game_before_game(self.data)
        self.data = self.add_no_goals_scored_cumulative_before_game(self.data)
        self.data = self.add_btts_cumulative_before_game(self.data)
        self.data = self.add_clean_sheets_cumulative_before_game(self.data)
        self.data = self.add_wins_losses_draws_before_game(self.data)
        self.data = self.add_points_per_game_before_game(self.data)
        self.data = self.add_form_in_last_5_games(self.data)
        self.data = self.add_rolling_goals_scored_and_conceded_in_last_5_games(self.data)
        self.data = self.add_rolling_corners_total_in_last_5_games(self.data)
        self.data = self.add_rolling_corners_by_team_in_last_5_games(self.data)
        self.data = self.add_rolling_cards_by_team_in_last_5_games(self.data)
        self.data = self.add_rolling_ball_possession_by_team_in_last_5_games(self.data)
        self.data = self.add_rolling_xg_by_team_in_last_5_games(self.data)
        self.data = self.add_rolling_shots_by_team_in_last_5_games(self.data)
        self.data = self.add_rolling_fouls_total_in_last_5_games(self.data)
        self.data = self.add_rolling_fouls_by_team_in_last_5_games(self.data)
        
        self.data = self.remove_working_variables()
        
        return self.data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for football analytics.')
    parser.add_argument('data_filepath', type=str, help='Path to the CSV file containing the data.')
    args = parser.parse_args()
    data = pd.read_csv(args.data_filepath)
    wrapper = FootballPredictorDataWrapper(data)
    data_processed = wrapper.run()
    data_processed.to_csv('test.csv', index=False)