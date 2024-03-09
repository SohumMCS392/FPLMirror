import pandas as pd

#get data
players_raw = pd.DataFrame(pd.read_csv("./data/2023-24/players_raw.csv"))
name_data = pd.DataFrame(pd.read_csv('./data/2023-24/player_idlist.csv'))

#merge on id
merged_df = pd.merge(players_raw, name_data, on='id', how='right')

#make id the key
merged_df.set_index('id', inplace=True)

#deleting unwanted data
columns_to_drop = ["chance_of_playing_next_round","chance_of_playing_this_round","code","corners_and_indirect_freekicks_text","cost_change_event","cost_change_event_fall","cost_change_start","cost_change_start_fall","creativity_rank","creativity_rank_type","direct_freekicks_order","direct_freekicks_text","dreamteam_count"
,"element_type","ep_next","ep_this",""]
merged_df.drop(columns=columns_to_drop, inplace=True)

merged_df.to_csv('player_raw_test.csv', index=False)

