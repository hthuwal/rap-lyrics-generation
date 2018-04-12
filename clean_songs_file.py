import os
songs_dir = 'top_9/'
save_cleaned_dir = 'top_9_cleaned/'
def get_songs_list(path):
    # result = [os.path.join(path, s) for s in os.listdir(path)]
    result = os.listdir(path)
    return result
def clear_song_file(filename, input_dir, output_dir):
    """
    Clean 
        Artist: Scarface
        Album:  The Diary
        Song:   The White Sheet
        Typed by: OHHLA.com
    """
    with open(os.path.join(input_dir, filename), 'r') as f:
        data = f.readlines()
        
    is_meta_row = lambda r : r.startswith('artist:') or r.startswith('album:') or r.startswith('song:') or r.startswith('typed by:')
    data = [r for r in data if not is_meta_row(r.lower())]
    
    if data[0] == '\n':
        data.pop(0)
    
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.writelines(data)

songs = get_songs_list(songs_dir)
for i, s in enumerate(songs):
    clear_song_file(s, songs_dir, save_cleaned_dir)
    print(i, s)
print('All done!')

