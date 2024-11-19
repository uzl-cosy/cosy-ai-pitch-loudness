import numpy as np


def multiple_appends(listname, *element):
    listname.extend(element)


def compute_power_db(x, fs, win_len_sec=0.1, power_ref=10**(-12)):
    window_size = fs / 4  # Size of each window in samples
    hop_length = 512  # Hop size between windows in samples

    # Calculate the number of windows
    num_windows = (len(x) - window_size) // hop_length + 1

    power_db = []
    for i in range(int(num_windows)):
        eps = 0.001

        start_sample = int(i * hop_length)
        end_sample = int(start_sample + window_size)

        window_of_x = x[start_sample:end_sample]
        rms_amplitude = np.sqrt(np.mean(np.square(window_of_x)))

        if rms_amplitude == 0.0:
            rms_amplitude = rms_amplitude + eps
	
        rms_db = 20 * np.log10(rms_amplitude) 
	
        power_db.append(np.around(rms_db, decimals=1))

    return power_db


def compute_db_statistics(audioIN_DB_chunks):
    db_statistics_overall = []
    db_mean = np.mean(list(map(np.mean, audioIN_DB_chunks)))
    db_max = np.max(list(map(np.max, audioIN_DB_chunks)))
    db_min = np.min(list(map(np.min, audioIN_DB_chunks)))
    multiple_appends(db_statistics_overall, db_mean, db_max, db_min)
    return db_statistics_overall
