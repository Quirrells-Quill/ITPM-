import time

entry_time = {}

def update_dwell_time(id):
    if id not in entry_time:
        entry_time[id] = time.time()

    return time.time() - entry_time[id]