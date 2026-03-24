track_history = {}
entered_ids = set()
exited_ids = set()

def count_entries_exits(id, cx, cy, line_y):
    global track_history, entered_ids, exited_ids

    if id not in track_history:
        track_history[id] = []

    track_history[id].append((cx, cy))

    if len(track_history[id]) > 2:
        track_history[id].pop(0)

    if len(track_history[id]) == 2:
        prev_y = track_history[id][0][1]
        curr_y = track_history[id][1][1]

        if prev_y < line_y and curr_y >= line_y:
            entered_ids.add(id)
        elif prev_y > line_y and curr_y <= line_y:
            exited_ids.add(id)

    return len(entered_ids), len(exited_ids)