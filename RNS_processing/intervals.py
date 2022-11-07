def get_intervals(raw):
    all_intervals = []
    seizure_interval_start = []
    start_inter = 0
    end_inter = 0
    for i in range(len(raw.annotations)):
        if(raw.annotations[i]['description'] == 'eof'):
            end_inter = raw.annotations[i]['onset']
            all_intervals.append([start_inter, end_inter])
            start_inter = end_inter
        elif(raw.annotations[i]['description'] == 'sz_on'):
            seizure_interval_start.append(start_inter)

    if(all_intervals[-1][1] != raw.times[-1]):
        all_intervals.append([start_inter, raw.times[-1]])


    # From all_intervals, get intervals corresponding to seizures, and intervals with no seizure activity
    non_sz_intervals = []
    sz_intervals = []
    for interval in all_intervals:
        if(interval[0] not in seizure_interval_start):
            non_sz_intervals.append(interval)
        else:
            sz_intervals.append(interval)
    # sz_intervals and non_sz_intervals hold this

    return all_intervals, seizure_interval_start, non_sz_intervals, sz_intervals