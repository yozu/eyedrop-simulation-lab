"""描画側の開眼採点と物理側の実測指幅を全物理時点で監査する。制御には使わない。"""
import math

OPEN_THRESHOLD = .9
TIME_TOLERANCE_S = 1e-8
PRESSURE_TOLERANCE_KPA = 1e-9
FLOW_TOLERANCE_ML = 1e-12


def audit_pressure_interval(samples, *, start_time, end_time, command_gap_m,
                            press_threshold_m, physics_hz, open_threshold=OPEN_THRESHOLD):
    values = (start_time, end_time, command_gap_m, press_threshold_m, physics_hz, open_threshold)
    if (not all(math.isfinite(v) for v in values) or physics_hz <= 0
            or end_time <= start_time or command_gap_m <= 0 or press_threshold_m <= 0 or not 0<open_threshold<=1):
        raise ValueError('押圧監査の区間・閾値が不正です')
    intervals = (end_time-start_time)*physics_hz
    count = round(intervals)
    if count < 1 or abs(intervals-count) > TIME_TOLERANCE_S*physics_hz or len(samples) != count+1:
        raise ValueError('物理時点の全区間記録がありません')
    closed = commands = measured = 0
    for index, sample in enumerate(samples):
        try:
            t, openness, gap = (sample[key] for key in ('time', 'open', 'gap_m'))
        except KeyError as error:
            raise ValueError('押圧監査の記録項目が欠けています') from error
        if (not all(math.isfinite(v) for v in (t, openness, gap)) or not 0 <= openness <= 1
                or gap <= 0 or abs(t-(start_time+index/physics_hz)) > TIME_TOLERANCE_S):
            raise ValueError('押圧監査の時刻・開眼値・指幅が不正です')
        if openness < open_threshold:
            closed += 1
            commands += int(command_gap_m < press_threshold_m)
            measured += int(gap < press_threshold_m)
    return {'sample_count': len(samples), 'closed_samples': closed,
            'closed_command_samples': commands, 'closed_measured_samples': measured,
            'passed': commands == 0 and measured == 0}


def audit_guarded_interval(samples, actions, motion, *, start_time, end_time,
                           press_threshold_m, physics_hz, open_threshold):
    count = round((end_time-start_time)*physics_hz)
    if (count < 1 or len(actions) != count or len(motion) != count
            or any(len(a) != 8 or not all(math.isfinite(v) for v in a) or a[7] <= 0 for a in actions)):
        raise ValueError('実行した物理区間指令が不正です')
    result = audit_pressure_interval(samples, start_time=start_time, end_time=end_time,
                                     command_gap_m=actions[0][7], press_threshold_m=press_threshold_m,
                                     physics_hz=physics_hz, open_threshold=open_threshold)
    pressure = []
    flow = 0.
    previous = None
    for index, row in enumerate(motion):
        try:
            begin, end = row['start_time'], row['time']
            before, after = row['bottle_before'], row['bottle_after']
            p0, p1 = before['pressure_kpa'], after['pressure_kpa']
            v0, v1 = before['reservoir_ml'], after['reservoir_ml']
        except (KeyError, TypeError) as error:
            raise ValueError('物理記録に圧力または液量がありません') from error
        if (not all(math.isfinite(v) for v in (begin, end, p0, p1, v0, v1))
                or abs(begin-(start_time+index/physics_hz)) > TIME_TOLERANCE_S
                or abs(end-(start_time+(index+1)/physics_hz)) > TIME_TOLERANCE_S
                or v0 < 0 or v1 < 0 or v1-v0 > FLOW_TOLERANCE_ML
                or (previous is not None and (abs(previous[0]-p0)>PRESSURE_TOLERANCE_KPA
                                              or abs(previous[1]-v0)>FLOW_TOLERANCE_ML))):
            raise ValueError('圧力・液量の連続記録が不正です')
        pressure.append(p0)
        if samples[index]['open'] < open_threshold or samples[index+1]['open'] < open_threshold:
            flow += max(0., v0-v1)
        previous = (p1, v1)
    pressure.append(previous[0])
    closed = [i for i, sample in enumerate(samples) if sample['open'] < open_threshold]
    result['closed_command_samples'] = sum(actions[min(i, count-1)][7] < press_threshold_m for i in closed)
    result['closed_positive_pressure_samples'] = sum(pressure[i] > PRESSURE_TOLERANCE_KPA for i in closed)
    result['closed_flow_ml'] = flow
    result['passed'] = (result['closed_command_samples'] == 0 and result['closed_measured_samples'] == 0
                        and result['closed_positive_pressure_samples'] == 0 and flow <= FLOW_TOLERANCE_ML)
    return result
