# -*- coding: utf-8 -*-
"""
五条优先调度规则(PDR), 与schedule_rule/schedule_rule.py中的实现逻辑一致,
但直接作用于任意已构造的环境对象(支持OOD算例的额外生成参数)。
规则集合与论文6.5节一致: SPT / FIFO / MRT / MRT_SPT / MRT_FIFO。
"""
import time


def _run(env, pick):
    """通用调度循环: pick(env, candidate_actions) -> action"""
    env.reset()
    steps = 0
    start = time.time()
    while True:
        steps += 1
        candidates = env.get_candidate_actions()
        action = pick(env, candidates)
        env.step(action)
        if env.done:
            break
    elapsed = time.time() - start
    return env.completion_time, elapsed / steps


def spt(env):
    """最短加工时间优先"""
    def pick(env, cands):
        times = [env.time_pjm_dict[(a[0], a[1])][a[2]] for a in cands]
        return cands[times.index(min(times))]
    return _run(env, pick)


def fifo(env):
    """先到先服务: 机器空闲时间, 工序索引"""
    def pick(env, cands):
        return min(cands, key=lambda a: (env.machine_dict[a[2]].time_end,
                                         env.kind_task_tuple.index((a[0], a[1]))))
    return _run(env, pick)


def _remaining_times(env, cands, task_time_ave):
    remain_list = []
    for (v, j, m) in cands:
        remain = 0
        for job in env.task_p_dict[v][j:]:
            remain += task_time_ave[(v, job)]
        remain_list.append(remain)
    return remain_list


def mrt(env):
    """工序剩余加工时间最长优先"""
    task_time_ave = None

    def pick(env, cands):
        nonlocal task_time_ave
        if task_time_ave is None:
            task_time_ave = {task: env.time_pj_dict[task] for task in env.kind_task_tuple}
        remain_list = _remaining_times(env, cands, task_time_ave)
        return cands[remain_list.index(max(remain_list))]
    return _run(env, pick)


def mrt_spt(env):
    """剩余加工时间最长 + 最短处理时间破平"""
    task_time_ave = None

    def pick(env, cands):
        nonlocal task_time_ave
        if task_time_ave is None:
            task_time_ave = {task: env.time_pj_dict[task] for task in env.kind_task_tuple}
        remain_list = _remaining_times(env, cands, task_time_ave)
        maxtime = max(remain_list)
        tied = [i for i, x in enumerate(remain_list) if x == maxtime]
        best_i = min(tied, key=lambda i: env.time_pjm_dict[(cands[i][0], cands[i][1])][cands[i][2]])
        return cands[best_i]
    return _run(env, pick)


def mrt_fifo(env):
    """剩余加工时间最长 + 先到先服务破平"""
    task_time_ave = None

    def pick(env, cands):
        nonlocal task_time_ave
        if task_time_ave is None:
            task_time_ave = {task: env.time_pj_dict[task] for task in env.kind_task_tuple}
        remain_list = _remaining_times(env, cands, task_time_ave)
        maxtime = max(remain_list)
        tied = [cands[i] for i, x in enumerate(remain_list) if x == maxtime]
        return min(tied, key=lambda a: (env.machine_dict[a[2]].time_end,
                                        env.kind_task_tuple.index((a[0], a[1]))))
    return _run(env, pick)


PDR_RULES = {'SPT': spt, 'FIFO': fifo, 'MRT': mrt, 'MRT_SPT': mrt_spt, 'MRT_FIFO': mrt_fifo}
