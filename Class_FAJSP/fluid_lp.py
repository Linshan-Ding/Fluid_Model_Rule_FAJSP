"""
流体网络LP的scipy/HiGHS求解后端。

与class_fajsp.FAJSP.fluid_model中的docplex模型完全等价：
    max  z
    s.t. sum_m x[m,vj] * mu[m,vj] >= z * Q_vj          （完工率约束, 对所有工序类型vj）
         sum_vj x[m,vj] <= 1                            （机器容量约束, 对所有机器m）
         rate_sum(pre) >= rate_sum(cur)                 （流体可行性约束, 仅对初始瞬态数量为0且有前置的工序）
         0 <= x <= 1, z >= 0
docplex的目标 maximize min_vj(rate_sum/Q_vj) 通过引入辅助变量z等价转化。
云端环境无CPLEX时使用本模块, 结果与docplex一致(线性规划最优值唯一)。
"""
import numpy as np
from scipy.optimize import linprog


def solve_fluid_lp(machine_tuple, kind_task_tuple, kind_task_m_dict, machine_pj_dict,
                   process_rate_m_part_dict, fluid_number, fluid_number_time, pre_pj_dict):
    """
    求解流体LP, 返回 {(m,(v,j)): 时间分配比例x} 字典（与docplex的get_value_dict格式一致）。
    """
    # 变量索引: 所有可行的(机器, 工序类型)对 + 末位的吞吐率z
    var_index = {}
    for m in machine_tuple:
        for vj in kind_task_m_dict[m]:
            var_index[(m, vj)] = len(var_index)
    nx = len(var_index)
    z_idx = nx
    n_var = nx + 1

    c = np.zeros(n_var)
    c[z_idx] = -1.0  # linprog做最小化, 最大化z即最小化-z

    A_ub, b_ub = [], []

    # 完工率约束: -sum_m mu*x + Q*z <= 0
    for vj in kind_task_tuple:
        row = np.zeros(n_var)
        for m in machine_pj_dict[vj]:
            row[var_index[(m, vj)]] = -process_rate_m_part_dict[m][vj]
        row[z_idx] = fluid_number[vj]
        A_ub.append(row)
        b_ub.append(0.0)

    # 机器容量约束: sum_vj x <= 1
    for m in machine_tuple:
        row = np.zeros(n_var)
        for vj in kind_task_m_dict[m]:
            row[var_index[(m, vj)]] = 1.0
        A_ub.append(row)
        b_ub.append(1.0)

    # 流体可行性约束: rate_sum(cur) - rate_sum(pre) <= 0
    for VJ in kind_task_tuple:
        if fluid_number_time[VJ] == 0 and pre_pj_dict[VJ]:
            for vj in pre_pj_dict[VJ]:
                row = np.zeros(n_var)
                for m in machine_pj_dict[VJ]:
                    row[var_index[(m, VJ)]] += process_rate_m_part_dict[m][VJ]
                for m in machine_pj_dict[vj]:
                    row[var_index[(m, vj)]] -= process_rate_m_part_dict[m][vj]
                A_ub.append(row)
                b_ub.append(0.0)

    bounds = [(0.0, 1.0)] * nx + [(0.0, None)]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f"流体LP求解失败: {res.message}")

    return {key: float(res.x[i]) for key, i in var_index.items()}
