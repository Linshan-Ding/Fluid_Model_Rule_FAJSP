import numpy as np
from Class_FAJSP.class_fajsp import FAJSP


"""
FAJSP 环境配置
1.读取FAJSP实例数据
2.定义环境参数
3.初始化环境,reset()函数
4.step()函数
"""

class FAJSP_Environment(FAJSP):
    def __init__(self,**kwargs):
        super().__init__( **kwargs)
        self.step_count = 0 #决策步
        self.last_step_time = 0
        self.step_time = 0 #决策时间点
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        self.completion_time = 0  # 当前时间点完工时间
        self.completion_time_last = 0  # 上一时间步完工时间
        #状态，动作属性
        self.current_state = None # 当前状态
        self.next_state = None  # 下一步状态
        self.actions = ((p, j, m) for m in self.machine_tuple for (p, j) in self.kind_task_tuple if m in self.kind_task_m_dict[(p, j)])  # 动作空间
        self.reward = 0 #累计回报
        self.reward_sum =None
        self.fluid_solution = None
        #记录甘特图
        self.schedule_log = []

    def reset(self):
        """环境重置状态"""
        # 清理之前创建的工件对象，嘻嘻不知道哪里重复初始化了，不然总工序数多一倍
        self.task_dict.clear()
        for p in self.part_dict.keys():
            self.part_dict[p].part_now_list.clear()
            self.part_dict[p].part_unselected_list.clear()
            self.part_dict[p].part_unprocessed_list.clear()
            self.part_dict[p].part_processed_list.clear()
            self.part_dict[p].buffer = 0
            self.part_dict[p].count = 0

        self.reset_parameter() #初始化参数对象列表和字典
        self.reset_object()
        self.fluid_solution = self.fluid_model()
        self.update_fluid_parameter(self.fluid_solution)
        self.reward_sum = 0  # 累计回报
        # 重置时间步和当前时间点
        self.step_count = 0
        self.last_step_time = 0
        self.step_time = 0
        self.completion_time = 0  # 当前时间点完工时间
        self.completion_time_last = 0  # 上一时间步完工时间
        #重置状态属性
        self.current_state = self.state_extract()  # 当前状态
        self.next_state = None  # 下一步状态
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        #记录甘特图
        self.schedule_log = []

    def state_extract(self):
        """状态提取函数"""
        #初始化状态矩阵
        proc_rate_matrix = np.zeros((self.machine_count, len(self.kind_task_tuple))) # 实际加工速率矩阵
        job_flow_matrix = np.zeros((len(self.kind_task_tuple), len(self.kind_task_tuple)))  # 物料工序流矩阵
        fluid_rate_matrix =  np.zeros((self.machine_count, len(self.kind_task_tuple)))# 流体加工速率矩阵
        pend_job_matrix = np.zeros((3, len(self.kind_task_tuple)))  # 待加工物料矩阵
        # #统一矩阵维度88*88
        # proc_rate_matrix = np.zeros((88,88))
        # job_flow_matrix = np.zeros((88,88))
        # fluid_rate_matrix = np.zeros((88,88))
        # pend_job_matrix = np.zeros((88,88))
        #静态特征
        for m in self.process_machine_tuple:
            for (p, j) in self.time_mpj_dict[m].keys():
                proc_rate_matrix[m][self.kind_task_tuple.index((p, j))] = 1/self.time_mpj_dict[m][(p, j)]

        for j_pre in self.kind_task_tuple:
            for j_cur in self.kind_task_tuple:
                if j_pre in self.pre_pj_dict[j_cur]:
                    idx_j_pre = self.kind_task_tuple.index(j_pre)
                    idx_j_cur = self.kind_task_tuple.index(j_cur)
                    idx_j_pre_cost = self.pre_pj_dict[j_cur].index(j_pre)
                    job_flow_matrix[idx_j_pre][idx_j_cur] = self.cost_pj_dict[j_cur][idx_j_pre_cost ]

        # 基于流体模型在线计算的动态特征
        for (m,(p,j)),rate in self.fluid_solution.items():
            idx = self.kind_task_tuple.index((p, j))
            fluid_rate_matrix[m][idx] = rate

        # 待处理物料矩阵
        for (p, j) in self.kind_task_tuple:
            idx = self.kind_task_tuple.index((p, j))
            part_object = self.part_dict[idx]

            pend_job_matrix[0][idx] = self.total_cost_dict[(p, j)]  # 初始各工序待加工数量
            pend_job_matrix[1][idx] = len(part_object.part_unprocessed_list)  # 当前未处理工序数
            pend_job_matrix[2][idx] = part_object.fluid_unprocessed_number  # 流体模型中未处理数量
        # print(pend_job_matrix)
        #
        # proc_rate_matrix[:proc_rate_matrix_old.shape[0], :proc_rate_matrix_old.shape[1]] = proc_rate_matrix_old
        # job_flow_matrix[:job_flow_matrix_old.shape[0], :job_flow_matrix_old.shape[1]] = job_flow_matrix_old
        # fluid_rate_matrix[:fluid_rate_matrix_old.shape[0], :fluid_rate_matrix_old.shape[1]] = fluid_rate_matrix_old
        # pend_job_matrix[:pend_job_matrix_old.shape[0], :pend_job_matrix_old.shape[1]] = pend_job_matrix_old

        return [proc_rate_matrix, job_flow_matrix, fluid_rate_matrix, pend_job_matrix]


    """
    代办,根据count选候选动作
    """
    def get_candidate_actions(self):
        """构建当前决策点的候选动作集"""
        candidate_actions = []
        for p,part_object in self.part_dict.items():
            (v,j) = self.kind_task_tuple[p]
            # 检查1: 必须有待处理工件（关键修复）
            if not part_object.part_unprocessed_list:
                continue  # 如果没有待处理工件，直接跳过
            pre_vj  = [pre_vj for pre_vj in self.pre_pj_dict[(v,j)]]    #该工序的前置工序
            if not pre_vj:
                for m in self.machine_pj_dict[(v,j)]:
                    if len(part_object.part_now_list) >= 1 and self.machine_dict[m].state == 0:
                        candidate_actions.append((v,j,m))
            if len(pre_vj) == 1:
                idx = self.kind_task_tuple.index(pre_vj[0])
                for m in self.machine_pj_dict[(v,j)]:
                    if self.part_dict[idx].count >= 1 and self.machine_dict[m].state == 0:
                        candidate_actions.append((v,j,m))
            if len(pre_vj) == 2:
                pre_idx_0 = self.pre_pj_dict[(v,j)].index(pre_vj[0])
                pre_idx_1 = self.pre_pj_dict[(v,j)].index(pre_vj[1])
                idx_0 = self.kind_task_tuple.index(pre_vj[0])
                idx_1 = self.kind_task_tuple.index(pre_vj[1])
                for m in self.machine_pj_dict[(v, j)]:
                    if self.part_dict[idx_0].count >= self.cost_count[(v,j)][1][pre_idx_0] and self.part_dict[idx_1].count >= self.cost_count[(v,j)][1][pre_idx_1] and self.machine_dict[m].state == 0:
                        candidate_actions.append((v, j, m))
        return candidate_actions


    def action_features(self,action):
        """构建动作特征向量"""
        vj = (action[0], action[1])  # 选择的工序
        m = action[2]  # 选择的机器
        [PR_vjm,MFN_jj,FPR_vjm,PM_vj] = self.state_extract()
        idx = self.kind_task_tuple.index((vj))
        rate_vjm = [PR_vjm[m][idx]]
        fluid_rate_vjm = [FPR_vjm[m][idx]]
        flow_matrix = [i for i in MFN_jj[:,idx]]
        pend_matrix = [i for i in PM_vj[:,idx]]
        feature_vector = {action: rate_vjm+fluid_rate_vjm+flow_matrix+pend_matrix}
        # print(feature_vector)
        return feature_vector

    def step(self,action):
        """环境执行一步动作"""
        #执行动作 action = (p,j,m))
        (v,j,m) = action
        vj_selected = (v,j)
        m_selected = m
        p = self.kind_task_tuple.index(vj_selected)
        kind_task_object_selected = self.part_dict[p]  # 工序对象
        part_object_selected = kind_task_object_selected.part_unprocessed_list[0] #待加工工件
        machine_object_selected = self.machine_dict[m_selected]  # 机器对象
        #更新工件对象属性
        part_object_selected.time_begin = self.step_time #工序开工时间
        part_object_selected.machine = m_selected #选择的机器
        part_object_selected.time_end = self.step_time + self.time_mpj_dict[m_selected][vj_selected]  # 完工时间
        kind_task_object_selected.part_unprocessed_list.remove(part_object_selected) #更新未分配机器的工件对象列表
        #甘特图日志
        self.schedule_log.append({
            "number":part_object_selected.number,
            "job": v,
            "op": j,
            "machine": m_selected,
            "start": part_object_selected.time_begin,
            "end": part_object_selected.time_end
        })
        #更新count属性和part_now_list属性
        if not self.pre_pj_dict[(v, j)]:  # 前置工序列表为空
            if kind_task_object_selected.part_now_list:
                del kind_task_object_selected.part_now_list[0]
        else:  # 有前置工序
            for pre_job in self.pre_pj_dict[(v, j)]:
                if pre_job:  # 如果前置工序存在
                    pre_job_idx = self.kind_task_tuple.index(pre_job)
                    pre_idx = self.pre_pj_dict[(v, j)].index(pre_job)
                    self.part_dict[pre_job_idx].count -= self.cost_count[(v, j)][1][pre_idx]
        #更新机器对象
        machine_object_selected.state = 1  # 更新机器状态
        machine_object_selected.time_end = part_object_selected.time_end  # 更新机器完工时间
        machine_object_selected.part_list.append(part_object_selected)  # 更新机器已加工工序对象列表
        machine_object_selected.job_object = part_object_selected  # 更新机器正在加工的工件对象
        machine_object_selected.unprocessed_part_dict[vj_selected] -= 1  # 更新该机器未加工工序pj的数量，这个属性由流体模型确定，后续再更新流体模型参数就会影响这个值，后续手动更新流体模型参数
        self.completion_time = max(self.completion_time, part_object_selected.time_end)
        #判断是否移动时间钟
        while len(self.get_candidate_actions()) == 0:
            time_point_list = [self.machine_dict[m].time_end for m in self.machine_tuple
                               if self.machine_dict[m].time_end > self.step_time]
            self.step_time = min(time_point_list)
            # print(self.step_time)
            #更新工件对象的缓冲区容量，完成当前工件则count+1
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end == self.step_time:
                    job_object = machine_object.job_object
                    p=job_object.part #工件索引
                    machine_object.state = 0
                    self.part_dict[p].count += 1     #一个工序完成计数+1
            # 更新机器状态和正在加工的工件对象
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end <= self.step_time:
                    machine_object.state = 0  # 更新机器状态
            #更新流体属性
            for p, part_object in self.part_dict.items():
                (v,j) = self.kind_task_tuple[p]
                part_object.fluid_unprocessed_number = max(0,self.total_cost_dict[(v, j)] - part_object.fluid_rate_sum * self.step_time)
            # 判断是否终止
            if sum(len(part_object.part_unprocessed_list) for p, part_object in self.part_dict.items()) == 0:
                self.done = True
                break
        # 提取新的状态、计算回报值、判断周期循环是否结束
        # 更新流体更新时间
        self.step_count += 1
        self.last_step_time = self.step_time
        self.reward = self.compute_reward()  # 即时奖励
        self.reward_sum += self.reward  # 更新累计回报
        self.completion_time_last = self.completion_time
        self.current_state = self.state_extract()  # 提取当前状态

        return self.current_state, self.reward, self.done



    def compute_reward(self):
        """计算即时奖励"""
        return -(self.completion_time - self.completion_time_last) / self.fluid_completed_time
        # return -(self.completion_time - self.completion_time_last)
