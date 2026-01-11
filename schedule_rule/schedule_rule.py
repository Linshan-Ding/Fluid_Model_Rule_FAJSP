from env.env import FAJSP_Environment
import random
import time

class RULE:
    def __init__(self,data_config):
        self.data_config = data_config
        self.config_str = f"{data_config[0]}_{data_config[1]}_{data_config[2]}_{data_config[3]}_{data_config[4]}_{data_config[5]}"
        M_p, M_a, product_count, kind_count, J_r, N_p = data_config
        self.env = FAJSP_Environment(
            M_p=M_p, M_a=M_a,
            kind_count=kind_count,
            product_count=product_count,
            J_r=J_r, N_p=N_p
        )
        self.step = 0 #计算决策步
        self.queue = []


    def random(self):
        """随机选择动作"""
        random.seed(42) #可复现
        self.env.reset()  # 初始化环境
        start_time =time.time()
        while 1:
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions()  # 获取可选动作
            action = random.choice(candidate_action_list)
            self.env.step(action)
            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step


    def spt(self):
        """最短加工时间优先"""
        self.env.reset() #初始化环境
        start_time =time.time()
        while True :
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions() #获取可选动作
            #获取可选动作时间列表
            time_tuple = ([self.env.time_pjm_dict[(a[0],a[1])][a[2]] for a in candidate_action_list])
            action_time = min(time_tuple)
            action = candidate_action_list [time_tuple.index(action_time)]
            self.env.step(action)

            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step

    def lpt(self):
        """最长加工时间优先"""
        self.env.reset()  # 初始化环境
        start_time = time.time()
        while True:
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions()  # 获取可选动作
            # 获取可选动作时间列表
            time_tuple = ([self.env.time_pjm_dict[(a[0], a[1])][a[2]] for a in candidate_action_list])
            action_time = max(time_tuple)
            action = candidate_action_list[time_tuple.index(action_time)]
            self.env.step(action)
            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step

    def fifo(self):
        """先到先服务：机器空闲时间，工序索引"""
        self.env.reset()
        start_time = time.time()
        while True:
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions()
            selected = min(
                candidate_action_list,
                key=lambda a: (
                    self.env.machine_dict[a[2]].time_end,
                    self.env.kind_task_tuple.index((a[0], a[1]))
                )
            )
            self.env.step(selected)
            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step

    def mrt(self):
        """工序剩余加工时间最长优先"""
        self.env.reset()
        task_time_ave = {task : self.env.time_pj_dict[task] for task in self.env.kind_task_tuple}
        start_time = time.time()
        while True:
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions()
            task_time_remain = []
            for action in candidate_action_list :
                (v,j,m) = action
                remain = 0
                for job in self.env.task_p_dict[v][j:]:
                    remain +=  task_time_ave [(v,job)]
                task_time_remain.append(remain)
            action_idx = task_time_remain.index(max(task_time_remain))  #对于多道相同工序最大值max只返回第一个最大
            action = candidate_action_list[action_idx]
            self.env.step(action)
            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step

    def mrt_spt(self):
        """工序剩余加工时间最长 + 最短处理时间 优先"""
        self.env.reset()
        task_time_ave = {task: self.env.time_pj_dict[task] for task in self.env.kind_task_tuple}
        start_time = time.time()
        while True:
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions()
            task_time_remain = []
            for action in candidate_action_list:
                (v, j, m) = action
                remain = 0
                for job in self.env.task_p_dict[v][j:]:
                    remain +=  task_time_ave[(v, job)]
                task_time_remain.append(remain)
            maxtime = max(task_time_remain)
            maxtime_task_idx = [i for i ,x in enumerate(task_time_remain) if x == maxtime]
            time_dict = {}
            for i in maxtime_task_idx:
                a = candidate_action_list[i]
                time_dict[i] = self.env.time_pjm_dict[(a[0], a[1])][a[2]]
            min_value = min(time_dict.values())
            idx = [i for i,x in time_dict.items() if x == min_value]
            action = candidate_action_list[idx[0]]
            self.env.step(action)
            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step

    def mrt_lpt(self):
        """工序剩余加工时间最长 + 最短处理时间 优先"""
        self.env.reset()
        task_time_ave = {task: self.env.time_pj_dict[task] for task in self.env.kind_task_tuple}
        start_time = time.time()
        while True:
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions()
            task_time_remain = []
            for action in candidate_action_list:
                (v, j, m) = action
                remain = 0
                for job in self.env.task_p_dict[v][j:]:
                    remain +=  task_time_ave[(v, job)]
                task_time_remain.append(remain)
            maxtime = max(task_time_remain)
            maxtime_task_idx = [i for i ,x in enumerate(task_time_remain) if x == maxtime]
            time_dict = {}
            for i in maxtime_task_idx:
                a = candidate_action_list[i]
                time_dict[i] = self.env.time_pjm_dict[(a[0], a[1])][a[2]]
            min_value = max(time_dict.values())
            idx = [i for i,x in time_dict.items() if x == min_value]
            action = candidate_action_list[idx[0]]
            self.env.step(action)
            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step

    def mrt_fifo(self):
        """工序剩余加工时间最长 + 先到先服务 优先"""
        self.env.reset()
        start_time = time.time()
        task_time_ave = {task: self.env.time_pj_dict[task] for task in self.env.kind_task_tuple}
        while True:
            self.step += 1
            candidate_action_list = self.env.get_candidate_actions()
            task_time_remain = []
            for action in candidate_action_list:
                (v, j, m) = action
                remain = 0
                for job in self.env.task_p_dict[v][j:]:
                    remain +=  task_time_ave[(v, job)]
                task_time_remain.append(remain)
            maxtime = max(task_time_remain)
            maxtime_task_idx = [i for i ,x in enumerate(task_time_remain) if x == maxtime]
            action_list = []
            for idx in maxtime_task_idx:
                action_list.append(candidate_action_list[idx])
            selected = min(
                action_list,
                key=lambda a: (
                    self.env.machine_dict[a[2]].time_end,
                    self.env.kind_task_tuple.index((a[0], a[1]))
                )
            )
            self.env.step(selected)
            if self.env.done:
                break
        end_time = time.time()
        run_step = (end_time - start_time)/self.step
        completion_time = self.env.completion_time
        return completion_time , run_step

if __name__ == "__main__":
    # 所有算例配置
    data_list = [
        [4, 2, 2, 4, 2, 2], [4, 2, 2, 4, 2, 4], [4, 2, 2, 4, 2, 6],
        [4, 2, 4, 8, 4, 2], [4, 2, 4, 8, 4, 4], [4, 2, 4, 8, 4, 6],
        [4, 2, 6, 12, 6, 2], [4, 2, 6, 12, 6, 4], [4, 2, 6, 12, 6, 6],
        [8, 4, 2, 4, 2, 2], [8, 4, 2, 4, 2, 4], [8, 4, 2, 4, 2, 6],
        [8, 4, 4, 8, 4, 2], [8, 4, 4, 8, 4, 4], [8, 4, 4, 8, 4, 6],
        [8, 4, 6, 12, 6, 2], [8, 4, 6, 12, 6, 4], [8, 4, 6, 12, 6, 6],
        [12, 6, 2, 4, 2, 2], [12, 6, 2, 4, 2, 4], [12, 6, 2, 4, 2, 6],
        [12, 6, 4, 8, 4, 2], [12, 6, 4, 8, 4, 4], [12, 6, 4, 8, 4, 6],
        [12, 6, 6, 12, 6, 2], [12, 6, 6, 12, 6, 4], [12, 6, 6, 12, 6, 6]
    ]
    for data in data_list:
        rule = RULE(data)
        ct,adt = rule.mrt_fifo()
        print(f"算例{data}完工时间为{ct},平均决策时间为{adt*1000:6.2f}ms")
