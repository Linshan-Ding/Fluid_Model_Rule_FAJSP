
import csv
from random import randint
import numpy as np
import os
import assemble_method as am

"""
代办事项:
工件之间可以交叉混用，为每个产品设置组件构成
基于组件构成重构装配工序
优化并行装配工序的处理
"""


"""实例类"""
class Instance():
    def __init__(self, M_p,M_a,N):
        self.process_machine_count = M_p # 加工机器数
        self.assemble_machine_count = M_a # 装配机器数
        self.machine_count = self.process_machine_count + self.assemble_machine_count # 机器总数
        self.process_machine_tuple = tuple(m for m in range(self.process_machine_count)) # 机器元组
        self.assemble_machine_tuple = tuple(m for m in range(self.process_machine_count, self.process_machine_count + self.assemble_machine_count)) # 装配机器元组
        self.machine_tuple = tuple(m for m in range(self.machine_count)) # 机器元组
        self.kind_count = self.kind_count() # 工件类型数
        self.kind_r_tuple = tuple(r for r in range(self.kind_count))  # 工件类型元组
        self.product_count = self.product_num() # 产品数
        self.product_tuple = tuple( p for p in range(self.kind_count,self.kind_count+self.product_count)) # 产品元组
        self.kind_tuple = self.kind_r_tuple + self.product_tuple # 工件类型及产品类型元组

        self.file_name = 'machine_' + str(M_p) + '_' + str(M_a) + '_' +str(N)  # 算例文件夹名

        (self.task_p_dict, self.kind_task_tuple, self.machine_pj_dict, self.time_pjm_dict,self.kind_task_m_dict,self.time_mpj_dict, self.time_pj_dict,
        self.component_pr_dict,self.assemble_schedule_list,self.cost_pj_dict,self.count_p_dict,self.cost_aj_dict,
        self.pre_pj_dict,self.total_cost_dict,self.kind_number,self.cost_count)= self.information()


    """工件类型数量"""
    def kind_count(self):
        return randint(9, 12)

    """每种工件类型的工序数"""
    @property
    def P_r(self) :
        return randint(3, 5)

    """产品类型数量"""
    def product_num(self):
        return randint(4, 6)

    """每类产品数量"""
    @property
    def N_p(self):
        return randint(5, 50)

    """装配耗材"""
    @property
    def assemble_consumption(self):
        return [randint(1, 3), randint(1, 3)]

    """工序在可选机器上的加工时间"""
    def random_t_pjm(self) :
        return randint(40,400 )

    def information(self):
        """生成所有产品信息"""
        # 每类产品的组件
        component_pr_dict= am.allocation_no_duplicates(self.kind_r_tuple, self.product_tuple)
        #每类产品的装配顺序
        assemble_schedule_list = {p: am.assemble_schedule(len(component_pr_dict[p])) for p in self.product_tuple}
        # 产品和工件对应工序元组
        task_r_dict = {r: tuple(j for j in range(self.P_r)) for r in self.kind_r_tuple}
        task_a_dict = {p: tuple(j for j in range(len(assemble_schedule_list[p]))) for p in self.product_tuple}
        task_p_dict = {**task_a_dict, **task_r_dict}
        # 工序类型元组
        kind_task_tuple_r = tuple((r,j) for r in self.kind_r_tuple for j in task_r_dict[r])
        kind_task_tuple_a = tuple((p,j) for p in self.product_tuple for j in task_a_dict[p])
        kind_task_tuple = tuple((p,j) for p in self.kind_tuple for j in task_p_dict[p])
        # 工序在可选机器
        machine_rj_dict = {(r, j): tuple(np.random.choice(self.process_machine_tuple, randint(1, self.process_machine_count), replace=False)) for (r, j) in kind_task_tuple_r }
        machine_aj_dict = {(a, j): tuple(np.random.choice(self.assemble_machine_tuple, randint(1, self.assemble_machine_count), replace=False)) for (a, j) in kind_task_tuple_a }
        machine_pj_dict = {**machine_aj_dict,**machine_rj_dict}
        # 工序在可选机器上的加工时间
        time_pjm_dict = {(p,j): {m: self.random_t_pjm() for m in machine_pj_dict[(p,j)]} for (p,j) in kind_task_tuple}
        # 机器对应工序
        kind_task_m_dict = {m: tuple((p, j) for (p,j) in kind_task_tuple if m in machine_pj_dict[(p, j)]) for m in self.machine_tuple}
        # 机器对应工序的加工时间
        time_mpj_dict= {m: {(p, j): time_pjm_dict[(p, j)][m] for (p, j) in kind_task_m_dict[m]} for m in self.machine_tuple}
        #各工序加工时间均值
        time_pj_dict = {(p, j): sum([time_pjm_dict[(p, j)][m] for m in machine_pj_dict[(p, j)]]) / len(machine_pj_dict[(p, j)]) for (p, j) in kind_task_tuple }
        #各工序的耗材
        cost_rj_dict = {(r, j): 1 for r in self.kind_r_tuple for j in task_r_dict[r]}
        cost_aj_dict = {(p, j): self.assemble_consumption  for p in self.product_tuple for j in task_a_dict[p]}
        cost_pj_dict = {**cost_rj_dict, **cost_aj_dict}

        #创建o_pj的前置工序
        pre_rj_dict = {(r, j): [(r, j - 1)] if j > 0 else [] for r in self.kind_r_tuple for j in task_r_dict[r]}
        pre_aj_dict = {}
        for (p, j) in kind_task_tuple_a:
            if assemble_schedule_list[p][1] != 2:
                if j == 0:
                    pre_aj_dict[(p, j)] = [(component_pr_dict[p][j],max(task_p_dict[component_pr_dict[p][j]])), (component_pr_dict[p][j+1],max(task_p_dict[component_pr_dict[p][j+1]]))]
                else:
                    pre_aj_dict[(p, j)] = [(p, j - 1),(component_pr_dict[p][j+1],max(task_p_dict[component_pr_dict[p][j+1]]))]
            else: #暴力处理并行装配工序
                pre_aj_dict[(p, 0)] = [(component_pr_dict[p][0], max(task_p_dict[component_pr_dict[p][0]])), (component_pr_dict[p][1], max(task_p_dict[component_pr_dict[p][1]]))]
                pre_aj_dict[(p, 1)] = [(component_pr_dict[p][2], max(task_p_dict[component_pr_dict[p][2]])), (component_pr_dict[p][3], max(task_p_dict[component_pr_dict[p][3]]))]
                pre_aj_dict[(p, 2)] = [(p, 0), (p, 1)]
        pre_pj_dict = {**pre_rj_dict, **pre_aj_dict}

        #产品类型数量
        count_p_dict = {p: self.N_p for p in self.product_tuple}
        # 各工序总耗材
        product_process_paths ={p :[(p,j) for j in task_p_dict[p]] for p in self.product_tuple}
        total_cost_dict = am.calculate_total_cost(pre_pj_dict, cost_pj_dict, count_p_dict, product_process_paths)
        #工件及产品数量
        kind_number = tuple(total_cost_dict[(r,0)] for r in self.kind_r_tuple)+ tuple(count_p_dict.values())
        cost_count = {current_pj:[predecessors,cost_pj_dict[current_pj]] for current_pj, predecessors in pre_pj_dict.items()}

        return task_p_dict, kind_task_tuple,machine_pj_dict,time_pjm_dict,kind_task_m_dict,time_mpj_dict,time_pj_dict,\
            component_pr_dict,assemble_schedule_list,cost_pj_dict,count_p_dict,cost_aj_dict,\
            pre_pj_dict, total_cost_dict,kind_number,cost_count


    def write_file(self):
        """写入CSV文件"""
        os.makedirs(os.path.join('../data', self.file_name), exist_ok=True)  # 新建实例文件夹
        file_csv = {'based_data.csv': ['kind_count', 'process_machine_count','assemble_machine_count', 'product_count','kind_number'],
                    'process_data.csv': ['kind', 'task', 'machine_selectable', 'process_time', 'cost_count'],
                    'assemble_data.csv': [f'product{p}_component' for p in self.product_tuple]}

        for csv_name, header in file_csv.items():
            data_file = os.path.join('../data', self.file_name, csv_name)
            with open(data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                rows = []  # 初始化写入数据
                if csv_name == 'based_data.csv':
                    rows.append([self.kind_count, self.process_machine_count,self.assemble_machine_count, self.product_count,self.kind_number])
                elif csv_name == 'process_data.csv':
                    for (p, j) in self.kind_task_tuple:
                        time_machine_tuple = tuple(self.time_pjm_dict[(p, j)][m] for m in self.machine_pj_dict[(p, j)])
                        rows.append([p, j, self.machine_pj_dict[(p, j)], time_machine_tuple,self.cost_count[(p,j)]])
                else:
                    rows.append([self.component_pr_dict[p]for p in self.product_tuple])
                writer.writerows(rows)
        print("写入完成")

if __name__ == '__main__':
        M_p = 10
        M_a = 5
        """写入文件"""
        num= [i for i in range(10)]
        for N in num:
            instance = Instance(M_p, M_a, N)
            instance.write_file()
        """算例信息"""
        # instance = Instance(M_p, M_a,0)
        #
        # print("=== 实例信息 ===")
        # print("kind_count:", instance.kind_count)
        # print("product_count:", instance.product_count)
        # print("process_machine_count:", instance.process_machine_count)
        # print("assemble_machine_count:", instance.assemble_machine_count)
        # print("machine_count:", instance.machine_count)
        # print("kind_tuple:", instance.kind_tuple)
        #
        # print("\n=== task_p_dict (任务与工序映射) ===")
        # print(instance.task_p_dict)  # 例如：{'p0': (0, 1, 2), 'p1': (0, 1, 2), ...} 或者类似
        #
        # print("\n=== kind_task_tuple (工序元组) ===")
        # print(instance.kind_task_tuple)  # 如：(('r0', 0), ('r0', 1), ..., ('p0', 0), ...)
        #
        # print("\n=== machine_pj_dict (每个工序可选机器) ===")
        # print(instance.machine_pj_dict)
        #
        # print("\n=== time_pjm_dict (每个工序在各机器上的时间) ===")
        # print(instance.time_pjm_dict )  # 示例
        #
        # print("\n=== time_mpj_dict (每个工序在各机器上的时间) ===")
        # print(instance.time_mpj_dict)  # 示例
        #
        # print("\n=== component_pr_dict (每类产品的组件构成) ===")
        # print(instance.component_pr_dict)  # 如：{'p0': (0, 1, 2), 'p1': (1, 3), ...}
        #
        # print("\n=== assemble_schedule_list (装配顺序) ===")
        # print(instance.assemble_schedule_list)  # 装配流程列表
        #
        # print("\n=== cost_pj_dict (各工序成本) ===")
        # print(instance.cost_pj_dict)
        #
        # print("\n=== count_p_dict (各产品数量) ===")
        # print(instance.count_p_dict)
        #
        #
        # print("\n=== kind_number (工件和产品数量) ===")
        # print(instance.kind_number)
        #
        # print("\n=== pre_pj_dict (各工序的前置工序) ===")
        # print(instance.pre_pj_dict)
        #
        # print("\n=== total_cost_dict (各工序总成本) ===")
        # print(instance.total_cost_dict)
        #
        # print("\n=== cost_count (前置工序成本) ===")
        # print(instance.cost_count)
