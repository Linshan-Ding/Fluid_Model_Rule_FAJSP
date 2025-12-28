import csv
from random import randint, seed
import numpy as np
import os
from Class_FAJSP import assemble_method as am


"""实例类"""
class Instance():
    def __init__(self, M_p=None,M_a=None,product_count = None,kind_count = None,J_r = None ,N_p = None):
        seed(0)  # 固定随机种子
        np.random.seed(0)
        self.M_p = M_p
        self.M_a = M_a
        self.process_machine_count = M_p # 加工机器数
        self.assemble_machine_count = M_a # 装配机器数
        self.J_r = J_r  # 每种工件类型的工序数
        self.N_p = N_p  # 每类产品数量
        self.machine_count = self.process_machine_count + self.assemble_machine_count # 机器总数
        self.process_machine_tuple = tuple(m for m in range(self.process_machine_count)) # 机器元组
        self.assemble_machine_tuple = tuple(m for m in range(self.process_machine_count, self.process_machine_count + self.assemble_machine_count)) # 装配机器元组
        self.machine_tuple = tuple(m for m in range(self.machine_count)) # 机器元组
        self.kind_count = kind_count# 工件类型数
        self.kind_r_tuple = tuple(r for r in range(self.kind_count))  # 工件类型元组
        self.product_count = product_count # 产品数
        self.product_tuple = tuple( p for p in range(self.kind_count,self.kind_count+self.product_count)) # 产品元组
        self.kind_tuple = self.kind_r_tuple + self.product_tuple # 工件类型及产品类型元组

        self.file_name = str(M_p) + '_' + str(M_a)+ '_'+ str(self.product_count)+'_'+str(self.kind_count) +'_'+str(self.J_r) +'_'+str(self.N_p) # 算例文件夹名

        (self.task_p_dict, self.kind_task_tuple,self.kind_task_tuple_r, self.kind_task_tuple_a,self.machine_pj_dict, self.time_pjm_dict,self.kind_task_m_dict,self.time_mpj_dict, self.time_pj_dict,
        self.component_pr_dict,self.assemble_schedule_list,self.cost_pj_dict,self.count_p_dict,self.cost_aj_dict,
        self.pre_pj_dict,self.total_cost_dict,self.kind_number,self.cost_count)= self.information()
    # """工件类型数量"""
    # def kind_count(self):
    #     return randint(8,8)
    #
    # """每种工件类型的工序数"""
    # @property
    # def P_r(self) :
    #     return randint(4, 4)
    #
    # """产品类型数量"""
    # def product_num(self):
    #     return randint(4,4)
    #
    # """每类产品数量"""
    # @property
    # def N_p(self):
    #     return randint(2, 2)
    """装配耗材"""
    @property
    def assemble_consumption(self):
        return [randint(1, 2), randint(1, 2)]

    """工序在可选机器上的加工时间"""
    def random_t_pjm(self) :
        return randint(10,20 )

    def information(self):
        """生成所有产品信息"""
        # 每类产品的组件
        component_pr_dict= am.allocation_no_duplicates(self.kind_r_tuple, self.product_tuple)
        #每类产品的装配顺序
        assemble_schedule_list = {p: am.assemble_schedule(len(component_pr_dict[p])) for p in self.product_tuple}
        # 产品和工件对应工序元组
        task_r_dict = {r: tuple(j for j in range(self.J_r)) for r in self.kind_r_tuple}
        task_a_dict = {p: tuple(j for j in range(len(assemble_schedule_list[p]))) for p in self.product_tuple}
        task_p_dict = {**task_a_dict, **task_r_dict}
        # 工序类型元组
        kind_task_tuple_r = tuple((r,j) for r in self.kind_r_tuple for j in task_r_dict[r])
        kind_task_tuple_a = tuple((p,j) for p in self.product_tuple for j in task_a_dict[p])
        kind_task_tuple = tuple((p,j) for p in self.kind_tuple for j in task_p_dict[p])
        # 工序在可选机器 - 使用原生Python整数
        machine_rj_dict = {(r, j): tuple(int(m) for m in np.random.choice(self.process_machine_tuple, randint(1, self.process_machine_count), replace=False)) for (r, j) in kind_task_tuple_r }
        machine_aj_dict = {(a, j): tuple(int(m) for m in np.random.choice(self.assemble_machine_tuple, randint(1, self.assemble_machine_count), replace=False)) for (a, j) in kind_task_tuple_a }
        machine_pj_dict = {**machine_aj_dict,**machine_rj_dict}
        # 工序在可选机器上的加工时间 - 使用原生Python整数
        time_pjm_dict = {(p,j): {int(m): self.random_t_pjm() for m in machine_pj_dict[(p,j)]} for (p,j) in kind_task_tuple}
        # 机器对应工序
        kind_task_m_dict = {m: tuple((p, j) for (p,j) in kind_task_tuple if m in machine_pj_dict[(p, j)]) for m in self.machine_tuple}
        # 机器对应工序的加工时间 - 使用原生Python整数
        time_mpj_dict= {int(m): {(p, j): time_pjm_dict[(p, j)][m] for (p, j) in kind_task_m_dict[m]} for m in self.machine_tuple}
        #各工序加工时间均值
        time_pj_dict = {(p, j): sum([time_pjm_dict[(p, j)][m] for m in machine_pj_dict[(p, j)]]) / len(machine_pj_dict[(p, j)]) for (p, j) in kind_task_tuple }
        #各工序的耗材
        cost_rj_dict = {(r, j): [1] for r in self.kind_r_tuple for j in task_r_dict[r]}
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

        return task_p_dict, kind_task_tuple,kind_task_tuple_r,kind_task_tuple_a,machine_pj_dict,time_pjm_dict,kind_task_m_dict,time_mpj_dict,time_pj_dict,\
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
                        # 确保使用原生Python整数
                        machine_selectable = tuple(int(m) for m in self.machine_pj_dict[(p, j)])
                        time_machine_tuple = tuple(int(self.time_pjm_dict[(p, j)][m]) for m in self.machine_pj_dict[(p, j)])
                        rows.append([p, j, machine_selectable, time_machine_tuple, self.cost_count[(p,j)]])
                else:
                    rows.append([self.component_pr_dict[p] for p in self.product_tuple])
                writer.writerows(rows)
        print("写入完成")

if __name__ == '__main__':
        M_p = [4,8,12]
        M_a = [2,4,6]
        kind_count = [4,8,12]
        product_count = [2,4,6]
        J_r = [2,4,6]
        N_p = [2,4,6]
        data_list = []
        for x in range(3):
            mp = M_p[x]
            ma = M_a[x]
            for y in range(3):
                pc = product_count[y]
                kc = kind_count[y]
                jr = J_r[y]
                for z in range(3):
                    n_p = N_p[z]
                    data_list.append([mp,ma,pc,kc,jr,n_p])
        print(data_list)
        print(len(data_list))
        for data in data_list:
            instance = Instance(M_p=data[0], M_a=data[1],product_count=data[2], kind_count=data[3],  J_r=data[4], N_p=data[5])
            instance.write_file()

        # datalist=[[4, 2, 2, 4, 2, 2], [4, 2, 2, 4, 2, 4], [4, 2, 2, 4, 2, 6], [4, 2, 4, 8, 4, 2], [4, 2, 4, 8, 4, 4],
        #  [4, 2, 4, 8, 4, 6], [4, 2, 6, 12, 6, 2], [4, 2, 6, 12, 6, 4], [4, 2, 6, 12, 6, 6], [8, 4, 2, 4, 2, 2],
        #  [8, 4, 2, 4, 2, 4], [8, 4, 2, 4, 2, 6], [8, 4, 4, 8, 4, 2], [8, 4, 4, 8, 4, 4], [8, 4, 4, 8, 4, 6],
        #  [8, 4, 6, 12, 6, 2], [8, 4, 6, 12, 6, 4], [8, 4, 6, 12, 6, 6], [12, 6, 2, 4, 2, 2], [12, 6, 2, 4, 2, 4],
        #  [12, 6, 2, 4, 2, 6], [12, 6, 4, 8, 4, 2], [12, 6, 4, 8, 4, 4], [12, 6, 4, 8, 4, 6], [12, 6, 6, 12, 6, 2],
        #  [12, 6, 6, 12, 6, 4], [12, 6, 6, 12, 6, 6]]
        # data = datalist[0]
        # instance = Instance(M_p=data[0], M_a=data[1], product_count=data[2], kind_count=data[3], J_r=data[4],
        #                     N_p=data[5])
        # """验证算例信息"""
        # print("\n=== total_cost_dict (各工序总成本) ===")
        # for key, value in list(instance.total_cost_dict.items()):
        #     print(f"  {key}: {value}")

        # 最大工序数： 88
        # 对应数据： [4, 2, 6, 12, 6, 2]
        #获取最大工序数及对应数据
        # job_count_list = []
        # for data in datalist:
        #     instance = Instance(M_p=data[0], M_a=data[1],product_count=data[2], kind_count=data[3],  J_r=data[4], N_p=data[5])
        #     job_count_list.append(len(instance.kind_task_tuple))
        # max_jobs = max(job_count_list)
        # idx = job_count_list.index(max_jobs)
        # print("最大工序数：", max_jobs)
        # print("对应数据：", datalist[idx])
        # def compute_product_component_usage(component_pr_dict, cost_count):
        #     """
        #       计算所有产品对工件索引的消耗数量
        #       :param component_pr_dict: 产品->工件索引列表的字典
        #       :param cost_count: 装配关系字典
        #       :return: dict: {产品ID: [idx(count), ...]} 形式的字典
        #       """
        #     result = {}
        #
        #     for product_id in component_pr_dict:
        #         # 找到该产品的所有工序步骤
        #         steps = [step for (pid, step) in cost_count.keys() if pid == product_id]
        #         if not steps:
        #             result[product_id] = []
        #             continue
        #
        #         max_step = max(steps)
        #         root_node = (product_id, max_step)
        #
        #         # BOM展开函数
        #         def explode(node, mult, usage):
        #             # 基础件判断：节点不在cost_count中，或者前置工序为空
        #             if node not in cost_count or not cost_count[node][0]:
        #                 usage[node] += mult
        #                 return
        #
        #             predecessors, quantities = cost_count[node]
        #             for pred, qty in zip(predecessors, quantities):
        #                 explode(pred, mult * qty, usage)
        #
        #         usage = defaultdict(int)
        #         explode(root_node, 1, usage)
        #
        #         # 获取基础件索引映射 (pid,0) -> pid
        #         base_usage = defaultdict(int)
        #         for (pid, step), count in usage.items():
        #             if step == 0:  # 基础件
        #                 base_usage[pid] += count
        #
        #         # 获取该产品关注的工件索引列表
        #         indices = component_pr_dict[product_id]
        #
        #         # 生成无引号的格式 [idx(count)]，并对索引加1
        #         result[product_id] = [f"{idx + 1}({base_usage.get(idx, 0)})" for idx in indices]
        #
        #     return result
        #
        # """算例信息"""
        # for N in range(5):
        #     print(N)
        #     instance = Instance(M_p, M_a, N)
        #
        #     usage_result = compute_product_component_usage(instance.component_pr_dict, instance.cost_count)
        #
        #     for product_id, usage_list in usage_result.items():
        #         print(f" [", end="")
        #         print(", ".join(usage_list), end="")
        #         print("]")

        # print("=== 实例信息 ===")
        # print("kind_count:", instance.kind_count)
        # print("product_count:", instance.product_count)
        # print("process_machine_count:", instance.process_machine_count)
        # print("assemble_machine_count:", instance.assemble_machine_count)
        # print("machine_count:", instance.machine_count)
        # print("kind_tuple:", instance.kind_tuple)
        #
        # print("\n=== task_p_dict (任务与工序映射) ===")
        # print(instance.task_p_dict)
        #
        # print("\n=== kind_task_tuple (工序元组) ===")
        # print(instance.kind_task_tuple)
        #
        # print("\n=== machine_pj_dict (每个工序可选机器) ===")
        # for key, value in list(instance.machine_pj_dict.items()):  # 只显示前5个
        #     print(f"  {key}: {value}")
        #
        # print("\n=== time_pjm_dict (每个工序在各机器上的时间) ===")
        # for key, value in list(instance.time_pjm_dict.items()):  # 只显示前5个
        #     print(f"  {key}: {value}")
        #
        # print("\n=== time_mpj_dict (每个工序在各机器上的时间) ===")
        # for key, value in list(instance.time_mpj_dict.items()):  # 只显示前3个
        #     print(f"  {key}: {value}")
        #
        # print("\n=== component_pr_dict (每类产品的组件构成) ===")
        # print(instance.component_pr_dict)
        #
        # print("\n=== assemble_schedule_list (装配顺序) ===")
        # print(instance.assemble_schedule_list)
        #
        # print("\n=== cost_pj_dict (各工序成本) ===")
        # for key, value in list(instance.cost_pj_dict.items()):
        #     print(f"  {key}: {value}")
        #
        # print("\n=== count_p_dict (各产品数量) ===")
        # print(instance.count_p_dict)
        #
        # print("\n=== kind_number (工件和产品数量) ===")
        # print(instance.kind_number)
        #
        # print("\n=== pre_pj_dict (各工序的前置工序) ===")
        # print(instance.pre_pj_dict)
        #
        # print("\n=== total_cost_dict (各工序总成本) ===")
        # for key, value in list(instance.total_cost_dict.items()):
        #     print(f"  {key}: {value}")
        #
        # print("\n=== cost_count (前置工序成本) ===")
        # for key, value in list(instance.cost_count.items()):  # 只显示前5个
        #     print(f"  {key}: {value}")
        # print("\n=== time_pj_dict ===")
        # print(instance.time_pj_dict)
        #
