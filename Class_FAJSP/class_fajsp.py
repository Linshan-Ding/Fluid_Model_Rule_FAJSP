from docplex.mp.model import Model
from .Instance_generate import Instance
# from .medium_scale_Instance_generate import Instance
# from .large_scale_Instance_generate import Instance
"""
把每一道工序当成一个工件
"""
class Kind():
    """源工件类型类+终端工件类型类"""
    def __init__(self,v):
        self.kind =v #源工件对象，终端工件对象


"""
把每一道工序当成一个工件
"""
class Part():
    """
    定义部件类相当于定义工序o_vj类
    p是o_vj的索引
    """
    def __init__(self,p):
        self.part = p
        self.part_unprocess_list = []
        #基本属性
        self.task = p #索引
        self.machine_tuple = None  # 可选加工机器编号
        self.time_remain = None  # 后续剩余处理时间
        # 附加属性
        self.buffer = None #该工序段已经完成的工序的缓冲区容量,选择才减少
        self.count = None #该工序段已经完成的工序的容量，分配了才减少，主要针对交叉混用的装配工序
        self.part_now_list = []  # 处于该工序段的工件对象列表
        self.part_unprocessed_list = []  # 该工序段未被加工的工件对象列表同时也是未被加工的“工序”对象列表
        self.part_unselected_list = []  #该工序段可转移到part_now_list的对象
        self.part_processed_list = []  # 该工序段已加工的工序对象列表同时也是已加工的“工序“对象列表
        # 流体相关属性
        self.fluid_process_rate_m_dict = {}  # 流体中被各机器加工的速率
        self.fluid_machine_list = []  # 流体模型解可选加工机器
        self.fluid_time_sum = None  # 流体模型中该工序的加工时间
        self.fluid_rate_sum = None  # 流体模型中加工该工序的速率
        self.fluid_number = None  # 处于该工序段的流体数量
        self.fluid_unprocessed_number = None  # 未被加工的流体数



class Job(Part):
    """工件类"""
    def __init__(self, p, n):   #后续动态进行调整终端工件
        Part.__init__(self, p)  # 调用父类的构函
        # 基本属性
        self.number = n  # 该工件类型的第几个工件或者工件编号
        self.time_arrive = None  # 该工件的到达时间
        # 附加属性
        self.part_list = []  # 分配机器的工件对象列表
        self.part_unprocessed_list = []  # 未分配机器的“工序”对象列表或者未分配机器的工件对象列表

class Task(Job):
    """工序类"""
    def __init__(self, p, n):
        Job.__init__(self, p, n)  # 调用父类构函
        # 附加属性
        self.machine = None  # 选择的机器
        self.time_end = None  # 加工结束时间
        self.time_begin = None  # 加工开始时间
        """加工耗时"""
        @property
        def time_cost(self):
            return self.time_end - self.time_begin

class Machine():
    """机器类"""
    def __init__(self, m):
        # 基本属性
        self.machine_node = m  # 机器编号
        self.kind_task_tuple = None  # 可选加工工序类型元组
        # 附加属性
        self.state = 0  # 机器状态
        self.time_end = 0  # 机器完工时间
        self.part_list = []  # 机器已加工“工序”对象列表 机器已加工工件对象列表
        self.part_object = None  # 机器正在处理的工件对象
        self.unprocessed_part_dict = {}  # 未被m加工的各工件类型的工件总数/随着加工过程动态更新
        # 流体附加属性
        self.fluid_part_list = []  # 可选加工工件类型
        self.time_ratio_part_dict = {}  # 流体解中分配给各工序类型的时间比例
        self.fluid_process_rate_part_dict = {}  # 流体解中加工各工件类型的速率
        self.fluid_unprocessed_part_dict = {}  # 未被机器m加工的各工件类型流体总数

class FAJSP(Instance):
    """柔性装配作业车间调度类"""
    def __init__(self, **kwargs):
        Instance.__init__(self, **kwargs)  # 调用父类的构函
        # 实例化工件类型、工件、工序类型、工序和机器对象字
        self.fluid_completed_time = None #流体完工时间
        self.part_dict = {p:Part(p) for p in range(len(self.kind_task_tuple))} #工件（工序）对象列表（索引值表示）
        self.machine_dict = {m: Machine(m) for m in self.machine_tuple}  # 机器对象字典
        self.task_dict = {}  # (r,n)  # 工件对象字典
        self.process_rate_m_part_dict = {
            m: {(v, j): 1 / self.time_mpj_dict[m][(v, j)] for (v, j) in self.kind_task_m_dict[m]} for m in
            self.machine_tuple}  # 机器加工流体速率
        self.reset_parameter()
        # self.reset_object()
        # print("成功定义FAJSP类")

    def reset_parameter(self):
        """初始化各字典和参数"""
        for p,part_object in self.part_dict.items():
            (v,j) = self.kind_task_tuple[p] #工件p对应的工序对象
            part_object.part_unprocess_list = [] #未分配机器的工件对象列表？？
            part_object.machine_tuple = self.machine_pj_dict[(v,j)]  # 可选加工机器编号元组
            part_object.time_remain = sum(self.time_pj_dict[(v, j)] for j in self.task_p_dict[v][j:])  # 后续剩余处理时间
            part_object.job_now_list = []  # 处于该工序段的工件对象列表
            part_object.job_unprocessed_list = []  # 该工序段未被加工的工件对象列表
            part_object.task_processed_list = []  # 该工序段已加工的工序对象列表
        for m, machine_object in self.machine_dict.items():
            machine_object.kind_task_tuple = self.kind_task_m_dict[m]  # 可选加工工序类型元组
            machine_object.machine_state = 0  # 机器状态
            machine_object.time_end = 0  # 机器完工时间
            machine_object.part_list = []  # 机器已加工工序对象列表
            machine_object.job_object = None


    def reset_fluid_parameter(self):
        """初始化流体属性参数"""
        for part, part_object in self.part_dict.items():
            part_object.fluid_machine_list = []  # 流体模型中可选加工机器
            part_object.fluid_process_rate_m_dict = {}  # 被各机器加工的速率
        for m, machine_object in self.machine_dict.items():
            machine_object.fluid_part_list = []  # 流体解中可选加工工序类型列表
            machine_object.time_ratio_part_dict = {}  # 流体解中分配给各工序类型的时间比例
            machine_object.fluid_process_rate_part_dict = {}  # 流体解中加工各工序类型的速率
            machine_object.unprocessed_part_dict = {}  # 未被m加工的各工件的总数 (p,j)
            machine_object.fluid_unprocessed_part_dict = {}  # 流体解中未被机器m加工的各工序类型总数

    def reset_object(self):                    #重写该方法
        """
        添加工件对象和工序对象+更新流体模型和属性
        :return: 添加工序对象和机器对象+更新流体模型和属性
        """
        for p in self.part_dict.keys():
            (v,j) = self.kind_task_tuple[p]
            # batch_size = self.total_cost_dict[(v, j)]
            # print(f"工序索引 p={p}, 工序(v,j)=({v},{j}), 批量大小={batch_size}")
            n_start = 0
            n_end = n_start + self.total_cost_dict[(v,j)]
            # print(f"  创建 {n_end - n_start} 个工序工件")
            self.part_dict[p].buffer = 0
            self.part_dict[p].count = 0

            for n in range(n_start, n_end):
                part_object =  Task(p,n) #实例化工件对象
                part_object.part_list = []
                part_object.part_unprocessed_list = []
                if v in self.kind_r_tuple and j == 0:
                    idx = self.kind_task_tuple.index((v,j))
                    self.part_dict[idx].part_now_list.append(part_object)  # 初始时刻只有源工件有待加工工件
                else:
                    idx = self.kind_task_tuple.index((v, j))
                    self.part_dict[idx].part_unselected_list.append(part_object)
                # part_object.part_unprocessed_list.append(part_object)
                self.part_dict[p].part_unprocessed_list.append(part_object)  # 未完成工件对象列表
                self.task_dict[(p, n)] = part_object  # 加入工序字典
            # 初始化流体属性
        for p in self.part_dict.keys():
            self.part_dict[p].fluid_number = len(self.part_dict[p].part_now_list)  # 处于该工序段的流体数量
            self.part_dict[p].fluid_unprocessed_number = len(self.part_dict[p].part_unprocessed_list)  # 未被加工的流体数
        x = self.fluid_model()
        # 初始化流体属性
        self.reset_fluid_parameter()
        # 基于流体解更新流体属性
        self.update_fluid_parameter(x)

    def fluid_model(self):
        """
        最小化最大完工时间目标流体模型求解
        """
        # 初始化模型对象
        model = Model('LP')
        # 定义决策变量
        var_list = {(m, (v, j)) for m in self.machine_tuple for (v, j) in self.kind_task_m_dict[m]}
        X = model.continuous_var_dict(var_list, lb=0, ub=1, name='X')
        # 各流体初始未加工数量
        fluid_number = {(v,j):self.part_dict[self.kind_task_tuple.index((v,j))].fluid_unprocessed_number for (v,j) in self.kind_task_tuple}
        # 各流体初始瞬态数量
        fluid_number_time = {(v, j): self.part_dict[self.kind_task_tuple.index((v,j))].fluid_number for (v,j) in self.kind_task_tuple}
        process_rate_part_sum = {(v, j): sum(X[m, (v, j)] * self.process_rate_m_part_dict[m][(v, j)]
                                           for m in self.machine_pj_dict[(v, j)]) for (v, j) in self.kind_task_tuple}
        # 定义目标函数
        model.maximize(model.min(process_rate_part_sum[(v, j)] / fluid_number[(v, j)] for (v, j) in self.kind_task_tuple))
        # 机器利用率约束
        model.add_constraints(model.sum(X[m, (v, j)] for (v, j) in self.kind_task_m_dict[m]) <= 1
                                  for m in self.machine_tuple)
        # 解的可行性约束 (V,J)表示当前工序， (v,j)表示前置工序
        for (V, J) in self.kind_task_tuple:
            if fluid_number_time[(V, J)] == 0 and self.pre_pj_dict[(V, J)]:
                for (v, j) in self.pre_pj_dict[(V, J)]:
                    model.add_constraint(process_rate_part_sum[(v, j)] >= process_rate_part_sum[(V, J)])
        # 求解模型
        solution = model.solve()
        x = solution.get_value_dict(X)
        # 输出流体完工时间
        process_rate_pj_sum = {(v, j): sum(x[m, (v, j)] * self.process_rate_m_part_dict[m][(v, j)]
                                           for m in self.machine_pj_dict[(v, j)]) for (v, j) in self.kind_task_tuple}
        self.fluid_completed_time = max(
            fluid_number[(v, j)] / process_rate_pj_sum[(v, j)] for (v, j) in self.kind_task_tuple)

        # print("流体完工时间：", self.fluid_completed_time)
        # print(solution)
        return x



    def update_fluid_parameter(self, x):
            """基于流体解更新流体参数"""
            for (m, (v, j)), rate in x.items():
                p = self.kind_task_tuple.index((v,j))
                machine_object = self.machine_dict[m]
                part_object = self.part_dict[p]
                machine_object.time_ratio_part_dict[(v, j)] = rate  # 流体解中分配给各工序类型的时间比例
                part_object.fluid_process_rate_m_dict[m] = rate * self.process_rate_m_part_dict[m][(v, j)]
                machine_object.fluid_process_rate_part_dict[(v, j)] = rate * self.process_rate_m_part_dict[m][(v, j)]
                if rate != 0:
                    machine_object.fluid_part_list.append((v, j))
                    part_object.fluid_machine_list.append(m)
            for p, part_object in self.part_dict.items():
                part_object.fluid_rate_sum = sum(part_object.fluid_process_rate_m_dict.values())  # 工序类型处理速率
                part_object.fluid_time_sum = 1 / part_object.fluid_rate_sum  # 工序类型的加工时间
            for m, machine_object in self.machine_dict.items():
                for (v, j) in machine_object.kind_task_tuple:  # 添加除流体模型中可选工序类型外的工序
                    p = self.kind_task_tuple.index((v, j))
                    part_object = self.part_dict[p]
                    # 未被m加工的各工序类型流体数量
                    machine_object.fluid_unprocessed_part_dict[(v, j)] = \
                        part_object.fluid_unprocessed_number * \
                        machine_object.fluid_process_rate_part_dict[(v, j)] / part_object.fluid_rate_sum
                    # 未被m加工的工序o_pj的总数 (p,j)
                    machine_object.unprocessed_part_dict[(v, j)] = machine_object.fluid_unprocessed_part_dict[
                        (v, j)]
                    # 流体解中未被机器m加工的各工序类型总数
                    machine_object.fluid_unprocessed_part_dict[(p, j)] = machine_object.fluid_unprocessed_part_dict[
                        (v, j)]


if __name__ == "__main__":
    data_list = [[4, 2, 2, 4, 2, 2], [4, 2, 2, 4, 2, 4], [4, 2, 2, 4, 2, 6], [4, 2, 4, 8, 4, 2], [4, 2, 4, 8, 4, 4],
                 [4, 2, 4, 8, 4, 6], [4, 2, 6, 12, 6, 2], [4, 2, 6, 12, 6, 4], [4, 2, 6, 12, 6, 6], [8, 4, 2, 4, 2, 2],
                 [8, 4, 2, 4, 2, 4], [8, 4, 2, 4, 2, 6], [8, 4, 4, 8, 4, 2], [8, 4, 4, 8, 4, 4], [8, 4, 4, 8, 4, 6],
                 [8, 4, 6, 12, 6, 2], [8, 4, 6, 12, 6, 4], [8, 4, 6, 12, 6, 6], [12, 6, 2, 4, 2, 2],
                 [12, 6, 2, 4, 2, 4],
                 [12, 6, 2, 4, 2, 6], [12, 6, 4, 8, 4, 2], [12, 6, 4, 8, 4, 4], [12, 6, 4, 8, 4, 6],
                 [12, 6, 6, 12, 6, 2],
                 [12, 6, 6, 12, 6, 4], [12, 6, 6, 12, 6, 6]]
    for data in data_list:
        fajsp_instance = FAJSP(M_p=data[0], M_a=data[1], product_count=data[2], kind_count=data[3], J_r=data[4], N_p=data[5])

    # print("未显示流体解，将101，205,207行注释取消")
    # fajsp_instance = FAJSP(M_p=10, M_a=5, N=0)

