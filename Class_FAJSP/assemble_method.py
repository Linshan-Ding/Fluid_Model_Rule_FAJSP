import random
from random import randint, sample
import numpy as np
from collections import defaultdict

"""分割产品数"""
def split(n: int, parts: int) :
    """分割产品数"""
    quotient, remainder = divmod(n, parts)
    return [quotient + 1] * remainder + [quotient] * (parts - remainder)

"""装配序列选择"""
def assemble_schedule(n):
    """装配序列选择"""
    result = []
    max_twos = n // 2

    for count_2 in range(1, max_twos + 1):
        count_1 = n - 2 * count_2
        if count_1 >= 0:
            combo = [2] * count_2 + [1] * count_1
            if combo not in result:
                result.append(combo)
    sol=random.choice(result)
    if sol[1] == 2:
        sol.append(0) #0仅仅表示未显示工序
    return sol

"""对列表中的每个数应用assemble_schedule函数并随机选择一种方案"""
def process_list(numbers):
    combined_results = []
    for num in numbers:
        schedules = assemble_schedule(num)
        if schedules:  # 确保有可选方案
            combined_results.append(random.choice(schedules))
    return combined_results

"""随机分组"""
def random_group(numbers, pattern):
    random.shuffle(numbers)
    return tuple(numbers[sum(pattern[:i]):sum(pattern[:i+1])] for i in range(len(pattern)))

"""计算消耗量"""
def process_operations(operations, product_number):
    """
    处理工序数据，最后一道工序乘以产品系数

    :param operations: 原始工序字典 { (product, idx): [val1, val2] }
    :param product_number: 产品数量 { product: number }
    :return: 处理后的工序字典
    """
    # 获取所有产品和工序索引
    products = set()
    max_indices = {}
    for (product, idx), _ in operations.items():
        products.add(product)
        if product not in max_indices or idx > max_indices[product]:
            max_indices[product] = idx

    # 创建一个新的字典来存储处理后的结果
    processed_ops = operations.copy()

    for product in products:
        max_idx = max_indices[product]
        # 从后往前处理工序
        for idx in range(max_idx, -1, -1):
            key = (product, idx)
            if idx == max_idx:
                # 最后一道工序，乘以产品数量
                multiplier = product_number[product]
                processed_ops[key] = [
                    val * multiplier for val in processed_ops[key]
                ]
            else:
                # 获取后续工序的第一个元素作为乘数
                next_key = (product, idx + 1)
                multiplier = processed_ops[next_key][0]
                # 当前工序的所有元素乘以后续工序的第一个元素
                processed_ops[key] = [
                    val * multiplier for val in processed_ops[key]
                ]
    return processed_ops

"""提取工件与消耗的对应关系"""
def extract_operation_mapping(product_data, operation_data):
    """
    从工序消耗字典和产品工件字典中提取对应关系
    :param product_data: 产品工件字典 {产品: [工件1, 工件2, ...]}
    :param operation_data: 工序消耗字典 {(产品, 工序索引): [消耗1, 消耗2]}
    :return: 工件到消耗的映射字典 {工件: 消耗}
    """
    mapping = {}

    for (product, op_idx), consumes in operation_data.items():
        if product in product_data:
            workpieces = product_data[product]
            # 确保工件数据长度足够
            if op_idx + 1 < len(workpieces):
                # 第一道工序的两个消耗对应前两个工件
                if op_idx == 0:
                    mapping[workpieces[op_idx]] = consumes[0]
                    mapping[workpieces[op_idx + 1]] = consumes[1]
                # 后续工序的第二个消耗对应下一个工件
                else:
                    if op_idx + 1 < len(workpieces):
                        mapping[workpieces[op_idx + 1]] = consumes[1]
    return dict(sorted(mapping.items()))


def process_dicts(cost_dict, p_dict):

    result = []

    # 遍历cost_dict中所有索引为1的项
    for key in cost_dict:
        process, index = key
        if index == 1:  # 只处理索引为1的项
            value1 = cost_dict[key][0]  # 取列表第一个元素
            p_value = p_dict[process]  # 从p_dict获取对应值
            result.append((value1, p_value))

    # 添加p_dict中没有对应cost_dict索引1项的值
    for process in p_dict:
        has_index1 = any(key == (process, 1) for key in cost_dict)
        if not has_index1:
            result.append(p_dict[process])

    # 将结果列表展平为元组
    flattened = []
    for item in result:
        if isinstance(item, tuple):
            flattened.extend(item)
        else:
            flattened.append(item)

    return tuple(flattened)


"""为每个产品分配工件"""
def allocation_no_duplicates(source, targets):
    component_dict = {}
    source_list = list(source)
    used_elements = set()  # 用于跟踪已使用的元素

    for t in targets:
        # 随机分配3-4个元素
        num_elements = randint(3, 4)
        selected_elements = []

        # 优先选择未使用的元素
        unused_elements = list(set(source_list) - used_elements)

        # 确保有足够的不同元素可供选择
        if len(unused_elements) >= num_elements:
            # 如果未使用元素足够，从中选择
            selected_elements = sample(unused_elements, num_elements)
        else:
            # 如果未使用元素不足，先用完未使用的，再从已使用的补充
            selected_elements = unused_elements.copy()
            remaining = num_elements - len(unused_elements)
            # 从已使用元素中选择不重复的补充
            available_elements = [x for x in source_list if x not in selected_elements]
            if len(available_elements) >= remaining:
                selected_elements.extend(sample(available_elements, remaining))
            else:
                # 如果源元素不足，减少分配数量
                selected_elements.extend(available_elements)
                num_elements = len(selected_elements)

        # 更新已使用元素集合
        used_elements.update(selected_elements)
        component_dict[t] = tuple(selected_elements)

    return component_dict

"""计算每个工序的总成本"""
def calculate_total_cost(pre_pj_dict, cost_pj_dict, final_products, product_process_paths):
    """
    计算每个工序的总成本

    参数:
        pre_pj_dict: 工序前置关系字典 {(p,j): [(前置工序)...]}
        cost_pj_dict: 工序消耗成本字典 {(p,j): 成本或成本列表}
        final_products: 最终产品数量字典 {产品: 数量}

    返回:
        total_cost_dict: 各工序总成本字典 {(p,j): 总成本}
    """
    # 初始化总成本字典
    total_cost_dict = defaultdict(int)

    # 反向映射：建立每个工序被哪些后续工序依赖
    reverse_deps = defaultdict(list)
    for (p, j), deps in pre_pj_dict.items():
        for (dep_p, dep_j) in deps:
            reverse_deps[(dep_p, dep_j)].append((p, j))

    # 计算每个最终产品的工序路径
    product_process_paths = product_process_paths

    # 遍历所有最终产品
    for product, quantity in final_products.items():
        # 获取该产品的工序路径
        process_path = product_process_paths[product]

        # 从最终工序反向计算需求
        demand = defaultdict(int)
        demand[process_path[-1]] = quantity

        # 反向传播需求
        visited = set()
        stack = [process_path[-1]]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            # 获取当前工序的前置工序
            predecessors = pre_pj_dict.get(current, [])

            # 获取当前工序的成本结构
            costs = cost_pj_dict.get(current, 1)
            if not isinstance(costs, list):
                costs = [costs] * len(predecessors)

            # 计算每个前置工序的需求量
            for i, (pred_p, pred_j) in enumerate(predecessors):
                if i < len(costs):
                    pred_cost = costs[i]
                else:
                    pred_cost = 1  # 默认成本

                # 计算前置工序需求 = 当前工序需求 * 消耗系数
                pred_demand = demand[current] * pred_cost
                demand[(pred_p, pred_j)] += pred_demand

                # 添加到处理栈
                stack.append((pred_p, pred_j))

        # 累加到总成本
        for (p, j), qty in demand.items():
            total_cost_dict[(p, j)] += qty

    return dict(total_cost_dict)