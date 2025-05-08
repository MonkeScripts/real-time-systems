import math
import numpy as np
import pandas as pd
import simpy
from collections import defaultdict
import logging
from math import lcm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Task:
    def __init__(self, name: str, wcet: float, period: float, component_id: str, priority=None, is_sporadic=False):
        """
        任务类，支持周期性和零星任务。
        参数：
            name: 任务名称
            wcet: 最坏情况执行时间（基于标称核心速度）
            period: 周期（周期性任务）或最小到达间隔（零星任务）
            component_id: 所属组件ID
            priority: 任务优先级（RM调度使用，周期越短优先级越高）
            is_sporadic: 是否为零星任务
        """
        self.name = name
        self.wcet = wcet
        self.period = period
        self.component_id = component_id
        self.priority = priority if priority is not None else float('inf')
        self.deadline = period  # 隐式截止时间等于周期
        self.is_sporadic = is_sporadic
        self.response_times = []  # 存储每次调度的响应时间
        self.wcrt = None  # 最坏情况响应时间

    def get_utilization(self):
        return self.wcet / self.period

    def is_schedulable(self):
        """检查任务是否可调度（WCRT ≤ 截止时间）"""
        return self.wcrt <= self.deadline if self.wcrt is not None else True

    def __str__(self):
        return (f"Task(name={self.name}, wcet={self.wcet}, period={self.period}, "
                f"component_id={self.component_id}, priority={self.priority}, "
                f"is_sporadic={self.is_sporadic}, wcrt={self.wcrt})")

class Component:
    def __init__(self, component_id: str, scheduler: str, budget: float, period: float, core_id: str, priority=None):
        """组件类，管理任务并计算BDR参数"""
        if not isinstance(component_id, str) or not component_id:
            raise ValueError("Component ID must be a non-empty string")
        if scheduler not in ['EDF', 'RM']:
            raise ValueError("Scheduler must be 'EDF' or 'RM'")
        if budget <= 0:
            raise ValueError("Budget must be positive")
        if period <= 0:
            raise ValueError("Period must be positive")
        if not isinstance(core_id, str) or not core_id:
            raise ValueError("Core ID must be a non-empty string")

        self.id = component_id
        self.scheduler = scheduler
        self.budget = budget
        self.period = period
        self.core_id = core_id
        self.priority = priority if priority is not None else float('inf')
        self.tasks = []
        self.bdr_alpha, self.bdr_delta = 0.0, 0.0
        self.update_bdr_parameters()

    def add_task(self, task: Task):
        self.tasks.append(task)
        self.update_bdr_parameters()

    def get_utilization(self):
        return sum(task.get_utilization() for task in self.tasks)

    def dbf_edf(self, t: float):
        if t <= 0:
            return 0.0
        demand = 0.0
        for task in self.tasks:
            demand += math.floor((t + task.deadline) / task.period) * task.wcet
        return demand

    def dbf_fps(self, t: float, task_index: int):
        if t <= 0 or task_index >= len(self.tasks):
            return 0.0
        task = self.tasks[task_index]
        demand = task.wcet
        for i, other_task in enumerate(self.tasks):
            if i >= task_index:
                break
            demand += math.ceil(t / other_task.period) * other_task.wcet
        return demand

    def update_bdr_parameters(self):
        """使用Half-Half算法更新BDR参数"""
        if not self.tasks:
            self.bdr_alpha = self.budget / self.period
            self.bdr_delta = self.period
            logger.info(f"Component {self.id}: No tasks, bdr_alpha={self.bdr_alpha:.2f}, bdr_delta={self.bdr_delta:.2f}")
            return

        U = self.get_utilization()
        if U <= 0:
            self.bdr_alpha = self.budget / self.period
            self.bdr_delta = self.period
            logger.warning(f"Component {self.id}: Zero utilization, bdr_alpha={self.bdr_alpha:.2f}, bdr_delta={self.bdr_delta:.2f}")
            return

        max_deadline = max(task.deadline for task in self.tasks)
        delta = max_deadline / 2
        step = max_deadline / 100
        max_attempts = 200

        for _ in range(max_attempts):
            schedulable = True
            if self.scheduler == 'EDF':
                t_max = int(2 * max_deadline)
                for t in range(1, t_max + 1):
                    sbf = max(0.0, U * (t - delta))
                    if self.dbf_edf(t) > sbf:
                        schedulable = False
                        break
            else:  # RM
                for i in range(len(self.tasks)):
                    t = self.tasks[i].period
                    sbf = max(0.0, U * (t - delta))
                    if self.dbf_fps(t, i) > sbf:
                        schedulable = False
                        break
            if schedulable:
                break
            delta += step

        self.bdr_alpha = min(U, self.budget / self.period)
        self.bdr_delta = max(delta, self.period / 2)
        if self.bdr_delta <= 0:
            self.bdr_delta = self.period
            logger.warning(f"Component {self.id}: Delta forced to period: bdr_delta={self.bdr_delta:.2f}")
        logger.info(f"Component {self.id}: bdr_alpha={self.bdr_alpha:.2f}, bdr_delta={self.bdr_delta:.2f}")

    def is_schedulable(self):
        return all(task.is_schedulable() for task in self.tasks)

    def __str__(self):
        return (f"Component(id={self.id}, scheduler={self.scheduler}, budget={self.budget}, "
                f"period={self.period}, core_id={self.core_id}, priority={self.priority}, "
                f"tasks={len(self.tasks)}, bdr_alpha={self.bdr_alpha:.2f}, bdr_delta={self.bdr_delta:.2f})")

class Core:
    def __init__(self, core_id: str, speed_factor: float, scheduler: str):
        """核心类，管理组件并调整任务WCET"""
        self.core_id = core_id
        self.speed_factor = speed_factor
        self.scheduler = scheduler
        self.components = []

    def __str__(self):
        return (f"Core(core_id={self.core_id}, speed_factor={self.speed_factor}, "
                f"scheduler={self.scheduler}, components={[c.id for c in self.components]})")

class System:
    def __init__(self, tasks_file: str, arch_file: str, budgets_file: str):
        """系统类，加载输入并分配任务和组件"""
        self.arch_file = arch_file
        self.budgets_file = budgets_file
        self.tasks_file = tasks_file
        self.cores = {}
        self.components = {}
        self.tasks = {}
        self.load_inputs()
        self.assign_components_to_cores()
        self.assign_tasks_to_components()

    def load_inputs(self):
        """从CSV文件加载核心、组件和任务配置"""
        required_arch_columns = ['core_id', 'speed_factor', 'scheduler']
        required_budgets_columns = ['component_id', 'scheduler', 'budget', 'period', 'core_id', 'priority']
        required_tasks_columns = ['task_name', 'wcet', 'period', 'component_id', 'priority']

        arch_df = pd.read_csv(self.arch_file)
        if not all(col in arch_df.columns for col in required_arch_columns):
            raise ValueError(f"architecture.csv missing required columns: {required_arch_columns}")
        for _, row in arch_df.iterrows():
            self.cores[row['core_id']] = Core(row['core_id'], row['speed_factor'], row['scheduler'])

        budgets_df = pd.read_csv(self.budgets_file)
        if not all(col in budgets_df.columns for col in required_budgets_columns):
            raise ValueError(f"budgets.csv missing required columns: {required_budgets_columns}")
        for _, row in budgets_df.iterrows():
            self.components[row['component_id']] = Component(
                row['component_id'], row['scheduler'], row['budget'], row['period'], row['core_id'], row['priority']
            )

        tasks_df = pd.read_csv(self.tasks_file)
        if not all(col in tasks_df.columns for col in required_tasks_columns):
            raise ValueError(f"tasks.csv missing required columns: {required_tasks_columns}")
        for _, row in tasks_df.iterrows():
            is_sporadic = row.get('is_sporadic', False)  # 支持零星任务标志
            self.tasks[row['task_name']] = Task(
                row['task_name'], row['wcet'], row['period'], row['component_id'], row['priority'], is_sporadic
            )

        logger.info(f"Loaded {len(self.cores)} cores, {len(self.components)} components, {len(self.tasks)} tasks")

    def assign_tasks_to_components(self):
        for task in self.tasks.values():
            component = self.components[task.component_id]
            component.add_task(task)

    def assign_components_to_cores(self):
        for component in self.components.values():
            core = self.cores[component.core_id]
            core.components.append(component)

    def get_hyperperiod(self) -> int:
        """计算超周期（任务周期的最小公倍数）"""
        if not self.tasks:
            return 1
        periods = [int(task.period * 1000) for task in self.tasks.values()]  # 转换为整数
        return lcm(*periods) // 1000 if periods else 1

class Simulator:
    def __init__(self, system: System):
        """仿真器类，运行调度仿真并计算WCRT"""
        self.system = system
        self.environment = simpy.Environment()
        self.resource = {core.core_id: simpy.Resource(self.environment, capacity=1) for core in self.system.cores.values()}
        self.remaining_wcet = defaultdict(lambda: defaultdict(float))

    def schedule_edf(self, items, time: float, is_task=True):
        if is_task:
            return sorted(items, key=lambda t: t.period * (math.floor(time / t.period) + 1))
        return sorted(items, key=lambda c: c.period * (math.floor(time / c.period) + 1))

    def schedule_rm(self, items):
        return sorted(items, key=lambda a: a.priority)

    def calculate_wcrt(self, component: Component, core: Core):
        """为组件中的每个任务计算WCRT（基于固定优先级和轮询服务器）"""
        if component.scheduler != 'RM':
            logger.warning(f"Component {component.id}: WCRT calculation only supports RM scheduling")
            return

        # 按优先级排序任务（周期越短优先级越高）
        tasks = sorted(component.tasks, key=lambda t: t.priority)
        server_period = component.bdr_delta / 2
        server_budget = component.bdr_alpha * server_period if component.bdr_alpha < 1.0 else float('inf')

        for i, task in enumerate(tasks):
            # 初始化响应时间
            R = task.wcet / core.speed_factor
            while True:
                interference = 0
                for j in range(i):  # 高优先级任务干扰
                    high_prio_task = tasks[j]
                    interference += math.ceil(R / high_prio_task.period) * (high_prio_task.wcet / core.speed_factor)

                # 轮询服务器的延迟
                if server_budget != float('inf'):
                    # 服务器等待时间（最坏情况为服务器周期减去预算）
                    server_delay = server_period - server_budget
                    # 任务可能需要等待多个服务器周期
                    num_server_periods = math.ceil((R + interference) / server_budget)
                    R_new = task.wcet / core.speed_factor + interference + server_delay * num_server_periods
                else:
                    R_new = task.wcet / core.speed_factor + interference

                if abs(R_new - R) < 1e-6 or R_new > task.deadline:
                    break
                R = R_new

            task.wcrt = R
            logger.info(f"Task {task.name}: WCRT={task.wcrt:.2f}, Deadline={task.deadline:.2f}, "
                        f"Schedulable={task.wcrt <= task.deadline}")

    def process_task(self, task: Task, core: Core, execution_time: float, instance: int):
        with self.resource[core.core_id].request() as req:
            yield req
            yield self.environment.timeout(execution_time)
            self.remaining_wcet[task][instance] -= execution_time
            if abs(self.remaining_wcet[task][instance]) < 1e-6:
                release_time = task.period * instance
                response_time = self.environment.now - release_time
                task.response_times.append(response_time)
                del self.remaining_wcet[task][instance]
                logger.info(f"Component {task.component_id}: Task {task.name} (instance {instance}) completed, "
                            f"Response Time={response_time:.2f}")

    def process_component(self, component: Component, core: Core):
        period = component.bdr_delta / 2
        budget = float("inf") if abs(component.bdr_alpha - 1.0) < 1e-6 else component.bdr_alpha * period

        while True:
            yield self.environment.timeout(period)
            available_budget = budget
            active_tasks = []
            for task in component.tasks:
                current_instance = math.floor(self.environment.now / task.period)
                release_time = task.period * current_instance
                absolute_deadline = release_time + task.deadline
                if release_time <= self.environment.now < absolute_deadline:
                    if current_instance not in self.remaining_wcet[task]:
                        self.remaining_wcet[task][current_instance] = task.wcet / core.speed_factor
                    active_tasks.append((task, current_instance))

            if active_tasks:
                if component.scheduler == "EDF":
                    scheduled_tasks = self.schedule_edf([t[0] for t in active_tasks], self.environment.now, True)
                    scheduled_tasks = [(t, next(i for t2, i in active_tasks if t2 == t)) for t in scheduled_tasks]
                else:  # RM
                    scheduled_tasks = self.schedule_rm([t[0] for t in active_tasks])
                    scheduled_tasks = [(t, next(i for t2, i in active_tasks if t2 == t)) for t in scheduled_tasks]

                for task, instance in scheduled_tasks:
                    remaining_wcet = self.remaining_wcet[task][instance]
                    if remaining_wcet <= 0:
                        continue
                    execution_time = min(remaining_wcet, available_budget)
                    yield from self.process_task(task, core, execution_time, instance)
                    available_budget -= execution_time
                    if available_budget <= 0:
                        break

    def process_core(self, core: Core):
        while True:
            full_util_component = next((c for c in core.components if abs(c.bdr_alpha - 1.0) < 1e-6), None)
            if full_util_component:
                yield from self.process_component(full_util_component, core)
            else:
                active_components = [c for c in core.components if abs(self.environment.now % (c.bdr_delta / 2)) < 1e-6]
                if active_components:
                    if core.scheduler == "EDF":
                        scheduled_components = self.schedule_edf(active_components, self.environment.now, False)
                    else:
                        scheduled_components = self.schedule_rm(active_components)
                    for component in scheduled_components:
                        yield from self.process_component(component, core)

    def run_simulation(self):
        """运行仿真并计算WCRT"""
        # 先计算WCRT
        for core in self.system.cores.values():
            for component in core.components:
                self.calculate_wcrt(component, core)

        # 运行仿真
        for core in self.system.cores.values():
            self.environment.process(self.process_core(core))
        hyperperiod = self.system.get_hyperperiod()
        logger.info(f"Running simulation until hyperperiod: {hyperperiod}")
        self.environment.run(until=hyperperiod)

    def report(self):
        """生成WCRT和仿真结果报告"""
        results = []
        for core in self.system.cores.values():
            for component in core.components:
                component_schedulable = True
                task_details = []
                for task in component.tasks:
                    avg_response_time = np.mean(task.response_times) if task.response_times else float('nan')
                    max_response_time = np.max(task.response_times) if task.response_times else float('nan')
                    task_schedulable = task.is_schedulable()
                    component_schedulable &= task_schedulable
                    task_details.append({
                        "task_name": task.name,
                        "is_sporadic": task.is_sporadic,
                        "wcrt": task.wcrt,
                        "deadline": task.deadline,
                        "task_schedulable": task_schedulable,
                        "avg_response_time": avg_response_time,
                        "max_response_time": max_response_time
                    })
                results.append({
                    "core_id": core.core_id,
                    "component_id": component.id,
                    "component_schedulable": component_schedulable,
                    "task_summary": task_details
                })

        df = pd.DataFrame(results)
        df.to_csv("wcrt_results.csv", index=False)
        logger.info("Generated WCRT and simulation results in wcrt_results.csv")

def half_half_algorithm(budget: float, period: float):
    """Half-Half算法，计算BDR参数"""
    alpha = budget / period
    delta = 2 * (period - budget)
    if delta <= 0:
        delta = period
    return alpha, delta

# 示例运行
if __name__ == "__main__":
    tiny_test_arch_path = "C:\\Users\\17675\\Desktop\\02225\\4-large-test-case\\architecture.csv"
    tiny_test_budgets_path = "C:\\Users\\17675\\Desktop\\02225\\4-large-test-case\\budgets.csv"
    tiny_test_tasks_path = "C:\\Users\\17675\\Desktop\\02225\\4-large-test-case\\tasks.csv"
    tiny_system = System(tasks_file=tiny_test_tasks_path, arch_file=tiny_test_arch_path, budgets_file=tiny_test_budgets_path)
    tiny_simulator = Simulator(tiny_system)
    tiny_simulator.run_simulation()
    tiny_simulator.report()