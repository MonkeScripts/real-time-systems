import pandas as pd
from math import lcm
from core import Core
from component import Component
from task import Task


class System:
    def __init__(self, tasks_file: str, arch_file: str, budgets_file: str):
        """
        A class representing a system.

        Attributes:
            arch_file (str): file path of arch.csv
            budgets_file (str): file path of budget.csv
            tasks_file (str): file path of tasks.csv
        """
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
        '''
        Loads inputs from given csv file names, store them in their respective dictionaries        
        '''
        arch_df = pd.read_csv(self.arch_file)
        for _, row in arch_df.iterrows():
            core_id = row['core_id']
            speed_factor = row['speed_factor']
            scheduler = row['scheduler']
            self.cores[row['core_id']] = Core(core_id, speed_factor, scheduler)
        budgets_df = pd.read_csv(self.budgets_file)
        for _, row in budgets_df.iterrows():
            component_id = row['component_id']
            scheduler = row['scheduler']
            budget = row['budget']
            period = row['period']
            core_id = row['core_id']
            priority = row['priority']
            self.components[component_id] = Component(component_id, scheduler, budget, period, core_id, priority)

        tasks_df = pd.read_csv(self.tasks_file)
        for _, row in tasks_df.iterrows():
            name = row['task_name']
            wcet = row['wcet']
            period = row['period']
            component_id = row['component_id']
            priority = row['priority']
            self.tasks[name] = Task(name, wcet, period, component_id, priority)
        print(f"loaded dataframes: \n{arch_df}\n{budgets_df}\n{tasks_df}")

    def assign_tasks_to_components(self):
        """
        Add tasks to components
        """
        for task in self.tasks.values():
            component = self.components[task.component_id]
            component.tasks.append(task)
    
    def assign_components_to_cores(self):
        """
        Add components to cores
        """
        for component in self.components.values():
            core = self.cores[component.core_id]
            core.components.append(component)

    def get_hyperperiod(self) -> int:
        """
        Mainly used to determine how long the simulation would run.
        """
        periods = [task.period for task in self.tasks.values()]
        return lcm(*periods) if periods else 1
