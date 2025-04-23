import pandas as pd
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
    
    
    def load_inputs(self):
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
            name = row['name']
            wcet = row['wcet']
            period = row['period']
            component_id = row['component_id']
            priority = row['priority']
            self.tasks[name] = Task(name, wcet, period, priority, component_id)
