# tasks.csv - Task Set
# This file describes the individual tasks within the system.

# Columns:

# task_name: The name of the task (string).
# wcet: The Worst-Case Execution Time (WCET) of the task in time units, assuming a nominal core speed (speed_factor = 1.0) (float).
# period: The period of the task in time units (float).
# component_id: The ID of the component to which the task is assigned (string).
# priority: The priority of the task (integer). This column is only relevant for tasks within components that use RM scheduling.
# For EDF components, this column will be empty. Priorities are assigned based on the Rate Monotonic (RM) principle (shorter period = higher priority).
import math
class Task:
    def __init__(self, name: str, wcet: float, period: float, component_id: str, priority=None):
        """
        A class representing a task.

        Attributes:
            name: name of the task
            wcet (float): Worst case execution time
            period (float): Fixed amount of time between when one instance of the task starts (or becomes ready to run) and when the next instance begins.
            priority (float): Priority of the task compared to other tasks
            deadline (float): Equal to the period as mentioned by the project description
        """
        self.name  = name
        self.wcet = wcet
        self.period = period
        self.component_id = component_id
        # The priority of the task (integer). 
        # This column is only relevant for tasks within components that use RM scheduling. 
        # For EDF components, this column will be empty. 
        # Priorities are assigned based on the Rate Monotonic (RM) principle (shorter period = higher priority).
        self.priority = priority if priority is not None and not math.isnan(priority) else 1.0 / period
        self.deadline = period
        # Store all the response times every period
        self.response_times= []

    def __str__(self):
        return (f"task {self.name} with wcet:{self.wcet}, period:{self.period}, component_id:{self.component_id}, priority{self.priority}")
