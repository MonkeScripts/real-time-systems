from collections import defaultdict
import math
class Component:
    def __init__(self, component_id: str, scheduler: str, budget: float, period: float, core_id: str, priority=None):
        """
        A class representing a component.

        Attributes:
            component_id (str): id of component
            scheduler (str): either EDF or RM
            budget (float): The initial budget (Q) for the component, in time units (float). This value represents the resource allocation for each component.
            period (float): The initial period (P) for the component, in time units (float).
            core_id (str):  The ID of the core to which the component is assigned (string).
            priority (int): The priority of the component (integer). This column is only relevant for components within cores that use RM scheduling as the top level scheduler. For EDF cores, this column will be empty. Priorities are assigned based on the Rate Monotonic (RM) principle (shorter period = higher priority).
            tasks (list(task)): list of task to be scheduled and ran by the component
            bdr_alpha (float): Resource availability factor. The fraction of processing time guaranteed.
            bdr_delta (float): Maximum delay in resource allocation. Bounds the worst-case delay in getting the allocated resource.
        """
        self.id = component_id
        self.scheduler = scheduler
        self.budget = budget
        self.period = period
        self.core_id = core_id
        self.priority = (
            priority
            if priority is not None and not math.isnan(priority)
            else 1.0 / self.period
        )
        self.tasks = []
        self.bdr_alpha, self.bdr_delta = self.half_half_algorithm(self.budget, self.period)

    def __str__(self):
        return (f"Component(id={self.id}, scheduler={self.scheduler}, budget={self.budget}, "
                f"period={self.period}, core_id={self.core_id}, priority={self.priority}, "
                f"tasks={len(self.tasks)}, bdr_alpha={self.bdr_alpha:.2f}, bdr_delta={self.bdr_delta:.2f})")

    def half_half_algorithm(self, budget:float, period:float):
        """
        Implements the Half-Half Algorithm for real-time systems.
        This algorithm calculates two parameters, alpha and delta, based on 
        the given budget and period. These parameters can be used for 
        scheduling or resource allocation in real-time systems.
        Parameters:
        -----------
        budget : float
            The allocated execution time or computational budget.
        period : float
            The time period over which the budget is allocated.
        Returns:
        --------
        alpha : represents the utilization factor, calculated as the ratio 
          of budget to period.
        delta : represents a derived parameter, calculated as twice the 
          difference between the period and the budget.
        """
        alpha = budget / period
        delta = 2 * (period - budget)
        return alpha, delta
