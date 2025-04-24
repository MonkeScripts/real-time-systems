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
        self.priority = priority if priority is not None else float('inf')
        self.tasks = []
        self.bdr_alpha = None
        self.bdr_delta = None
