import simpy
from math import floor
from system import System
from core import Core
from component import Component
from task import Task

class Simulator:
    def __init__(self, system: System):
        """
        A class representing a Simulator

        Attributes:
            system (System): System of the simulator
            environment (simpy Environment): simpy Environment
        """
        self.system = system
        self.environment = simpy.Environment()
        self.resource = {core.core_id: simpy.Resource(self.environment, capacity=1) for core in self.system.core.values()}

    def schedule_edf(self, items, time:float, is_task=True):
        """
        Schedules items based on the Earliest Deadline First (EDF) algorithm.

        This method sorts a list of items (either tasks or components) by their 
        earliest deadline. For tasks, the deadline is calculated based on the 
        task deadline + task_period * number of iterations. For components, the sorting is based 
        on their the supposed next period the task should be running.

        Args:
            items (list): A list of items to be scheduled. Each item is expected 
                          to have attributes `deadline` and `period` if `is_task` 
                          is True, or just `period` if `is_task` is False.
            time (float): The current time, used to calculate the next deadline 
                          for each item.
            is_task (bool, optional): A flag indicating whether the items are 
                                      tasks (True) or components (False). 
                                      Defaults to True.

        Returns:
            list: A sorted list of items based on their earliest deadline.
        """
        if is_task:
            return sorted(items, key= lambda t: t.deadline + t.period * floor(time / t.period))
        return sorted(items, key= lambda c: c.period + c.period * floor(time / c.period))

    # TODO FIGURE OUT HOW THE PRIORITY IS CALCULATED - CHECK AGAINST PERIOD
    # TODO check whether it should be ascending or descending
    def schedule_rm(self, items):
        """
        Schedules a list of items using the Rate Monotonic (RM) scheduling algorithm.
        The RM scheduling algorithm prioritizes tasks based on their priority value,        Args:
            items (list): A list of either Task or Component, where each item is expected to have a 
                          'priority' attribute.
        Returns:
            list: A sorted list of items, ordered by their priority in ascending order.
        """

        return sorted(items, key= lambda a: a.priority)

    def process_task(self, task: Task, core: Core, remaining_wcet: float):
        """
        Simulates the processing of a task on a specific core.
        Args:
            task (Task): The task to be processed.
            core (Core): The core on which the task will be executed.
            remaining_wcet (float): The remaining worst-case execution time (WCET) of the task to be processed.
        Yields:
            SimPy events:
            - A request for the core's resource, simulating the task acquiring 
              the core for execution.
            - A timeout event, representing the time taken to execute the task 
              based on the core's speed factor.
        Side Effects:
            Updates the task's response_times list with the current simulation time 
            modulo the task's period, representing the task's response time.
        """
        yield self.resource[core.core_id].request()
        execution_time = remaining_wcet * core.speed_factor
        yield self.environment.timeout(execution_time)
        # TODO check this, why modulo??
        task.response_times.append(self.environment.now % task.period)

    def process_component(self, component: Component, core:Core):
        """
        Simulates the processing of a component on a given core in a real-time system.
        This method models the execution of tasks within a component, adhering to the 
        specified scheduling policy (e.g., EDF or RM). It accounts for task deadlines, 
        available budget, and the core's speed factor to determine the execution behavior.
        Args:
            component (Component): The component containing tasks to be processed. 
            core (Core): The core on which the component's tasks are executed. 
        Yields:
            simpy.events.Timeout: A timeout event representing the periodic execution 
                                    of the component.
            simpy.events.Process: A process event for executing individual tasks.
        Raises:
            ValueError: If the specified scheduler type in the component is not supported.
        Notes:
            - The method uses the component's scheduler type to determine the scheduling 
                policy (e.g., EDF or RM).
            - Tasks are scheduled based on their deadlines or priorities, and execution 
                is constrained by the available budget and core speed factor.
            - If the available budget is insufficient to execute a task fully, the task 
                is processed up to the remaining budget.
        """
        while True:
            yield self.environment.timeout(component.period)
            #TODO check whether we need the speed factor
            available_budget = component.budget * core.speed_factor
            active_tasks = []
            scheduled_tasks = []
            for task in component.tasks:
                task_deadline = task.deadline + task.period * floor(self.environment.now() / task.period)
                if self.environment.now() >= task_deadline - task.period: 
                    active_tasks.append(task_deadline)
            if active_tasks:
                if component.scheduler == 'EDF':
                    scheduled_tasks = self.schedule_edf(active_tasks, self.environment.now(), True)
                if component.scheduler == 'RM':
                    scheduled_tasks = self.schedule_rm(active_tasks)
                else:
                    raise ValueError(f"Not an appropriate scheduler {component.scheduler}")
            if scheduled_tasks:
                for task in scheduled_tasks:
                    if available_budget <= 0:
                        print("No budget left")
                        break
                    estimated_wcet = task.wcet * core.speed_factor
                    if estimated_wcet > available_budget:
                        # Use up everything
                        yield from self.process_task(task, core, available_budget)
                        break
                    else:
                        yield from self.process_task(task, core, estimated_wcet)
                        available_budget -= estimated_wcet
                        
    def process_core(self, core:Core):
        """
        Processes the components of a given core based on their scheduling policy.
        Args:
            core (Core): The core object containing components and scheduling policy.
        Raises:
            ValueError: If the core's scheduler is not recognized (neither 'EDF' nor 'RM').
        Behavior:
            - Continuously checks for active components whose periods align with the current
              simulation time (`self.environment.now()`).
            - Depending on the core's scheduler type:
                - 'EDF' (Earliest Deadline First): Calls `schedule_edf` to determine the
                  order of execution for active components.
                - 'RM' (Rate Monotonic): Calls `schedule_rm` to determine the order of
                  execution for active components.
            - Processes the scheduled components by invoking `process_component` for each.
        """

        while True:
            active_components = [c for c in core.components if self.environment.now() % c.period == 0]
            scheduled_components = []
            if active_components:
                if core.scheduler == 'EDF':
                    scheduled_components = self.schedule_edf(active_components, self.environment.now(), False)
                if core.scheduler == 'RM':
                    scheduled_components = self.schedule_rm(active_components)
                else:
                    raise ValueError(f"Not an appropriate scheduler {core.scheduler}")
            if scheduled_components:
                for component in scheduled_components:
                    yield from self.process_component(component, core)
    
    def run_simulation(self):
        for core in self.system.core.values():
            self.env.process(self.process_core(core))
        self.env.run(until=self.system.hyperperiod)