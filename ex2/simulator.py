import simpy
from math import floor
import numpy as np
import pandas as pd
from system import System
from core import Core
from component import Component
from task import Task
from collections import defaultdict


class Simulator:
    def __init__(self, system: System):
        self.system = system
        self.environment = simpy.Environment()
        self.resource = {
            core.core_id: simpy.Resource(self.environment, capacity=1)
            for core in self.system.cores.values()
        }
        # Track remaining WCET: {task: {instance: remaining_wcet}}
        self.remaining_wcet = defaultdict(lambda: defaultdict(float))

    def schedule_edf(self, items, time: float, is_task=True):
        if is_task:
            return sorted(items, key=lambda t: t.period * (floor(time / t.period) + 1))
        return sorted(items, key=lambda c: c.period * (floor(time / c.period) + 1))

    def schedule_rm(self, items):
        return sorted(items, key=lambda a: a.priority)

    def process_task(
        self, task: Task, core: Core, execution_time: float, instance: int
    ):
        with self.resource[core.core_id].request() as req:
            yield req
            yield self.environment.timeout(execution_time)
            self.remaining_wcet[task][instance] -= execution_time
            if abs(self.remaining_wcet[task][instance]) < 1e-6:  # Job completed
                release_time = task.period * instance
                response_time = self.environment.now - release_time
                task.response_times.append(response_time)
                del self.remaining_wcet[task][instance]
                print(
                    f"Component {task.component_id}: Task {task.name} (instance {instance}) completed"
                )

    def process_component(self, component: Component, core: Core):
        """
        Simulates the processing of a component on a given core.
        Unified scheduling loop for full utilization and normal BDR cases.
        Tracks task completion to ensure all tasks execute correctly.
        """
        # Set period and budget
        period = component.bdr_delta / 2
        if abs(component.bdr_alpha - 1.0) < 1e-6:  # Full utilization
            budget = float("inf")  # Unlimited budget
        else:
            budget = component.bdr_alpha * period
            print(
                f"Component {component.id}: bdr_delta/2 = {period}, budget = {budget}"
            )

        while True:
            yield self.environment.timeout(period)
            available_budget = budget / core.speed_factor
            if budget != float("inf"):
                print(
                    f"Component {component.id}: available_budget = {available_budget}"
                )

            # Collect active tasks
            active_tasks = []
            for task in component.tasks:
                current_instance = floor(self.environment.now / task.period)
                release_time = task.period * current_instance
                absolute_deadline = release_time + task.deadline
                if release_time <= self.environment.now < absolute_deadline:
                    if current_instance not in self.remaining_wcet[task]:
                        self.remaining_wcet[task][current_instance] = (
                            task.wcet / core.speed_factor
                        )
                    active_tasks.append((task, current_instance))

            # Schedule tasks
            if active_tasks:
                if component.scheduler == "EDF":
                    scheduled_tasks = self.schedule_edf(
                        [t[0] for t in active_tasks], self.environment.now, True
                    )
                    scheduled_tasks = [
                        (t, next(i for t2, i in active_tasks if t2 == t))
                        for t in scheduled_tasks
                    ]
                elif component.scheduler == "RM":
                    scheduled_tasks = self.schedule_rm([t[0] for t in active_tasks])
                    scheduled_tasks = [
                        (t, next(i for t2, i in active_tasks if t2 == t))
                        for t in scheduled_tasks
                    ]
                else:
                    raise ValueError(f"Unsupported scheduler {component.scheduler}")

                # Execute tasks
                for task, instance in scheduled_tasks:
                    remaining_wcet = self.remaining_wcet[task][instance]
                    if remaining_wcet <= 0:
                        continue  # Skip completed tasks
                    print(
                        f"Component {component.id}: Running task {task.name} (instance {instance}) with remaining_wcet: {remaining_wcet}, budget: {available_budget}"
                    )
                    if available_budget <= 0:
                        print(f"Component {component.id}: No budget left")
                        break
                    execution_time = min(remaining_wcet, available_budget)
                    yield from self.process_task(task, core, execution_time, instance)
                    available_budget -= execution_time
                    if budget != float("inf"):
                        print(
                            f"Component {component.id}: Remaining budget = {available_budget}"
                        )

    def process_core(self, core: Core):
        while True:
            # Prioritize full-utilization component
            full_util_component = next(
                (c for c in core.components if abs(c.bdr_alpha - 1.0) < 1e-6), None
            )
            if full_util_component:
                yield from self.process_component(full_util_component, core)
            else:
                active_components = [
                    c
                    for c in core.components
                    if self.environment.now % (c.bdr_delta / 2) == 0
                ]
                if active_components:
                    if core.scheduler == "EDF":
                        scheduled_components = self.schedule_edf(
                            active_components, self.environment.now, False
                        )
                    elif core.scheduler == "RM":
                        scheduled_components = self.schedule_rm(active_components)
                    else:
                        raise ValueError(f"Unsupported scheduler {core.scheduler}")

                    for component in scheduled_components:
                        yield from self.process_component(component, core)

    def run_simulation(self):
        for core in self.system.cores.values():
            self.environment.process(self.process_core(core))
        hyperperiod = self.system.get_hyperperiod()
        print(f"Running simulation until hyperperiod: {hyperperiod}")
        self.environment.run(until=hyperperiod)

    def report(self):
        results = []
        for core in self.system.cores.values():
            for component in core.components:
                component_schedulable = True
                task_details = []
                for task in component.tasks:
                    avg_response_time = (
                        np.mean(task.response_times) if task.response_times else 0
                    )
                    max_response_time = (
                        np.max(task.response_times) if task.response_times else 0
                    )
                    task_schedulable = (
                        all(rt <= task.deadline for rt in task.response_times)
                        if task.response_times
                        else True
                    )
                    component_schedulable &= task_schedulable
                    task_details.append(
                        {
                            "task_name": task.name,
                            "task_schedulable": task_schedulable,
                            "avg_response_time": avg_response_time,
                            "max_response_time": max_response_time,
                        }
                    )
                results.append(
                    {
                        "component_id": component.id,
                        "component_schedulable": component_schedulable,
                        "task_summary": task_details,
                    }
                )
        pd.DataFrame(results).to_csv("results.csv", index=False)
        print("Generated sim results")
