import math
from system import System

class Analysis:
    def __init__(self, system: System):
        self.system = system
        self.analysis_results = {}

    def dbf_edf(self, component, t):
        """EDF demand bound function"""
        demand = 0
        for task in component.tasks:
            adjusted_wcet = task.wcet / self.system.cores[component.core_id].speed_factor
            if t >= task.deadline:
                instances = math.floor((t + task.period - task.deadline) / task.period)
                demand += instances * adjusted_wcet
        return demand

    def dbf_fps(self, component, task_idx, t):
        """FPS demand bound function"""
        tasks = sorted(component.tasks, key=lambda task: task.priority)
        if task_idx >= len(tasks):
            return 0

        task = tasks[task_idx]
        core = self.system.cores[component.core_id]
        adjusted_wcet = task.wcet / core.speed_factor
        
        demand = adjusted_wcet
        
        for i in range(task_idx):
            high_prio_task = tasks[i]
            adjusted_high_wcet = high_prio_task.wcet / core.speed_factor
            demand += math.ceil(t / high_prio_task.period) * adjusted_high_wcet

        return demand

    def sbf_bdr(self, alpha, delta, t):
        """BDR supply bound function"""
        if t <= delta:
            return 0
        return alpha * (t - delta)
    
    def compute_optimal_bdr(self, component):
        """Compute optimal BDR parameters"""
        utilization = sum(task.wcet / (task.period * self.system.cores[component.core_id].speed_factor) 
                         for task in component.tasks)

        if not component.tasks:
            return 0, 0

        # Start with alpha equal to utilization
        alpha = utilization
        
        # Find minimum delta that makes component schedulable
        max_deadline = max(task.deadline for task in component.tasks)
        min_period = min(task.period for task in component.tasks)
        
        # Try delta = 0 first
        delta = 0
        if self._is_component_schedulable_with_bdr(component, alpha, delta):
            return alpha, delta
        
        # If not schedulable with delta=0, find minimum delta
        delta_min = 0
        delta_max = min(max_deadline * 0.2, min_period * 0.3)
        epsilon = 0.001
        
        # Binary search for minimum delta
        while delta_max - delta_min > epsilon:
            delta = (delta_min + delta_max) / 2
            
            if self._is_component_schedulable_with_bdr(component, alpha, delta):
                delta_max = delta
            else:
                delta_min = delta
        
        delta = delta_max
        
        # If still not schedulable, incrementally increase alpha
        alpha_increment = 0.01
        max_alpha = min(1.0, utilization * 1.2)  # Don't go too high
        
        while alpha <= max_alpha:
            if self._is_component_schedulable_with_bdr(component, alpha, delta):
                return alpha, delta
            
            # Try with smaller delta first
            for test_delta in [0, delta * 0.5, delta * 0.75, delta]:
                if self._is_component_schedulable_with_bdr(component, alpha, test_delta):
                    return alpha, test_delta
            
            alpha += alpha_increment
        
        # Last resort: use minimum feasible alpha
        return alpha, delta
    
    def _is_component_schedulable_with_bdr(self, component, alpha, delta):
        """Check component schedulability"""
        utilization = sum(task.wcet / (task.period * self.system.cores[component.core_id].speed_factor) 
                         for task in component.tasks)
        
        if utilization > alpha:
            return False
        
        if utilization >= alpha * 0.99:
            return False
        
        if component.scheduler == "EDF":
            # For EDF components
            test_points = set()
            
            for task in component.tasks:
                test_points.add(task.deadline)
                test_points.add(task.period)
            
            if delta > 0:
                test_points.add(delta)
            
            test_points = sorted([t for t in test_points if t > 0])
            
            for t in test_points[:10]:  # Limit test points for efficiency
                demand = self.dbf_edf(component, t)
                supply = self.sbf_bdr(alpha, delta, t)
                if demand > supply:
                    return False
            
            return True
        else:  # RM/FPS
            # For RM components
            tasks = sorted(component.tasks, key=lambda task: task.priority)
            
            for i, task in enumerate(tasks):
                # Response time analysis
                wcet = task.wcet / self.system.cores[component.core_id].speed_factor
                response_time = wcet
                
                converged = False
                for iteration in range(20):
                    workload = self.dbf_fps(component, i, response_time)
                    
                    if alpha > 0:
                        new_response_time = delta + workload / alpha
                    else:
                        new_response_time = float('inf')
                    
                    if abs(new_response_time - response_time) < 0.001:
                        converged = True
                        break
                        
                    response_time = new_response_time
                    
                    if response_time > task.deadline * 10:
                        return False
                
                if not converged or response_time > task.deadline:
                    return False
            
            return True
        
    def half_half_transform(self, alpha, delta):

        #Transform BDR interface to resource supply task using Half-Half Algorithm
        #Parameters:alpha: availability factor of BDR model delta: partition delay of BDR model

        if alpha >= 1.0:
            return float('inf'), float('inf')
        
        if delta == float('inf') or delta == 0:
            return float('inf'), float('inf')
            
        wcet = (alpha * delta) / (2 * (1 - alpha))
        period = delta / (2 * (1 - alpha))
        
        return wcet, period
    
    def analyze(self):
        """Perform analysis"""
        for core in self.system.cores.values():
            core_components = [comp for comp in self.system.components.values() 
                               if comp.core_id == core.core_id]

            for component in core_components:
                alpha, delta = self.compute_optimal_bdr(component)
                component.bdr_alpha = alpha
                component.bdr_delta = delta

                schedulable = self.check_component_schedulability(component)
                if alpha < 1.0 and delta != float('inf') and delta > 0:
                    wcet, period = self.half_half_transform(alpha, delta)
                    resource_supply_task = {
                        "wcet": wcet,
                        "period": period
                }
                else:
                    resource_supply_task = {
                        "wcet": float('inf'),
                        "period": float('inf'),
                        "note": "Component not schedulable"
                    }

                self.analysis_results[component.id] = {
                    "bdr_alpha": alpha,
                    "bdr_delta": delta,
                    "schedulable": schedulable,
                    "resource_supply_task": resource_supply_task
                }

            core_schedulable = self.check_core_schedulability(core)
            self.analysis_results[core.core_id] = {
                "schedulable": core_schedulable
            }

    def check_component_schedulability(self, component):
        """Check component schedulability"""
        return self._is_component_schedulable_with_bdr(component, component.bdr_alpha, component.bdr_delta)

    def check_core_schedulability(self, core):
        """Check core schedulability"""
        components = [comp for comp in self.system.components.values() 
                      if comp.core_id == core.core_id]

        if not components:
            return True

        # Check individual component schedulability
        all_components_schedulable = all(self.analysis_results[comp.id]["schedulable"] 
                                       for comp in components 
                                       if comp.id in self.analysis_results)
        
        if not all_components_schedulable:
            return False
        
        # Total bandwidth check
        total_alpha = sum(comp.bdr_alpha for comp in components)
        
        if core.scheduler == "EDF":
            # For EDF cores, need more careful check when at the boundary
            if total_alpha > 1.0:
                return False
            elif total_alpha == 1.0:
                # When exactly at 1.0, check individual components more carefully
                for comp in components:
                    # Check if component's actual demand exceeds its allocation
                    comp_utilization = sum(task.wcet / (task.period * self.system.cores[comp.core_id].speed_factor) 
                                         for task in comp.tasks)
                    # If any component's utilization exceeds its BDR alpha, system is not schedulable
                    if comp_utilization > comp.bdr_alpha:
                        return False
                # If total is exactly 1.0, be conservative
                return False
            else:
                return True
        else:  # RM/FPS
            # Keep existing RM logic unchanged
            n = len(components)
            if n == 0:
                return True

            rm_bound = n * (2**(1/n) - 1)

            # If within RM bound, definitely schedulable
            if total_alpha <= rm_bound:
                return True

            # If exceeds 1.0, definitely not schedulable
            if total_alpha > 1.0:
                return False

            # For small systems, be a bit more lenient
            if len(self.system.components) <= 10:
                return total_alpha <= rm_bound * 1.1

            # For larger systems, be stricter
            return False

    def generate_report(self):
        """Generate report"""
        report = {
            "system_schedulable": True,
            "cores": {},
            "components": {}
        }

        system_schedulable = True
        
        for comp_id, component in self.system.components.items():
            if comp_id in self.analysis_results:
                comp_result = self.analysis_results[comp_id]
                comp_schedulable = comp_result["schedulable"]
                
                if not comp_schedulable:
                    system_schedulable = False
                    
                report["components"][comp_id] = {
                    "scheduler": component.scheduler,
                    "bdr_alpha": comp_result["bdr_alpha"],
                    "bdr_delta": comp_result["bdr_delta"],
                    "schedulable": comp_schedulable,
                    "tasks": {}
                }

                for task in component.tasks:
                    report["components"][comp_id]["tasks"][task.name] = {
                        "wcet": task.wcet,
                        "period": task.period,
                        "deadline": task.deadline
                    }

        for core_id, core in self.system.cores.items():
            if core_id in self.analysis_results:
                core_result = self.analysis_results[core_id]
                core_schedulable = core_result["schedulable"]
                
                if not core_schedulable:
                    system_schedulable = False
                    
                report["cores"][core_id] = {
                    "scheduler": core.scheduler,
                    "speed_factor": core.speed_factor,
                    "schedulable": core_schedulable
                }

        report["system_schedulable"] = system_schedulable
        return report
    
    def main():
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='ADAS Hierarchical Scheduling Analysis Tool')
        parser.add_argument('--tasks', required=True, help='Path to tasks CSV file')
        parser.add_argument('--arch', required=True, help='Path to architecture CSV file')
        parser.add_argument('--budgets', required=True, help='Path to budgets CSV file')
        parser.add_argument('--output', default='analysis_results.json', help='Output file path')
        args = parser.parse_args()

        # Initialize system
        system = System(args.tasks, args.arch, args.budgets)

        # Create analysis tool
        analyzer = Analysis(system)

        # Perform analysis
        analyzer.analyze()

        # Generate report
        report = analyzer.generate_report()

        # Save report
        import json
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=4)

        # Print summary
        print(f"System schedulable: {report['system_schedulable']}")
        for core_id, core_info in report['cores'].items():
            print(f"Core {core_id}: {'Schedulable' if core_info['schedulable'] else 'Not Schedulable'}")

        for comp_id, comp_info in report['components'].items():
            print(f"Component {comp_id}: {'Schedulable' if comp_info['schedulable'] else 'Not Schedulable'}")
            print(f"  BDR parameters: alpha={comp_info['bdr_alpha']:.3f}, delta={comp_info['bdr_delta']:.3f}")