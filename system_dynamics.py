import math
import numpy as np
from typing import List, Dict, Any

def system_dynamics_sim(user_initial=1000, 
                       months=12,
                       marketing_spend=20000, 
                       referral_rate=0.02,
                       churn_rate=0.05, 
                       acquisition_efficiency=0.01,
                       seasonality=False):
    """
    Flash DNA system dynamics:
    - referral inflow
    - marketing inflow
    - churn outflow
    Optionally includes a seasonality factor.
    """
    arr = []
    curr_users = user_initial

    for m in range(months):
        if seasonality:
            factor = 1 + 0.1 * math.sin(m*(2*math.pi/12))
        else:
            factor = 1

        inflow_referral = referral_rate* curr_users* factor
        inflow_marketing= marketing_spend* acquisition_efficiency* factor
        outflow = churn_rate* curr_users

        curr_users= curr_users + inflow_referral + inflow_marketing - outflow
        arr.append(curr_users)

    return arr

def virality_sim(user_initial=100,
                k_factor=0.2,
                conversion_rate=0.3,
                cycles=12,
                cycle_length_days=14):
    """
    Flash DNA K-factor viral growth simulation.
    Returns dict with 'users','new_users','days','is_viral', etc.
    """
    invites_per_user= k_factor/ conversion_rate if conversion_rate>0 else 0
    users= [user_initial]
    new_users= [user_initial]
    total_invites= [user_initial* invites_per_user]

    for c in range(1, cycles):
        cycle_new_users= total_invites[c-1]* conversion_rate
        cycle_total_users= users[c-1] + cycle_new_users
        cycle_invites= cycle_new_users* invites_per_user

        new_users.append(cycle_new_users)
        users.append(cycle_total_users)
        total_invites.append(cycle_invites)

    days= [c* cycle_length_days for c in range(cycles)]
    is_viral= k_factor>1.0

    time_to_10x= None
    time_to_100x= None
    if is_viral:
        for i, user_count in enumerate(users):
            if user_count>= user_initial*10 and time_to_10x is None:
                time_to_10x= days[i]
            if user_count>= user_initial*100 and time_to_100x is None:
                time_to_100x= days[i]

    return {
        "is_viral": is_viral,
        "users": users,
        "new_users": new_users,
        "days": days,
        "time_to_10x": time_to_10x,
        "time_to_100x": time_to_100x,
        "final_users": users[-1],
        "growth_multiple": users[-1]/ user_initial if user_initial>0 else 0
    }

def calculate_growth_metrics(user_array: List[float]) -> Dict[str, float]:
    """
    Flash DNA + NEW(UI):
    Basic growth metrics from a user time series: initial, final, multiple, MoM.
    """
    if not user_array or len(user_array)<2:
        return {
            "initial_users": 0,
            "final_users": 0,
            "growth_multiple": 1.0,
            "avg_mom_growth_rate": 0.0
        }
    initial= user_array[0]
    final= user_array[-1]
    multiple= final/ initial if initial>0 else 1
    months= len(user_array)
    if months>1 and initial>0:
        avg_mom= (multiple**(1/(months-1)))- 1
    else:
        avg_mom= 0.0

    return {
        "initial_users": initial,
        "final_users": final,
        "growth_multiple": multiple,
        "avg_mom_growth_rate": avg_mom
    }

def run_sensitivity_analysis(params: dict, param_ranges: dict):
    """
    Flash DNA placeholder from NEW(UI) code,
    possibly used for scenario range scanning. Not fully implemented here.
    """
    return []

def cohort_retention_projection(base_retention: List[float],
                               cohort_size: int=1000,
                               months: int=12,
                               improvement_rate: float=0.02) -> Dict[str,Any]:
    """
    Flash DNA analysis approach: 
    Project future retention based on base retention curve + monthly improvement.
    """
    if len(base_retention)<2:
        base_retention= [1.0, 0.7]

    while len(base_retention)< months:
        base_retention.append(base_retention[-1]*0.9)

    cohorts= []
    active_users= [0]* months

    for c in range(months):
        cohort= [0]* months
        improvement= (1+ improvement_rate)** c
        for m in range(c, months):
            idx= m- c
            if idx< len(base_retention):
                ret= min(1.0, base_retention[idx]* improvement)
                cohort[m]= cohort_size* ret
        cohorts.append(cohort)
        for m in range(months):
            active_users[m]+= cohort[m]

    base_ltv= sum(base_retention)* cohort_size
    last_cohort= cohorts[-1]
    improved_ltv= sum(last_cohort)
    ltv_improvement= (improved_ltv/ base_ltv- 1)*100 if base_ltv>0 else 0

    return {
        "cohorts": cohorts,
        "active_users": active_users,
        "base_ltv": base_ltv,
        "improved_ltv": improved_ltv,
        "ltv_improvement": ltv_improvement
    }
