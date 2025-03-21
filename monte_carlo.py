import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from config import MC_SIMULATION_RUNS

@dataclass
class SimulationResult:
    success_probability: float
    percentiles: Dict[str, Dict[int,float]]
    sensitivity: Dict[str,float]
    scenarios: pd.DataFrame
    correlation_matrix: pd.DataFrame

class MonteCarloSimulator:
    """
    HPC synergy BFS–free => performs Monte Carlo simulations for risk assessment
    """

    def __init__(self, num_simulations: int= MC_SIMULATION_RUNS):
        self.num_simulations= num_simulations

    def run_simulation(self, startup_data: Dict[str,Any]) -> SimulationResult:
        base_params= self._extract_base_parameters(startup_data)
        param_dists= self._define_parameter_distributions(base_params)
        param_samples= self._generate_parameter_samples(param_dists)
        outcomes= self._calculate_simulation_outcomes(param_samples, startup_data)
        return self._analyze_simulation_results(outcomes, param_samples)

    def _extract_base_parameters(self, sd: Dict[str,Any]) -> Dict[str,float]:
        return {
            "monthly_revenue": sd.get("monthly_revenue",50000),
            "burn_rate": sd.get("burn_rate",30000),
            "user_growth_rate": sd.get("user_growth_rate",0.1),
            "churn_rate": sd.get("churn_rate",0.05),
            "cac": sd.get("customer_acquisition_cost",300),
            "arpu": sd.get("avg_revenue_per_user",100),
            "gross_margin": sd.get("gross_margin_percent",70)/100 if sd.get("gross_margin_percent",70)>1 else sd.get("gross_margin_percent",0.7),
            "current_cash": sd.get("current_cash",0.5)*1_000_000,
            "monthly_active_users": sd.get("monthly_active_users",1000),
            "referral_rate": sd.get("referral_rate",0.02)
        }

    def _define_parameter_distributions(self, base: Dict[str,float]) -> Dict[str,Dict[str,Any]]:
        return {
            "monthly_revenue": {
                "dist":"normal",
                "mean": base["monthly_revenue"],
                "std": base["monthly_revenue"]* 0.2
            },
            "burn_rate": {
                "dist":"normal",
                "mean": base["burn_rate"],
                "std": base["burn_rate"]* 0.1
            },
            "user_growth_rate": {
                "dist":"beta",
                "a": 2,
                "b": max(0.1,(2/base["user_growth_rate"])-2) if base["user_growth_rate"]>0 else 20
            },
            "churn_rate": {
                "dist":"beta",
                "a":2,
                "b": max(0.1,(2/base["churn_rate"])-2) if base["churn_rate"]>0 else 40
            },
            "cac": {
                "dist":"lognormal",
                "mean": np.log(max(1, base["cac"])),
                "std": 0.3
            },
            "arpu": {
                "dist":"normal",
                "mean": base["arpu"],
                "std": base["arpu"]* 0.15
            },
            "gross_margin": {
                "dist":"beta",
                "a": base["gross_margin"]* 10,
                "b": (1- base["gross_margin"])* 10
            },
            "referral_rate": {
                "dist":"beta",
                "a":2,
                "b": max(0.1,(2/base["referral_rate"])-2) if base["referral_rate"]>0 else 100
            }
        }

    def _generate_parameter_samples(self, param_dists: Dict[str,Dict[str,Any]])-> Dict[str, np.ndarray]:
        samples= {}
        for param, info in param_dists.items():
            d= info["dist"]
            if d=="normal":
                samples[param]= np.random.normal(info["mean"], info["std"], self.num_simulations)
            elif d=="beta":
                samples[param]= np.random.beta(info["a"], info["b"], self.num_simulations)
            elif d=="lognormal":
                samples[param]= np.random.lognormal(info["mean"], info["std"], self.num_simulations)
            elif d=="uniform":
                samples[param]= np.random.uniform(info["min"], info["max"], self.num_simulations)
        return samples

    def _calculate_simulation_outcomes(self, param_samples: Dict[str, np.ndarray], sd: Dict[str,Any]) -> Dict[str, np.ndarray]:
        n= self.num_simulations
        runway= np.zeros(n)
        final_users= np.zeros(n)
        ltv_cac= np.zeros(n)
        success_prob= np.zeros(n)
        year3_rev= np.zeros(n)
        year5_rev= np.zeros(n)
        profitable_month= np.zeros(n)

        stage= sd.get("stage","seed").lower()
        sector= sd.get("sector","saas").lower()

        for i in range(n):
            br= param_samples["burn_rate"][i]
            rev= param_samples["monthly_revenue"][i]
            net= rev- br
            cash= sd.get("current_cash",0.5)*1_000_000
            if net<=0:
                runway[i]= float('inf')
            else:
                runway[i]= cash/ net

            g= param_samples["user_growth_rate"][i]
            c= param_samples["churn_rate"][i]
            r= param_samples["referral_rate"][i]
            users= sd.get("monthly_active_users",1000)
            monthly_profit= net
            is_profitable= False

            rev_path= []
            cost_path= []
            user_path= []
            rev_month= rev
            cost_month= br

            for m in range(60):
                new_users= users* g
                churned= users* c
                referred= users* r
                users= users + new_users + referred - churned
                user_path.append(users)

                rev_month*= (1+ g)
                cost_month*= (1+ g* 0.7)
                monthly_profit= rev_month- cost_month

                rev_path.append(rev_month)
                cost_path.append(cost_month)
                if not is_profitable and monthly_profit>0:
                    is_profitable= True
                    profitable_month[i]= m+1

            final_users[i]= user_path[-1] if user_path else 0
            if len(rev_path)>36:
                year3_rev[i]= rev_path[36]* 12
            else:
                year3_rev[i]= rev_path[-1]* 12
            if len(rev_path)>59:
                year5_rev[i]= rev_path[59]* 12
            else:
                year5_rev[i]= rev_path[-1]* 12

            cac= param_samples["cac"][i]
            arpu= param_samples["arpu"][i]
            gm= param_samples["gross_margin"][i]
            if c<=0:
                c=0.01
            mo_contribution= arpu* gm
            ltv_val= mo_contribution/ c
            ratio= ltv_val/ max(1,cac)
            ltv_cac[i]= ratio

            factors= [
                min(1, runway[i]/18)*0.2,
                min(1, ratio/3)*0.3,
                min(1, g/ 0.1)*0.2,
                (1- min(1, profitable_month[i]/36))*0.2,
                min(1, final_users[i]/ 50_000)*0.1
            ]
            if sector in ["saas","software"]:
                factors[2]*=1.2
                factors[3]*=0.8
            elif sector in ["biotech","hardware"]:
                factors[0]*=1.5
                factors[2]*=0.7

            if stage in ["seed","pre-seed"]:
                factors[2]*=1.3
                factors[3]*=0.7
            elif stage in ["series-b","series-c","growth"]:
                factors[3]*=1.3

            sp= sum(factors)*100
            success_prob[i]= sp

        return {
            "runway": runway,
            "final_users": final_users,
            "ltv_cac_ratio": ltv_cac,
            "success_probability": success_prob,
            "year3_revenue": year3_rev,
            "year5_revenue": year5_rev,
            "profitable_month": profitable_month
        }

    def _analyze_simulation_results(self, outcomes: Dict[str,np.ndarray], param_samples: Dict[str,np.ndarray]) -> SimulationResult:
        success_prob= np.mean(outcomes["success_probability"])
        metrics= list(outcomes.keys())
        percentiles= {m:{} for m in metrics}
        for m in metrics:
            for p in [10,25,50,75,90]:
                percentiles[m][p]= float(np.percentile(outcomes[m], p))

        sensitivity={}
        sp_array= outcomes["success_probability"]
        for param, arr in param_samples.items():
            corr= np.corrcoef(arr, sp_array)[0,1]
            sensitivity[param]= corr

        df= pd.DataFrame(param_samples)
        for m in metrics:
            df[m]= outcomes[m]
        corr_mat= df.corr()

        return SimulationResult(
            success_probability= success_prob,
            percentiles= percentiles,
            sensitivity= dict(sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)),
            scenarios= df,
            correlation_matrix= corr_mat
        )
