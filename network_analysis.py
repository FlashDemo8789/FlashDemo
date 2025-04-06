import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import networkx as nx

class NetworkEffectAnalyzer:
    """
    analyze network effects in user data or marketplace interactions
    """

    def __init__(self):
        pass

    def analyze_network_effects(self,
                                user_data: Optional[pd.DataFrame]=None,
                                interaction_data: Optional[pd.DataFrame]=None,
                                company_data: Dict[str,Any]=None)-> Dict[str,Any]:
        if not isinstance(user_data, pd.DataFrame) or not isinstance(interaction_data, pd.DataFrame):
            return self._classify_network_effects(company_data)

        metrics={}
        metrics["network_density"]= self._calculate_network_density(interaction_data)
        metrics["viral_coefficient"]= self._calculate_viral_coefficient(user_data)
        metrics["interaction_frequency"]= self._calculate_interaction_frequency(interaction_data)
        metrics["power_user_percentage"]= self._calculate_power_users(interaction_data)

        if company_data.get("business_model","").lower()=="marketplace":
            metrics["cross_side_effects"]= self._calculate_cross_side_effects(user_data, interaction_data)
        else:
            metrics["cross_side_effects"]= 0

        metrics["network_strength_score"]= self._calculate_network_strength(metrics)
        metrics["predictions"]= self._generate_network_predictions(metrics, company_data)
        metrics["recommendations"]= self._generate_network_recommendations(metrics, company_data)
        return metrics

    def _classify_network_effects(self, cdata: Dict[str,Any])-> Dict[str,Any]:
        bm= cdata.get("business_model","").lower()
        sec= cdata.get("sector","").lower()
        nt= "none"
        if bm in ["marketplace","platform"]:
            nt= "two-sided"
        elif bm in ["social","community"]:
            nt= "direct"
        elif bm in ["saas","software"] and sec in ["collaboration","communication"]:
            nt= "direct"
        elif bm=="data_network":
            nt= "data"

        refer= cdata.get("referral_rate",0)
        conv= cdata.get("conversion_rate",0.1)
        viral= refer* conv
        gr= cdata.get("user_growth_rate",0)
        n_strength= 0
        if nt!="none":
            if gr>0.2:
                n_strength=80
            elif gr>0.1:
                n_strength=60
            elif gr>0.05:
                n_strength=40
            else:
                n_strength=20
        defensible= (nt!="none" and n_strength>50 and cdata.get("churn_rate",1)<0.1)

        return {
            "network_type": nt,
            "viral_coefficient": viral,
            "network_strength_score": n_strength,
            "defensible_network": defensible,
            "predictions": self._generate_network_predictions({"network_strength_score":n_strength}, cdata),
            "recommendations": self._generate_network_recommendations({"network_strength_score":n_strength}, cdata)
        }

    def _calculate_network_density(self, interactions: pd.DataFrame)-> float:
        try:
            if 'user_id_from' in interactions.columns and 'user_id_to' in interactions.columns:
                G= nx.from_pandas_edgelist(interactions, 'user_id_from','user_id_to', create_using=nx.DiGraph())
                return nx.density(G)
            else:
                unique_users= len(pd.concat([interactions['user_id_from'], interactions['user_id_to']]).unique())
                total_interactions= len(interactions)
                pot= unique_users*(unique_users-1)
                if pot>0:
                    return total_interactions/ pot
        except:
            pass
        return 0.05

    def _calculate_viral_coefficient(self, user_data: pd.DataFrame)-> float:
        if 'referred_by' in user_data.columns:
            try:
                referred_count= user_data['referred_by'].notna().sum()
                referring_users= user_data['referred_by'].dropna().unique()
                referring_count= len(referring_users)
                if referring_count>0:
                    return referred_count/ referring_count
            except:
                pass
        return 0.2

    def _calculate_interaction_frequency(self, interactions: pd.DataFrame)-> float:
        try:
            if 'user_id_from' in interactions.columns and 'date' in interactions.columns:
                byday= interactions.groupby(['user_id_from', pd.Grouper(key='date', freq='D')]).size()
                return float(byday.mean())
        except:
            pass
        return 1.0

    def _calculate_power_users(self, interactions: pd.DataFrame)-> float:
        try:
            if 'user_id_from' in interactions.columns:
                counts= interactions.groupby('user_id_from').size().sort_values(ascending=False)
                total_users= len(counts)
                if total_users==0: return 0
                power_cut= max(1, int(total_users*0.2))
                power_interactions= counts.iloc[:power_cut].sum()
                all_interactions= counts.sum()
                if all_interactions>0:
                    return power_interactions/ all_interactions
        except:
            pass
        return 0.6

    def _calculate_cross_side_effects(self, user_data: pd.DataFrame, interactions: pd.DataFrame)-> float:
        try:
            if 'user_type' in user_data.columns and 'user_id_from' in interactions.columns and 'user_id_to' in interactions.columns:
                merged= interactions.merge(
                    user_data[['user_id','user_type']],
                    left_on='user_id_from', right_on='user_id',
                    how='left'
                ).rename(columns={'user_type':'from_type'})
                merged= merged.merge(
                    user_data[['user_id','user_type']],
                    left_on='user_id_to', right_on='user_id',
                    how='left'
                ).rename(columns={'user_type':'to_type'})
                cross_side= merged[ merged['from_type']!= merged['to_type'] ]
                return len(cross_side)/ len(merged) if len(merged)>0 else 0
        except:
            pass
        return 0.5

    def _calculate_network_strength(self, mets: Dict[str,float])-> float:
        score= 0
        tw= 0
        if 'network_density' in mets:
            score+= mets['network_density']*100* 0.2
            tw+= 0.2
        if 'viral_coefficient' in mets:
            vc= min(100, mets['viral_coefficient']*100)
            score+= vc* 0.3
            tw+= 0.3
        if 'interaction_frequency' in mets:
            ifreq= min(100, mets['interaction_frequency']*20)
            score+= ifreq* 0.2
            tw+= 0.2
        if 'power_user_percentage' in mets:
            p= mets['power_user_percentage']*100
            score+= p* 0.15
            tw+= 0.15
        if 'cross_side_effects' in mets:
            c= mets['cross_side_effects']*100
            score+= c* 0.15
            tw+= 0.15
        if tw>0:
            return score/ tw
        return 50

    def _generate_network_predictions(self, mets: Dict[str,float], cdata: Dict[str,Any])-> List[str]:
        preds=[]
        vc= mets.get('viral_coefficient',0)
        net_strength= mets.get('network_strength_score',0)
        if vc>1:
            preds.append("K>1 => exponential user growth possible.")
        elif vc>0.5:
            preds.append("Significant WOM growth but not fully viral.")
        else:
            preds.append("Growth depends on marketing => limited viral effect.")
        if net_strength>75:
            preds.append("Strong network => significantly defensible.")
        elif net_strength>50:
            preds.append("Moderate network => partially defensible.")
        else:
            preds.append("Limited network => advantage must come from product or brand.")
        if net_strength>60:
            preds.append("Value/user likely to scale with network => improved unit economics.")
        else:
            preds.append("No clear evidence of inc. per-user value w/ scale.")
        return preds

    def _generate_network_recommendations(self, mets: Dict[str,float], cdata: Dict[str,Any])-> List[str]:
        recs=[]
        ns= mets.get('network_strength_score',0)
        vc= mets.get('viral_coefficient',0)
        if ns<30:
            recs.append("Redesign product loops for user interactions & network formation.")
        if vc<0.3:
            recs.append("Implement referral program with strong incentives.")
        bm= cdata.get('business_model','').lower()
        if bm=="marketplace":
            recs.append("Focus on liquidity in core segments before expansion.")
        elif bm=="platform":
            recs.append("Encourage partner ecosystem => reduce dev friction.")
        elif bm in ["social","community"]:
            recs.append("Optimize onboarding => immediate value pre-network scale.")
        recs.append("Measure & optimize activation & retention specifically for referred users.")
        return recs
