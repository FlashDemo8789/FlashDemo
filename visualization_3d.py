import numpy as np
import pandas as pd
from typing import List, Dict, Any

class InteractiveVisualization:
    """
    data generator for 3D scenario or network visuals
    Merged with minimal “NEW(UI)” references if any.
    """

    def generate_scenario_visualization(self, scenario_data: List[Dict[str,Any]])-> Dict[str,Any]:
        if not scenario_data or not isinstance(scenario_data, list):
            return self._generate_sample_scenario_data()
        try:
            df= pd.DataFrame(scenario_data)
        except:
            return self._generate_sample_scenario_data()
        if 'final_users' not in df.columns:
            df['final_users']=1000
        if 'success_probability' not in df.columns:
            df['success_probability']=0.5
        params= [c for c in df.columns if c not in ['final_users','success_probability']]
        corrs={}
        for p in params:
            corrs[p]= abs(df[p].corr(df['final_users']))
        top= sorted(corrs.items(), key=lambda x:x[1], reverse=True)[:3]
        top_params= [tp[0] for tp in top]
        while len(top_params)<3:
            for p in ['churn_rate','referral_rate','user_growth_rate']:
                if p not in top_params:
                    top_params.append(p)
                    break
        top_params= top_params[:3]
        return {
            'type':'3d_scatter',
            'x': {'name': top_params[0], 'values': df[top_params[0]].tolist()},
            'y': {'name': top_params[1], 'values': df[top_params[1]].tolist()},
            'z': {'name': top_params[2], 'values': df[top_params[2]].tolist()},
            'color': {'name': 'final_users', 'values': df['final_users'].tolist()},
            'size': {'name': 'success_probability', 'values': df['success_probability'].tolist()}
        }

    def generate_cohort_visualization(self, cohort_data: Any)-> Dict[str,Any]:
        if not cohort_data:
            return self._generate_sample_cohort_data()
        ret= None
        if hasattr(cohort_data,'retention'):
            ret= cohort_data.retention
        elif isinstance(cohort_data,dict) and 'retention' in cohort_data:
            ret= cohort_data['retention']
        if ret is None or not isinstance(ret, pd.DataFrame):
            return self._generate_sample_cohort_data()
        cohorts= ret.index.astype(str).tolist()
        periods= [str(c) for c in ret.columns.tolist()]
        x=[]
        y=[]
        z=[]
        for ci, cval in enumerate(cohorts):
            for pi, pval in enumerate(periods):
                x.append(ci)
                y.append(pi)
                z.append(float(ret.iloc[ci,pi]))
        return {
            'type':'3d_surface',
            'x': {'name':'Cohort','values': x,'labels': cohorts},
            'y': {'name':'Period','values': y,'labels': periods},
            'z': {'name':'Retention(%)','values': z}
        }

    def generate_network_visualization(self, interaction_data: pd.DataFrame=None)-> Dict[str,Any]:
        if not isinstance(interaction_data,pd.DataFrame):
            return self._generate_sample_network_data()
        try:
            from_users= interaction_data['user_id_from'].unique()
            to_users= interaction_data['user_id_to'].unique()
            all_users= np.union1d(from_users, to_users)
            nodes=[]
            for uid in all_users:
                outgoing= interaction_data[ interaction_data['user_id_from']== uid ].shape[0]
                incoming= interaction_data[ interaction_data['user_id_to']== uid ].shape[0]
                sz= float(np.log1p(outgoing+ incoming)+ 1)
                clr= (incoming/(outgoing+ incoming)) if (outgoing+ incoming)>0 else 0.5
                nodes.append({'id':str(uid),'size': sz, 'color': float(clr)})

            edges=[]
            for _, row in interaction_data.iterrows():
                edges.append({
                    'source': str(row['user_id_from']),
                    'target': str(row['user_id_to']),
                    'weight': 1
                })
            edict={}
            for e in edges:
                k= f"{e['source']}_{e['target']}"
                if k in edict:
                    edict[k]['weight']+= e['weight']
                else:
                    edict[k]= e
            uniq_edges= list(edict.values())
            return {
                'type':'network_graph',
                'nodes': nodes,
                'edges': uniq_edges
            }
        except Exception as e:
            print(f"Error => {str(e)}")
            return self._generate_sample_network_data()

    def _generate_sample_scenario_data(self)-> Dict[str,Any]:
        import numpy as np
        churn_vals= np.linspace(0.02,0.2,10)
        referral_vals= np.linspace(0.01,0.1,10)
        x=[]
        y=[]
        z=[]
        color=[]
        size=[]
        for c in churn_vals:
            for r in referral_vals:
                g= 0.1
                users= 1000*(1+ (g+ r- c)*12)
                sp= min(1.0, max(0.0,0.5+ 0.5*(r/0.1- c/0.1)))
                x.append(float(c))
                y.append(float(r))
                z.append(float(g))
                color.append(float(users))
                size.append(float(sp))
        return {
            'type':'3d_scatter',
            'x':{'name':'churn_rate','values': x},
            'y':{'name':'referral_rate','values': y},
            'z':{'name':'growth_rate','values': z},
            'color':{'name':'final_users','values': color},
            'size':{'name':'success_probability','values': size}
        }

    def _generate_sample_cohort_data(self)-> Dict[str,Any]:
        cohorts= [f"2023-{m:02d}" for m in range(1,7)]
        periods= [str(p) for p in range(6)]
        x=[]
        y=[]
        z=[]
        for ci,cval in enumerate(cohorts):
            base= 100- ci*5
            for pi,pval in enumerate(periods):
                ret= base*(0.85** pi)
                x.append(ci)
                y.append(pi)
                z.append(ret)
        return {
            'type':'3d_surface',
            'x':{'name':'Cohort','values': x,'labels': cohorts},
            'y':{'name':'Period','values': y,'labels': periods},
            'z':{'name':'Retention(%)','values': z}
        }

    def _generate_sample_network_data(self)-> Dict[str,Any]:
        import numpy as np
        nodes=[]
        for i in range(30):
            sz= np.random.uniform(1,5)
            clr= np.random.uniform(0,1)
            nodes.append({'id': f"user_{i}", 'size': float(sz), 'color': float(clr)})
        edges=[]
        for i in range(50):
            s= np.random.randint(0,30)
            t= np.random.randint(0,30)
            if s!= t:
                edges.append({
                    'source': f"user_{s}",
                    'target': f"user_{t}",
                    'weight': float(np.random.uniform(0.5,2.0))
                })
        return {
            'type':'network_graph',
            'nodes': nodes,
            'edges': edges
        }
