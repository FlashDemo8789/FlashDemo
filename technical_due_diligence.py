from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TechStackItem:
    name: str
    category: str
    maturity: float
    scalability: float
    market_adoption: float
    expertise_required: float

@dataclass
class TechnicalAssessment:
    overall_score: float
    architecture_score: float
    scalability_score: float
    tech_debt_score: float
    tech_stack: List[TechStackItem]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    risk_assessment: Dict[str,Any]

class TechnicalDueDiligence:
    """
    Performs technical due diligence on startup architecture & codebase
    """

    def __init__(self):
        self.tech_stack_database = self._load_tech_stack_database()

    def assess_technical_architecture(self, tech_data: Dict[str,Any]) -> TechnicalAssessment:
        stack = self._extract_tech_stack(tech_data)
        arch_score = self._score_architecture(tech_data, stack)
        scale_score = self._score_scalability(tech_data, stack)
        debt_score = self._score_tech_debt(tech_data)
        overall = arch_score*0.4 + scale_score*0.4 + debt_score*0.2
        strengths = self._identify_strengths(tech_data, stack, arch_score, scale_score, debt_score)
        weaknesses = self._identify_weaknesses(tech_data, stack, arch_score, scale_score, debt_score)
        recs = self._generate_recommendations(tech_data, stack, weaknesses)
        risk_assessment = self._assess_risks(tech_data, stack)
        return TechnicalAssessment(
            overall_score = overall,
            architecture_score = arch_score,
            scalability_score = scale_score,
            tech_debt_score = debt_score,
            tech_stack = stack,
            strengths = strengths,
            weaknesses = weaknesses,
            recommendations = recs,
            risk_assessment = risk_assessment
        )

    def _extract_tech_stack(self, td: Dict[str,Any]) -> List[TechStackItem]:
        stack = []
        if 'tech_stack' in td and isinstance(td['tech_stack'], list):
            for it in td['tech_stack']:
                if isinstance(it, dict) and 'name' in it:
                    name = it['name']
                    cat = it.get('category', self._categorize_tech(name))
                    stack.append(TechStackItem(
                        name = name,
                        category = cat,
                        maturity = it.get('maturity', self._get_tech_maturity(name)),
                        scalability = it.get('scalability', self._get_tech_scalability(name)),
                        market_adoption = it.get('market_adoption', self._get_tech_market_adoption(name)),
                        expertise_required = it.get('expertise_required', self._get_tech_expertise_required(name))
                    ))
        elif 'tech_description' in td and isinstance(td['tech_description'], str):
            import re
            desc = td['tech_description']
            patterns = [
                r'using ([A-Za-z0-9#+.\-]+)',
                r'built with ([A-Za-z0-9#+.\-]+)',
                r'([A-Za-z0-9#+.\-]+) for (?:backend|frontend|database)',
                r'([A-Za-z0-9#+.\-]+) (stack|framework|language|database)'
            ]
            found = set()
            for pat in patterns:
                matches = re.finditer(pat, desc, re.IGNORECASE)
                for m in matches:
                    nm = m.group(1).strip()
                    if nm and len(nm)>1:
                        found.add(nm)
            for nm in found:
                cat = self._categorize_tech(nm)
                stack.append(TechStackItem(
                    name = nm,
                    category = cat,
                    maturity = self._get_tech_maturity(nm),
                    scalability = self._get_tech_scalability(nm),
                    market_adoption = self._get_tech_market_adoption(nm),
                    expertise_required = self._get_tech_expertise_required(nm)
                ))
        if not stack:
            stack = self._generate_default_stack(td)
        return stack

    def _categorize_tech(self, n: str) -> str:
        n = n.lower()
        for cat, tch in self.tech_stack_database.items():
            for t in tch:
                if t.lower() == n or t.lower() in n:
                    return cat
        if any(db in n for db in ['sql', 'db', 'database', 'mongo', 'postgres', 'mysql', 'oracle']):
            return 'database'
        elif any(lang in n for lang in ['java', 'python', 'ruby', 'php', 'go', 'rust', 'c#', '.net', 'node', 'typescript', 'javascript', 'ts']):
            return 'language'
        elif any(fr in n for fr in ['react', 'vue', 'angular', 'svelte', 'ember']):
            return 'frontend'
        elif any(bk in n for bk in ['node', 'django', 'flask', 'rails', 'spring']):
            return 'backend'
        elif any(infra in n for infra in ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker']):
            return 'infrastructure'
        return 'other'

    def _generate_default_stack(self, tech_data: Dict[str,Any]) -> List[TechStackItem]:
        sector = tech_data.get('sector', '').lower()
        stack = []
        if sector in ['fintech', 'banking']:
            stack.append(TechStackItem('Java', 'language', 0.9, 0.8, 0.9, 0.7))
            stack.append(TechStackItem('Spring', 'backend', 0.8, 0.8, 0.8, 0.7))
            stack.append(TechStackItem('PostgreSQL', 'database', 0.9, 0.7, 0.9, 0.6))
            stack.append(TechStackItem('React', 'frontend', 0.8, 0.8, 0.9, 0.6))
            stack.append(TechStackItem('AWS', 'infrastructure', 0.9, 0.9, 0.9, 0.7))
        elif sector in ['ecommerce', 'retail']:
            stack.append(TechStackItem('PHP', 'language', 0.8, 0.6, 0.7, 0.5))
            stack.append(TechStackItem('MySQL', 'database', 0.9, 0.7, 0.9, 0.6))
            stack.append(TechStackItem('React', 'frontend', 0.8, 0.8, 0.9, 0.6))
            stack.append(TechStackItem('Redis', 'cache', 0.8, 0.8, 0.8, 0.6))
            stack.append(TechStackItem('AWS', 'infrastructure', 0.9, 0.9, 0.9, 0.7))
        elif sector in ['saas', 'enterprise']:
            stack.append(TechStackItem('Python', 'language', 0.8, 0.7, 0.9, 0.5))
            stack.append(TechStackItem('Django', 'backend', 0.8, 0.7, 0.8, 0.6))
            stack.append(TechStackItem('PostgreSQL', 'database', 0.9, 0.7, 0.9, 0.6))
            stack.append(TechStackItem('React', 'frontend', 0.8, 0.8, 0.9, 0.6))
            stack.append(TechStackItem('Docker', 'infrastructure', 0.8, 0.8, 0.8, 0.7))
        else:
            stack.append(TechStackItem('JavaScript', 'language', 0.9, 0.7, 0.9, 0.5))
            stack.append(TechStackItem('Node.js', 'backend', 0.8, 0.7, 0.9, 0.6))
            stack.append(TechStackItem('MongoDB', 'database', 0.8, 0.8, 0.8, 0.6))
            stack.append(TechStackItem('React', 'frontend', 0.8, 0.8, 0.9, 0.6))
            stack.append(TechStackItem('AWS', 'infrastructure', 0.9, 0.9, 0.9, 0.7))
        return stack

    def _get_tech_maturity(self, name: str) -> float:
        n = name.lower()
        mature = ['java', 'python', 'javascript', 'mysql', 'postgres', 'mongodb', 'react', 'angular', 'aws', 'azure', 'php', 'ruby', '.net', 'django', 'laravel', 'spring']
        stable = ['go', 'kotlin', 'typescript', 'vue', 'flutter', 'graphql', 'kubernetes', 'docker', 'terraform', 'nextjs', 'nuxt', 'svelte']
        emerging = ['rust', 'webassembly', 'deno', 'elixir', 'solidity', 'dart', 'haskell', 'clojure']
        if any(m in n for m in mature):
            return 0.9
        elif any(s in n for s in stable):
            return 0.7
        elif any(e in n for e in emerging):
            return 0.4
        return 0.5

    def _get_tech_scalability(self, name: str) -> float:
        n = name.lower()
        sc = ['kubernetes', 'aws', 'azure', 'gcp', 'kafka', 'cassandra', 'elasticsearch', 'dynamodb', 'redis', 'go', 'rust', 'scala', 'graphql', 'grpc']
        lim = ['mysql', 'sqlite', 'wordpress', 'php', 'monolith', 'rails']
        if any(x in n for x in sc):
            return 0.9
        elif any(x in n for x in lim):
            return 0.4
        return 0.6

    def _get_tech_market_adoption(self, name: str) -> float:
        n = name.lower()
        wide = ['javascript', 'python', 'java', 'c#', 'react', 'angular', 'vue', 'aws', 'mysql', 'postgres', 'mongodb', 'docker', 'git', 'node']
        moderate = ['typescript', 'go', 'kotlin', 'graphql', 'redis', 'kubernetes', 'terraform', 'kafka', 'flutter', 'django', 'laravel']
        niche = ['elm', 'purescript', 'haskell', 'clojure', 'erlang', 'crystal', 'ocaml', 'webassembly', 'gleam', 'reason']
        if any(x in n for x in wide):
            return 0.9
        elif any(x in n for x in moderate):
            return 0.7
        elif any(x in n for x in niche):
            return 0.3
        return 0.5

    def _get_tech_expertise_required(self, name: str) -> float:
        n = name.lower()
        complex_tech = ['kubernetes', 'rust', 'haskell', 'scala', 'distributed systems', 'blockchain', 'machine learning', 'graphql', 'webassembly', 'microservices']
        moderate_tech = ['java', 'typescript', 'go', 'react', 'vue', 'angular', 'docker', 'redis', 'aws', 'azure', 'gcp', 'elasticsearch']
        accessible = ['javascript', 'html', 'css', 'python', 'ruby', 'php', 'mysql', 'wordpress']
        if any(x in n for x in complex_tech):
            return 0.9
        elif any(x in n for x in moderate_tech):
            return 0.6
        elif any(x in n for x in accessible):
            return 0.3
        return 0.5

    def _score_architecture(self, td: Dict[str,Any], stack: List[TechStackItem]) -> float:
        score = 0.7
        arch_type = td.get("architecture_type", "").lower()
        if 'microservice' in arch_type:
            score += 0.1
        elif 'monolith' in arch_type:
            score -= 0.1
        if stack:
            avg_mat = sum(it.maturity for it in stack) / len(stack)
            avg_adpt = sum(it.market_adoption for it in stack) / len(stack)
            score += (avg_mat - 0.5) * 0.2
            score += (avg_adpt - 0.5) * 0.1
        if td.get("reported_issues", 0) > 5:
            score -= 0.1
        if td.get("has_architecture_docs", False):
            score += 0.05
        return max(0.1, min(1.0, score))

    def _score_scalability(self, td: Dict[str,Any], stack: List[TechStackItem]) -> float:
        score = 0.6
        curr_users = td.get("current_users", 0)
        if curr_users > 100_000:
            score += 0.1
        elif curr_users > 10_000:
            score += 0.05
        if stack:
            avg_scal = sum(it.scalability for it in stack) / len(stack)
            score += (avg_scal - 0.5) * 0.3
        arch = td.get("architecture_type", "").lower()
        if 'microservice' in arch or 'serverless' in arch:
            score += 0.1
        elif 'monolith' in arch:
            score -= 0.1
        db_tech = next((x for x in stack if x.category == 'database'), None)
        if db_tech:
            if any(db in db_tech.name.lower() for db in ['nosql', 'dynamodb', 'cassandra', 'mongodb']):
                score += 0.05
            elif any(db in db_tech.name.lower() for db in ['sqlite', 'access']):
                score -= 0.1
        infra = next((x for x in stack if x.category == 'infrastructure'), None)
        if infra:
            score += (0.5 - infra.scalability) * 0.4
        else:
            score += 0.2
        return max(0.1, min(1.0, score))

    def _score_tech_debt(self, td: Dict[str,Any]) -> float:
        score = 0.5
        tc = td.get("test_coverage", 0)
        if tc > 80:
            score += 0.2
        elif tc > 60:
            score += 0.1
        elif tc < 20:
            score -= 0.2
        if td.get("has_code_reviews", False):
            score += 0.1
        if td.get("has_documentation", False):
            score += 0.05
        bug = td.get("open_bugs", 0)
        if bug > 100:
            score -= 0.2
        elif bug > 50:
            score -= 0.1
        ack = td.get("acknowledged_tech_debt", 0)
        if ack > 0:
            score += 0.05
            score -= min(0.2, ack/10)
        if td.get("regular_refactoring", False):
            score += 0.1
        return max(0.1, min(1.0, score))

    def _identify_strengths(self, td, stack, arch_sc, scal_sc, debt_sc) -> list:
        s = []
        if arch_sc > 0.7:
            if 'microservice' in td.get("architecture_type", "").lower():
                s.append("Well-designed microservice architecture enables scaling")
            elif 'serverless' in td.get("architecture_type", "").lower():
                s.append("Serverless => cost efficiency & auto scaling")
            else:
                s.append("Architecture => suitable for advanced scaling needs")
        mod_tech = [it for it in stack if it.maturity > 0.7 and it.market_adoption > 0.7]
        if len(mod_tech) >= 2:
            nm = ", ".join(it.name for it in mod_tech[:2])
            s.append(f"Modern tech => {nm} => good adoption & maturity")
        if scal_sc > 0.7:
            s.append("Good scaling potential => growth-ready infrastructure")
        if debt_sc > 0.7:
            if td.get("test_coverage", 0) > 70:
                s.append(f"High test coverage => {td.get('test_coverage', 0)}% => lowers regression risk")
            if td.get('has_code_reviews', False):
                s.append("Consistent code reviews => knowledge sharing")
            if td.get('regular_refactoring', False):
                s.append("Regular refactoring => well-managed tech debt")
        if not s:
            s.append("Technology stack is adequate for standard usage.")
        return s

    def _identify_weaknesses(self, td, stack, arch_sc, scal_sc, debt_sc) -> list:
        w = []
        if arch_sc < 0.5:
            if 'monolith' in td.get('architecture_type', '').lower():
                w.append("Monolithic architecture => may hamper scaling")
            else:
                w.append("Architecture => inconsistent or unplanned approach")
        niche_tech = [it for it in stack if it.market_adoption < 0.4]
        if len(niche_tech) >= 2:
            nm = ", ".join(it.name for it in niche_tech[:2])
            w.append(f"Reliance on niche => {nm} => hiring/support challenges")
        high_expert = [it for it in stack if it.expertise_required > 0.8]
        if len(high_expert) >= 2:
            nm = ", ".join(it.name for it in high_expert[:2])
            w.append(f"Complex stack => {nm} => specialized experts required")

        if scal_sc < 0.5:
            w.append("Major rework needed for large scale deployment")
        if debt_sc < 0.5:
            if td.get("test_coverage", 0) < 30:
                w.append(f"Low test coverage => {td.get('test_coverage', 0)}% => risk of regressions")
            if td.get("open_bugs", 0) > 50:
                w.append(f"Large backlog => {td.get('open_bugs', 0)} open bugs => quality issues")
            if not td.get("has_documentation", False):
                w.append("Lack of documentation => slower onboarding")
        if not w:
            w.append("Technical stack may need optimization as business grows.")
        return w

    def _generate_recommendations(self, td, stack, weaknesses) -> list:
        recs = []
        for weak in weaknesses:
            wl = weak.lower()
            if 'monolithic' in wl:
                recs.append("Migrate from monolith => microservice or modular approach")
            elif 'niche' in wl:
                recs.append("Plan to reduce reliance on niche => mainstream stack for hiring")
            elif 'test coverage' in wl:
                recs.append("Improve test coverage => enable safer code refactoring")
            elif 'open bugs' in wl:
                recs.append("Prioritize bug triage => allocate engineering capacity")
            elif 'documentation' in wl:
                recs.append("Establish doc standards => knowledge transfer")

        low_mat = [it for it in stack if it.maturity < 0.5]
        if low_mat:
            nms = ", ".join(it.name for it in low_mat[:2])
            recs.append(f"Monitor stability of {nms} => fallback if issues.")
        if td.get("current_users", 0) > 10000 or td.get("user_growth_rate", 0) > 0.2:
            recs.append("Load testing & performance tuning => prepare for user growth")
            if not any("kubernetes" in it.name.lower() for it in stack if it.category == 'infrastructure'):
                recs.append("Evaluate container orchestration => improved deployment & scaling")
        if td.get("acknowledged_tech_debt", 0) > 5:
            recs.append("Create tech debt roadmap => structured refactoring tasks")
        if not td.get('has_code_reviews', False):
            recs.append("Implement mandatory code review => knowledge sharing")
        if not td.get('ci_cd', False):
            recs.append("Add CI/CD pipeline => frequent, reliable deployments")
        while len(recs) > 5:
            recs.pop()
        if len(recs) < 3:
            recs.append("Document architecture decisions => align with tech roadmap")
            recs.append("Regular system health checks & performance reviews")
        return recs

    def _assess_risks(self, td, stack) -> Dict[str,Any]:
        r = {
            'scaling_risk': self._assess_scaling_risk(td, stack),
            'maintenance_risk': self._assess_maintenance_risk(td, stack),
            'security_risk': self._assess_security_risk(td, stack),
            'talent_risk': self._assess_talent_risk(td, stack),
            'vendor_lock_in_risk': self._assess_vendor_lock_in_risk(td, stack)
        }
        w = {
            'scaling_risk': 0.3,
            'maintenance_risk': 0.2,
            'security_risk': 0.2,
            'talent_risk': 0.2,
            'vendor_lock_in_risk': 0.1
        }
        overall = sum(r[k] * w[k] for k in r)
        r['overall_risk'] = overall
        return r

    def _assess_scaling_risk(self, td, stack) -> float:
        risk = 0.5
        arch = td.get('architecture_type', '').lower()
        if 'microservice' in arch or 'serverless' in arch:
            risk -= 0.1
        elif 'monolith' in arch:
            risk += 0.1
        db = next((it for it in stack if it.category == 'database'), None)
        if db:
            risk += (0.5 - db.scalability) * 0.4
        else:
            risk += 0.1
        infra = next((it for it in stack if it.category == 'infrastructure'), None)
        if infra:
            risk += (0.5 - infra.scalability) * 0.4
        else:
            risk += 0.2
        cu = td.get('current_users', 0)
        if cu > 100000:
            risk -= 0.2
        elif cu > 10000:
            risk -= 0.1
        return max(0.1, min(0.9, risk))

    def _assess_maintenance_risk(self, td, stack) -> float:
        risk = 0.5
        tc = td.get('test_coverage', 0)
        if tc > 80:
            risk -= 0.2
        elif tc > 60:
            risk -= 0.1
        elif tc < 30:
            risk += 0.2
        if td.get('has_code_reviews', False):
            risk -= 0.1
        else:
            risk += 0.1
        if td.get('has_documentation', False):
            risk -= 0.1
        else:
            risk += 0.1
        ack = td.get('acknowledged_tech_debt', 0)
        risk += min(0.2, ack/10)
        if stack:
            avg_mat = sum(it.maturity for it in stack) / len(stack)
            risk += (0.5 - avg_mat) * 0.3
        return max(0.1, min(0.9, risk))

    def _assess_security_risk(self, td, stack) -> float:
        risk = 0.5
        if td.get('has_security_testing', False):
            risk -= 0.2
        if td.get('compliance_requirements', []):
            if not td.get('has_security_testing', False):
                risk += 0.2
        for it in stack:
            if it.maturity < 0.3:
                risk += 0.05
            if it.category == 'database':
                risk += (0.5 - it.maturity) * 0.2
            nm = it.name.lower()
            if any(v in nm for v in ['wordpress', 'php', 'flash']):
                risk += 0.1
        inc = td.get('security_incidents', 0)
        risk += min(0.3, inc * 0.1)
        return max(0.1, min(0.9, risk))

    def _assess_talent_risk(self, td, stack) -> float:
        risk = 0.5
        if stack:
            avg_exp = sum(it.expertise_required for it in stack) / len(stack)
            risk += (avg_exp - 0.5) * 0.3
            avg_adp = sum(it.market_adoption for it in stack) / len(stack)
            risk += (0.5 - avg_adp) * 0.3
        tsize = td.get('engineering_team_size', 0)
        if tsize < 3:
            risk += 0.2
        elif tsize < 8:
            risk += 0.1
        kpd = td.get('key_person_dependencies', 0)
        risk += min(0.3, kpd * 0.1)
        if td.get('has_documentation', False):
            risk -= 0.1
        else:
            risk += 0.1
        return max(0.1, min(0.9, risk))

    def _assess_vendor_lock_in_risk(self, td, stack) -> float:
        risk = 0.4
        pcount = 0
        for it in stack:
            nm = it.name.lower()
            if any(prop in nm for prop in ['aws', 'azure', 'gcp', 'salesforce', 'oracle', 'sap']):
                pcount += 1
                if any(hi in nm for hi in ['lambda', 'dynamodb', 'cosmosdb', 'bigtable']):
                    risk += 0.1
        if pcount > 3:
            risk += 0.2
        elif pcount > 1:
            risk += 0.1
        cints = td.get('custom_integrations', 0)
        risk += min(0.2, cints * 0.05)
        if td.get('has_vendor_exit_strategy', False):
            risk -= 0.2
        return max(0.1, min(0.9, risk))

    def _load_tech_stack_database(self) -> Dict[str,List[str]]:
        return {
            'language': ['Java','Python','Ruby','PHP','Go','Rust','C#','JavaScript','TypeScript'],
            'backend': ['Node','Django','Flask','Rails','Spring','.NET','Express'],
            'frontend': ['React','Vue','Angular','Svelte','Ember'],
            'database': ['MySQL','PostgreSQL','MongoDB','Redis','Cassandra','Oracle'],
            'infrastructure': ['AWS','Azure','GCP','Kubernetes','Docker','Terraform']
        }