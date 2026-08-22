import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from nexusrag.backend.ingestion.chunker import DocumentChunk

@dataclass
class Entity:
    id: str
    name: str
    type: str # Policy, Regulation, Department, Process, Requirement, System, Version, Organization
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relation:
    source_id: str
    target_id: str
    type: str # POLICY_APPLIES_TO, POLICY_REQUIRES, REGULATION_AFFECTS, DOCUMENT_REFERENCES, SUPERSEDES, CONFLICTS_WITH, REQUIREMENT_DEFINED_BY
    properties: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraphExtractor:
    @staticmethod
    def extract_from_chunks(chunks: List[DocumentChunk]) -> Tuple[List[Entity], List[Relation]]:
        entities: Dict[str, Entity] = {}
        relations: List[Relation] = []

        def get_or_add_entity(name: str, e_type: str, props: Dict[str, Any] = None) -> Entity:
            eid = f'{e_type}:{name}'.strip()
            if eid not in entities:
                entities[eid] = Entity(id=eid, name=name, type=e_type, properties=props or {})
            elif props:
                entities[eid].properties.update(props)
            return entities[eid]

        # Standard Organizations
        nexuscorp = get_or_add_entity('NexusCorp', 'Organization')

        for c in chunks:
            text = c.content
            text_lower = text.lower()
            doc_name = c.doc_name
            doc_meta = c.metadata
            version = str(c.version)
            year = str(c.year)
            dept = str(c.department)

            # 1. Document & Policy Entities
            if '2025' in doc_name:
                pol2025 = get_or_add_entity('Employee Operations Policy 2025', 'Policy', {'version': '1.0', 'year': '2025', 'doc': doc_name})
                relations.append(Relation(pol2025.id, nexuscorp.id, 'POLICY_APPLIES_TO', {'scope': 'Enterprise'}))
            
            if '2026' in doc_name and 'Employee' in doc_name:
                pol2026 = get_or_add_entity('Employee Operations Policy 2026', 'Policy', {'version': '2.0', 'year': '2026', 'doc': doc_name})
                relations.append(Relation(pol2026.id, nexuscorp.id, 'POLICY_APPLIES_TO', {'scope': 'Enterprise'}))
                
                # Check for supersedes
                if 'Employee Operations Policy 2025' in [e.name for e in entities.values()]:
                    pol2025 = get_or_add_entity('Employee Operations Policy 2025', 'Policy')
                    relations.append(Relation(pol2026.id, pol2025.id, 'SUPERSEDES', {'reason': 'Annual Policy Revision 2026'}))

            if 'SR402' in doc_name or 'sr-402' in doc_name.lower():
                reg_sr402 = get_or_add_entity('Regulation SR-402', 'Regulation', {'version': '3.1', 'year': '2026', 'authority': 'Global InfoSec Board'})
                relations.append(Relation(reg_sr402.id, nexuscorp.id, 'REGULATION_AFFECTS', {'scope': 'Cybersecurity & Data Protection'}))

            if 'IT_Infrastructure' in doc_name:
                it_manual = get_or_add_entity('IT Infrastructure Manual v3.4', 'Technical Manual', {'version': '3.4', 'year': '2026'})

            # Department Entity
            if dept and dept != 'Enterprise Operations':
                dept_ent = get_or_add_entity(dept, 'Department')

            # 2. Extract Specific Requirements & Processes
            # Attendance Requirement 75%
            if '75%' in text:
                req_75 = get_or_add_entity('75% On-Site Attendance Requirement', 'Requirement', {'threshold': '75%', 'policy': '2025'})
                pol2025 = get_or_add_entity('Employee Operations Policy 2025', 'Policy')
                relations.append(Relation(pol2025.id, req_75.id, 'POLICY_REQUIRES', {'penalty': 'HR Automated Warning'}))

            # Attendance Requirement 60%
            if '60%' in text:
                req_60 = get_or_add_entity('60% On-Site Attendance Requirement', 'Requirement', {'threshold': '60%', 'policy': '2026'})
                pol2026 = get_or_add_entity('Employee Operations Policy 2026', 'Policy')
                relations.append(Relation(pol2026.id, req_60.id, 'POLICY_REQUIRES', {'penalty': 'Collaborative Review'}))
                if 'Requirement:75% On-Site Attendance Requirement' in entities:
                    relations.append(Relation(req_60.id, 'Requirement:75% On-Site Attendance Requirement', 'CONFLICTS_WITH', {'type': 'Threshold Change (75% -> 60%)'}))

            # Remote Work Process
            if 'remote work' in text_lower or 'remote working' in text_lower:
                proc_remote = get_or_add_entity('Remote Work Authorization Process', 'Process')
                if '2025' in doc_name:
                    pol2025 = get_or_add_entity('Employee Operations Policy 2025', 'Policy')
                    relations.append(Relation(pol2025.id, proc_remote.id, 'POLICY_APPLIES_TO', {'limit': '1 day/week', 'approver': 'Department Manager'}))
                if '2026' in doc_name and 'Employee' in doc_name:
                    pol2026 = get_or_add_entity('Employee Operations Policy 2026', 'Policy')
                    relations.append(Relation(pol2026.id, proc_remote.id, 'POLICY_APPLIES_TO', {'limit': '3 days/week', 'approver': 'Department Director'}))

            # Process X (Incident Response & Server Provisioning)
            if 'process x' in text_lower:
                proc_x = get_or_add_entity('Process X (Incident Response & Provisioning)', 'Process')
                reg_sr402 = get_or_add_entity('Regulation SR-402', 'Regulation')
                relations.append(Relation(reg_sr402.id, proc_x.id, 'REGULATION_AFFECTS', {'mandate': '2-Hour Incident Notification & Audit Logging'}))
                
                req_sla = get_or_add_entity('15-Min Triage & 1-Hr Escalation SLA', 'Requirement')
                relations.append(Relation(proc_x.id, req_sla.id, 'REQUIREMENT_DEFINED_BY', {'standard': 'Sev-1/Sev-2 Protocol'}))

            # MFA and Encryption Requirements
            if 'mfa' in text_lower or 'multi-factor' in text_lower:
                req_mfa = get_or_add_entity('Mandatory Hardware MFA (SEC-01)', 'Requirement', {'standard': 'Hardware Key / Authenticator'})
                reg_sr402 = get_or_add_entity('Regulation SR-402', 'Regulation')
                relations.append(Relation(reg_sr402.id, req_mfa.id, 'REQUIREMENT_DEFINED_BY', {'rule': 'SMS 2FA Prohibited'}))
                
                proc_remote = get_or_add_entity('Remote Work Authorization Process', 'Process')
                relations.append(Relation(proc_remote.id, req_mfa.id, 'POLICY_REQUIRES', {'enforcement': 'Mandatory for all remote logins'}))

            if 'aes-256' in text_lower:
                req_enc = get_or_add_entity('AES-256 & TLS 1.3 Encryption (SEC-02)', 'Requirement')
                reg_sr402 = get_or_add_entity('Regulation SR-402', 'Regulation')
                relations.append(Relation(reg_sr402.id, req_enc.id, 'REQUIREMENT_DEFINED_BY', {'scope': 'Data at rest & transit'}))

            # 7-Year Retention Requirement
            if '7 years' in text_lower:
                req_ret = get_or_add_entity('7-Year Security Audit Log Retention (SEC-03)', 'Requirement')
                reg_sr402 = get_or_add_entity('Regulation SR-402', 'Regulation')
                relations.append(Relation(reg_sr402.id, req_ret.id, 'REQUIREMENT_DEFINED_BY', {'storage': 'Immutable Cloud Storage'}))

            # 2-Hour Incident Reporting
            if '2 hours' in text_lower or '2-hour' in text_lower:
                req_rep = get_or_add_entity('2-Hour Breach Reporting Mandate (SEC-04)', 'Requirement')
                reg_sr402 = get_or_add_entity('Regulation SR-402', 'Regulation')
                relations.append(Relation(reg_sr402.id, req_rep.id, 'REQUIREMENT_DEFINED_BY', {'penalty': 'Tier-3 Escalation'}))

            # Cross references between 2026 policy and SR-402
            if 'sr-402' in text_lower or 'sr402' in text_lower:
                if '2026' in doc_name and 'Employee' in doc_name:
                    pol2026 = get_or_add_entity('Employee Operations Policy 2026', 'Policy')
                    reg_sr402 = get_or_add_entity('Regulation SR-402', 'Regulation')
                    relations.append(Relation(pol2026.id, reg_sr402.id, 'DOCUMENT_REFERENCES', {'clause': 'Section 3 Remote Work Compliance'}))

        # Deduplicate relations
        unique_rel_map = {}
        for r in relations:
            key = f'{r.source_id}->{r.type}->{r.target_id}'
            if key not in unique_rel_map:
                unique_rel_map[key] = r
                
        return list(entities.values()), list(unique_rel_map.values())
