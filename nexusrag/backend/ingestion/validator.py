import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.xlsx', '.xls', '.csv', '.txt', '.md'}

class DocumentValidator:
    @staticmethod
    def validate_file(file_path: Path) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {'valid': False, 'error': f'File does not exist: {path}'}
        
        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return {'valid': False, 'error': f'Unsupported file extension: {ext}. Allowed: {ALLOWED_EXTENSIONS}'}
        
        size = path.stat().st_size
        if size == 0:
            return {'valid': False, 'error': 'File is empty (0 bytes).'}
        
        # Calculate SHA-256 Checksum
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        
        return {
            'valid': True,
            'file_name': path.name,
            'file_path': str(path),
            'extension': ext,
            'size_bytes': size,
            'checksum': sha256.hexdigest()
        }

    @staticmethod
    def infer_metadata_from_name(file_name: str) -> Dict[str, Any]:
        name_lower = file_name.lower()
        meta = {
            'doc_name': file_name,
            'version': '1.0',
            'year': '2026',
            'department': 'General',
            'category': 'Policy'
        }
        
        # Infer year
        for y in ['2024', '2025', '2026', '2027', '2028']:
            if y in name_lower:
                meta['year'] = y
                break
                
        # Infer version
        if '2025' in name_lower or 'v1' in name_lower or 'v1.0' in name_lower:
            meta['version'] = '1.0'
        elif '2026' in name_lower or 'v2' in name_lower or 'v2.0' in name_lower:
            meta['version'] = '2.0'
        elif 'v3' in name_lower or '3.4' in name_lower or '3.1' in name_lower:
            meta['version'] = '3.1'
            
        # Infer Department & Category
        if 'security' in name_lower or 'cyber' in name_lower or 'sr402' in name_lower:
            meta['department'] = 'Information Security & Compliance'
            meta['category'] = 'Regulation'
        elif 'it' in name_lower or 'infra' in name_lower or 'tech' in name_lower:
            meta['department'] = 'Information Technology'
            meta['category'] = 'Technical Manual'
        elif 'hr' in name_lower or 'employee' in name_lower or 'operations' in name_lower:
            meta['department'] = 'Human Resources & Operations'
            meta['category'] = 'Policy'
        elif 'audit' in name_lower or 'compliance' in name_lower:
            meta['department'] = 'Regulatory Compliance'
            meta['category'] = 'Audit Guidelines'
            
        return meta
