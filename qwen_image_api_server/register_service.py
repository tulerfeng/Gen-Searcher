#!/usr/bin/env python3
"""
Service registration script.
Usage: python register_service.py <service_name> <service_type> <port> [action]
action: register (default), update_status
"""

import sys
import os
import subprocess
import yaml
from datetime import datetime


def get_current_ip():
    """Resolve current host IP address."""
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        return result.stdout.strip().split()[0]
    except:
        return "127.0.0.1"


def register_service(service_name, service_type, port):
    """Register a service entry in the YAML file."""
    # YAML file path
    yaml_file = "services.yaml"
    
    # Current IP
    ip_address = get_current_ip()
    
    # Load existing data
    services = {}
    if os.path.exists(yaml_file):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                services = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to read YAML file: {e}")
            services = {}
    
    # Add / update service entry
    services[service_name] = {
        'ip_address': ip_address,
        'service_type': service_type,
        'port': port,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'running'
    }
    
    # Write YAML file
    os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(services, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Service {service_name} registered")
    print(f"IP address: {ip_address}")
    print(f"Port: {port}")


def update_service_status(service_name, status):
    """Update service status in the remote YAML file."""
    yaml_file = "services.yaml"
    
    # Load existing data
    services = {}
    if os.path.exists(yaml_file):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                services = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to read YAML file: {e}")
            return False
    
    # Update status
    if service_name in services:
        services[service_name]['status'] = status
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(services, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Service {service_name} status updated to: {status}")
        return True
    else:
        print(f"Service {service_name} does not exist")
        return False


def main():
    if len(sys.argv) < 4:
        print("Usage: python register_service.py <service_name> <service_type> <port> [action]")
        sys.exit(1)
    
    service_name = sys.argv[1]
    service_type = sys.argv[2]
    port = int(sys.argv[3])
    action = sys.argv[4] if len(sys.argv) > 4 else 'register'
    
    if action == 'register':
        register_service(service_name, service_type, port)
    elif action == 'update_status':
        status = sys.argv[5] if len(sys.argv) > 5 else 'running'
        update_service_status(service_name, status)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
